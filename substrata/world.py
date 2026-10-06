"""
The world: a fixed grid of sites, each empty or holding one cell.

One tick:
  1. sense      own state, neighbor messages, neighbor occupancy, own energy, local inflow, group context
  2. think      each cell runs its own two-layer network (genome-masked, expression-gated)
  3. learn      plasticity rule from the genome updates the cell's weights (Layer 1)
  4. energy     inflow in; metabolism, compute and bandwidth costs out; transfers between neighbors
  5. die        energy <= 0 or age past lifespan
  6. reproduce  cells with enough energy that choose to, into an empty neighbor; child genome mutated (Layer 3)

Layer 2 (bonds) is not built yet; the group context channel exists and carries zeros.
"""

from __future__ import annotations

import math

import torch

from . import genome as gn
from . import grid
from .config import Config
from .inflow import InflowField

K = grid.K


class World:
    def __init__(self, cfg: Config, device: torch.device):
        self.cfg = cfg
        self.dev = device
        c = cfg.cell
        self.G = G = cfg.substrate.grid
        self.N = N = G * G
        self.S, self.M, self.H, self.GC = c.state, c.msg, c.hidden, c.group_ctx

        # input:  state | neighbor messages | neighbor occupancy | energy, inflow | group ctx | bias
        self.d_in = self.S + K * self.M + K + 2 + self.GC + 1
        # output: state | message | transfer per neighbor | reproduce | reproduce direction | bond per neighbor
        self.d_out = self.S + self.M + K + 1 + K + K
        # expression context: energy, inflow, occupancy, age | group ctx
        self.d_ctx = 4 + self.GC
        self.p_max = self.d_in * self.H + self.H * self.d_out
        self.shp = gn.shapes(self.d_in, self.H, self.d_out, self.d_ctx)
        self.mutable = gn.mutable_genes(cfg.layers.layer3)
        m = cfg.mutation
        self.sigma = {"W1g": m.sigma_w, "W2g": m.sigma_w, "M1l": m.sigma_mask, "M2l": m.sigma_mask,
                      "P": m.sigma_learn, "E": m.sigma_learn, "Eb": m.sigma_learn}

        self.rng = torch.Generator(device=device).manual_seed(cfg.seed)
        self.nbr = grid.neighbor_index(G, device)
        self.tick = 0
        self._init_population()
        self._init_color_projection()
        self.reset_stats()

    # ------------------------------------------------------------------ setup
    def _zeros(self, *s, dtype=torch.float32):
        return torch.zeros(*s, device=self.dev, dtype=dtype)

    def _init_population(self):
        cfg, N = self.cfg, self.N
        self.field = InflowField(cfg, self.dev)
        self.inflow = self.field.at(0)

        if cfg.life.seed_mode == "clone":
            one = gn.random(1, self.shp, self.dev, self.rng)
            self.genome = {k: v.expand(N, *v.shape[1:]).clone() for k, v in one.items()}
        elif cfg.life.seed_mode == "diverse":
            self.genome = gn.random(N, self.shp, self.dev, self.rng)
        else:
            raise ValueError(cfg.life.seed_mode)

        self.alive = torch.rand(N, device=self.dev, generator=self.rng) < cfg.life.initial_density
        self.energy = torch.where(self.alive, cfg.energy.initial, 0.0)
        self.age = self._zeros(N, dtype=torch.int32)
        self.lifespan = self._draw_lifespan(N)
        self.gen = self._zeros(N, dtype=torch.int32)
        self.state = self._zeros(N, self.S)
        self.msg = self._zeros(N, self.M)
        self.group = self._zeros(N, self.GC)          # Layer 2 downward channel (zeros until bonds exist)
        self.give_out = self._zeros(N)                 # last tick's total transfer out (for viewing)
        self.W1 = self.genome["W1g"].clone()
        self.W2 = self.genome["W2g"].clone()
        self.M1 = (self.genome["M1l"] > 0).float()
        self.M2 = (self.genome["M2l"] > 0).float()
        self.gate = self._zeros(N, self.H)
        self.active = self._zeros(N)
        self._express(torch.arange(N, device=self.dev))

    def _draw_lifespan(self, n):
        L = self.cfg.life
        j = (torch.rand(n, device=self.dev, generator=self.rng) * 2 - 1) * L.lifespan_jitter
        return (L.lifespan * (1 + j)).round().to(torch.int32).clamp(min=1)

    def _init_color_projection(self):
        g = torch.Generator(device=self.dev).manual_seed(12345)
        n_w = self.d_in * self.H
        self._col_idx = torch.randint(0, n_w, (64,), device=self.dev, generator=g)
        self._col_proj = torch.randn(64 + 10 + self.H, 3, device=self.dev, generator=g) / math.sqrt(64 + 10 + self.H)

    def reset_stats(self):
        self.st = {"births": 0, "deaths": 0, "starved": 0, "transfer": 0.0, "ticks": 0,
                   "inflow_in": 0.0, "cost_out": 0.0, "overflow": 0.0, "transit_loss": 0.0,
                   "death_loss": 0.0}

    # ------------------------------------------------------------ expression
    def _context(self, idx):
        E = self.cfg.energy
        occ = grid.gather(self.alive.float().unsqueeze(1), self.G).squeeze(-1)[idx].mean(1, keepdim=True)
        age = (self.age[idx].float() / self.lifespan[idx].float()).unsqueeze(1)
        return torch.cat([(self.energy[idx] / E.max).unsqueeze(1), self.inflow[idx].unsqueeze(1),
                          occ, age, self.group[idx]], dim=1)

    def _express(self, idx):
        """Context-dependent expression: decide which hidden units are switched on."""
        g = self.genome
        if self.cfg.layers.expression == "context":
            ctx = self._context(idx)
            z = torch.bmm(ctx.unsqueeze(1), g["E"][idx]).squeeze(1) + g["Eb"][idx]
        else:
            z = g["Eb"][idx]
        gate = (z > 0).float()
        self.gate[idx] = gate
        a1 = (self.M1[idx] * gate.unsqueeze(1)).sum((1, 2))
        a2 = (self.M2[idx] * gate.unsqueeze(2)).sum((1, 2))
        self.active[idx] = a1 + a2

    # ------------------------------------------------------------ plasticity
    def _plastic(self, W, mask, pre, post, P, alive_f):
        s = self.cfg.layers.eta_scale
        P = torch.tanh(P)
        eta = (s * P[:, 0] * alive_f).view(-1, 1, 1)
        A, B, C, D = (P[:, i].view(-1, 1, 1) for i in range(1, 5))
        pre_, post_ = pre.unsqueeze(2), post.unsqueeze(1)
        W.add_(eta * (A * pre_ * post_ + B * pre_ + C * post_ + D) * mask)
        W.clamp_(-self.cfg.cell.w_max, self.cfg.cell.w_max)

    # ------------------------------------------------------------------ step
    @torch.no_grad()
    def step(self):
        cfg, G, N, S, M = self.cfg, self.G, self.N, self.S, self.M
        En, L = cfg.energy, cfg.life
        alive = self.alive
        alive_f = alive.float()
        if not self.field.static:
            self.inflow = self.field.at(self.tick)

        if self.tick % cfg.layers.express_every == 0:
            self._express(alive.nonzero().squeeze(1))

        # 1. sense
        occ = grid.gather(alive_f.unsqueeze(1), G).squeeze(-1)                    # [N, K]
        nmsg = grid.gather(self.msg, G).reshape(N, K * M)
        x = torch.cat([self.state, nmsg, occ, (self.energy / En.max).unsqueeze(1),
                       self.inflow.unsqueeze(1), self.group, torch.ones(N, 1, device=self.dev)], dim=1)

        # 2. think
        m1 = self.M1 * self.gate.unsqueeze(1)
        m2 = self.M2 * self.gate.unsqueeze(2)
        h = torch.tanh(torch.bmm(x.unsqueeze(1), self.W1 * m1).squeeze(1))
        out = torch.bmm(h.unsqueeze(1), self.W2 * m2).squeeze(1)
        o = 0
        new_state = torch.tanh(out[:, o:o + S]); o += S
        new_msg = torch.tanh(out[:, o:o + M]); o += M
        o_transfer = out[:, o:o + K]; o += K
        o_repro = out[:, o]; o += 1
        o_dir = out[:, o:o + K]; o += K
        # o_bond = out[:, o:o + K]   (Layer 2, next step)

        # 3. learn
        if cfg.layers.plasticity and self.tick % cfg.layers.plasticity_every == 0:
            self._plastic(self.W1, m1, x, h, self.genome["P"][:, 0], alive_f)
            self._plastic(self.W2, m2, h, torch.tanh(out), self.genome["P"][:, 1], alive_f)

        self.state = new_state * alive_f.unsqueeze(1)
        self.msg = new_msg * alive_f.unsqueeze(1)

        # 4. energy
        e = self.energy + self.inflow * alive_f
        cost = (En.base_cost + En.param_cost_full * self.active / self.p_max
                + En.msg_cost * self.msg.abs().sum(1)) * alive_f
        e = e - cost
        give = torch.sigmoid(o_transfer) * (En.transfer_rate / K) * e.clamp(min=0).unsqueeze(1) * occ
        give = give * alive_f.unsqueeze(1)
        received = grid.send(give, G).sum(1) * (1 - En.transfer_loss)
        give_tot = give.sum(1)
        e = e - give_tot + received
        overflow = (e - En.max).clamp(min=0)
        e = e - overflow
        self.give_out = give_tot
        self.st["overflow"] = self.st["overflow"] + (overflow.sum()).detach()
        self.st["transit_loss"] = self.st["transit_loss"] + (give_tot.sum() - received.sum()).detach()

        self.st["inflow_in"] = self.st["inflow_in"] + ((self.inflow * alive_f).sum()).detach()
        self.st["cost_out"] = self.st["cost_out"] + (cost.sum()).detach()
        self.st["transfer"] = self.st["transfer"] + (give_tot.sum()).detach()

        # 5. die
        self.age += alive.int()
        starved = alive & (e <= 0)
        dying = starved | (alive & (self.age >= self.lifespan))

        # 6. reproduce into a neighbor site that was empty at the start of the tick
        empty_nbr = (1 - occ) > 0.5
        wants = (alive & ~dying & (torch.sigmoid(o_repro) > 0.5)
                 & (e >= L.repro_threshold) & empty_nbr.any(1))
        kstar = o_dir.masked_fill(~empty_nbr, -float("inf")).argmax(1)
        claim = torch.nn.functional.one_hot(kstar, K).float() * wants.float().unsqueeze(1)
        prio = torch.rand(N, device=self.dev, generator=self.rng) + 1e-3
        inc = grid.send(claim * prio.unsqueeze(1), G)                             # claims arriving at each site
        best, kw = inc.max(1)
        born = best > 0
        q = born.nonzero().squeeze(1)
        opp = torch.tensor(grid.OPP, device=self.dev)
        p = self.nbr[q, opp[kw[q]]]                                                # parent of each child

        child_e = e[p] * L.child_fraction
        e[p] -= child_e

        # apply deaths (energy left in a dying cell is lost)
        self.alive = alive & ~dying
        self.st["death_loss"] = self.st["death_loss"] + (e[dying].clamp(min=0).sum()).detach()
        e = torch.where(self.alive, e, torch.zeros_like(e))
        self.state[dying] = 0
        self.msg[dying] = 0

        # apply births
        e[q] = child_e
        self.energy = e
        if q.numel():
            self._birth(q, p)

        self.st["births"] += int(q.numel())
        self.st["deaths"] = self.st["deaths"] + dying.sum()
        self.st["starved"] = self.st["starved"] + starved.sum()
        self.st["ticks"] += 1
        self.tick += 1

    def _birth(self, q, p):
        g = self.genome
        child = {k: v[p].clone() for k, v in g.items()}
        if self.mutable:
            gn.mutate(child, self.mutable, self.cfg.mutation.rate, self.sigma, self.rng)
        for k, v in child.items():
            g[k][q] = v
        self.W1[q] = child["W1g"]
        self.W2[q] = child["W2g"]
        self.M1[q] = (child["M1l"] > 0).float()
        self.M2[q] = (child["M2l"] > 0).float()
        self.alive[q] = True
        self.age[q] = 0
        self.lifespan[q] = self._draw_lifespan(q.numel())
        self.gen[q] = self.gen[p] + 1
        self.state[q] = 0
        self.msg[q] = 0
        self._express(q)

    # --------------------------------------------------------------- viewing
    @torch.no_grad()
    def genome_color(self):
        """3-vector per site: a fixed random projection of genes, scaled by the
        population's spread, so similar genomes look similar. A clone is uniform gray."""
        g = self.genome
        feat = torch.cat([g["W1g"].reshape(self.N, -1)[:, self._col_idx], g["P"].reshape(self.N, -1),
                          g["Eb"]], dim=1)
        z = feat @ self._col_proj
        a = self.alive
        if a.any():
            mu, sd = z[a].mean(0), z[a].std(0) if a.sum() > 1 else torch.zeros(3, device=self.dev)
            z = (z - mu) / (sd + 1e-6)
        return 0.5 + 0.5 * torch.tanh(z / 2 * self.cfg.observe.color_scale)

    def total_energy(self):
        return float(self.energy.sum())
