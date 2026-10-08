"""
The world: a fixed grid of sites, each empty or holding one cell.

One tick:
  1. sense      own state, neighbor messages, neighbor occupancy, own energy, local inflow, group context
  2. think      each cell runs its own two-layer network (genome-masked, expression-gated)
  3. learn      plasticity rule from the genome updates the cell's weights (Layer 1)
  4. energy     inflow in; metabolism, compute and bandwidth costs out; transfers between neighbors
  5. die        energy <= 0 or age past lifespan
  6. reproduce  cells with enough energy that choose to, into an empty neighbor; child genome mutated (Layer 3)

Layer 2 (bonds, when layers.bonds is on): between steps 2 and 4, cells form and break bonds with
neighbors by mutual choice. Energy evens out across bonds (the membrane: sharing reaches only the
group), and each cell's group context is a running average of its bonded neighbors' state, which
feeds back into its network and its gene expression (the downward channel).
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
                      "P": m.sigma_learn, "E": m.sigma_learn, "Eb": m.sigma_learn, "R": m.sigma_learn,
                      "A": m.sigma_learn}

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

        self.genome["A"].fill_(cfg.life.adhesion_init)   # start neutral; evolution decides how sticky to be
        self.alive = torch.rand(N, device=self.dev, generator=self.rng) < cfg.life.initial_density
        self.energy = torch.where(self.alive, cfg.energy.initial, 0.0)
        self.age = self._zeros(N, dtype=torch.int32)
        self.lifespan = self._draw_lifespan(N)
        self.gen = self._zeros(N, dtype=torch.int32)
        self.state = self._zeros(N, self.S)
        self.msg = self._zeros(N, self.M)
        self.group = self._zeros(N, self.GC)          # Layer 2 downward channel (zeros until bonds exist)
        self.give_out = self._zeros(N)                 # last tick's total transfer out (for viewing)
        self.reward = self._zeros(N)                   # energy change vs. the cell's recent average (reward gating)
        self.baseline = self._zeros(N)
        self.bond = torch.zeros(N, K, dtype=torch.bool, device=self.dev)   # bond[p, k]: p is bonded to p + DIRS[k]
        self.labels = None                             # bonded-group label per site (refreshed every group_every ticks)
        self._opp = torch.tensor(grid.OPP, device=self.dev)
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
                   "death_loss": 0.0, "bond_flow": 0.0, "leak": 0.0, "washed": 0,
                   "splits": 0, "attached_births": 0}

    # ------------------------------------------------------------ expression
    def _context(self, idx):
        E = self.cfg.energy
        occ = grid.gather(self.alive.float().unsqueeze(1), self.G).squeeze(-1)[idx].mean(1, keepdim=True)
        age = (self.age[idx].float() / self.lifespan[idx].float()).unsqueeze(1)
        return torch.cat([(self.energy[idx] / E.max).unsqueeze(1), self._sensed_inflow()[idx].unsqueeze(1),
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

    def _sensed_inflow(self):
        return self.inflow if self.cfg.substrate.sense_inflow else torch.zeros_like(self.inflow)

    # ------------------------------------------------------------ plasticity
    def _plastic(self, W, mask, pre, post, P, alive_f, R=None):
        s = self.cfg.layers.eta_scale
        P = torch.tanh(P)
        gain = alive_f
        if R is not None:   # mix plain correlation (gate 0) with reward-driven learning (gate 1); the gene decides
            gate = torch.sigmoid(R)
            gain = alive_f * ((1 - gate) + gate * self.reward)
        eta = (s * P[:, 0] * gain).view(-1, 1, 1)
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
                       self._sensed_inflow().unsqueeze(1), self.group, torch.ones(N, 1, device=self.dev)], dim=1)

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
        o_bond = out[:, o:o + K]

        # 3. learn
        if cfg.layers.plasticity and self.tick % cfg.layers.plasticity_every == 0:
            rg = cfg.layers.reward_gating
            R = self.genome["R"]
            self._plastic(self.W1, m1, x, h, self.genome["P"][:, 0], alive_f, R[:, 0] if rg else None)
            self._plastic(self.W2, m2, h, torch.tanh(out), self.genome["P"][:, 1], alive_f, R[:, 1] if rg else None)

        self.state = new_state * alive_f.unsqueeze(1)
        self.msg = new_msg * alive_f.unsqueeze(1)

        # Layer 2: bonds and the group context
        Ly = cfg.layers
        if Ly.bonds:
            nb_alive = occ > 0.5
            theirs = grid.gather(o_bond, G).gather(2, self._opp.view(1, K, 1).expand(N, K, 1)).squeeze(2)
            if Ly.bond_mode == "mutual":
                form = (o_bond > Ly.bond_form) & (theirs > Ly.bond_form)
            elif Ly.bond_mode in ("birth_only", "colony"):
                form = torch.zeros_like(self.bond)          # new bonds come only from births
            else:
                raise ValueError(f"unknown layers.bond_mode: {Ly.bond_mode}")
            if Ly.bond_mode == "colony":
                brk = torch.zeros_like(self.bond)           # no leaving by choice; only strain splits groups
                if self.tick % cfg.observe.group_every == 0:
                    self._fragment()
            else:
                brk = (o_bond < Ly.bond_break) | (theirs < Ly.bond_break)
            self.bond = (self.bond | form) & ~brk & alive.unsqueeze(1) & nb_alive
            bf = self.bond.float()
            nb_group = grid.gather(self.group, G)                                   # [N, K, GC]
            self.group = ((self.state[:, :self.GC] + (bf.unsqueeze(2) * nb_group).sum(1))
                          / (1 + bf.sum(1, keepdim=True))) * alive_f.unsqueeze(1)
        n_bonds = self.bond.float().sum(1)

        # 4. energy
        e = self.energy + self.inflow * alive_f
        cost = (En.base_cost + En.param_cost_full * self.active / self.p_max
                + En.msg_cost * self.msg.abs().sum(1) + En.bond_cost * n_bonds) * alive_f
        e = e - cost
        if En.exposure_leak > 0:
            # energy escapes through exposed faces; bonded faces are sealed
            leak = En.exposure_leak * e.clamp(min=0) * (K - n_bonds) / K * alive_f
            e = e - leak
            cost = cost + leak
            self.st["leak"] = self.st["leak"] + leak.sum().detach()
        give = torch.sigmoid(o_transfer) * (En.transfer_rate / K) * e.clamp(min=0).unsqueeze(1) * occ
        give = give * alive_f.unsqueeze(1)
        received = grid.send(give, G).sum(1) * (1 - En.transfer_loss)
        give_tot = give.sum(1)
        e = e - give_tot + received
        if Ly.bonds:
            # energy evens out across each bond; flows are equal and opposite, so nothing is created or lost
            e_nb = grid.gather(e.unsqueeze(1), G).squeeze(-1)
            delta = Ly.bond_share * (self.bond.float() * (e_nb - e.unsqueeze(1))).sum(1) / K
            e = e + delta
            self.st["bond_flow"] = self.st["bond_flow"] + delta.clamp(min=0).sum().detach()
        overflow = (e - En.max).clamp(min=0)
        e = e - overflow
        if cfg.layers.reward_gating:
            # reward = this tick's energy change against the cell's own recent average (before any birth costs)
            dE = (e - self.energy) * alive_f
            self.reward = torch.tanh((dE - self.baseline) / cfg.layers.reward_scale) * alive_f
            self.baseline = (0.9 * self.baseline + 0.1 * dE) * alive_f
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
        Sb = cfg.substrate
        if Sb.washout_rate > 0:
            whole, frac = int(Sb.washout_rate), Sb.washout_rate - int(Sb.washout_rate)
            n_ev = whole + int(float(torch.rand(1, device=self.dev, generator=self.rng)) < frac)
            if n_ev:
                ys = torch.arange(N, device=self.dev) // G
                xs = torch.arange(N, device=self.dev) % G
                hit = torch.zeros(N, dtype=torch.bool, device=self.dev)
                for _ in range(n_ev):
                    cy, cx = (torch.randint(0, G, (2,), device=self.dev, generator=self.rng)).tolist()
                    dy = (ys - cy).abs(); dy = torch.minimum(dy, G - dy)
                    dx = (xs - cx).abs(); dx = torch.minimum(dx, G - dx)
                    hit |= (dy * dy + dx * dx) <= Sb.washout_radius ** 2
                p_off = Sb.washout_strength * torch.exp(-Sb.washout_grip * n_bonds)
                washed = hit & alive & ~dying & (torch.rand(N, device=self.dev, generator=self.rng) < p_off)
                dying = dying | washed
                self.st["washed"] = self.st["washed"] + washed.sum()

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
        self.group[dying] = 0
        if Ly.bonds:   # bonds to the dead dissolve
            self.bond &= self.alive.unsqueeze(1) & (grid.gather(self.alive.float().unsqueeze(1), G).squeeze(-1) > 0.5)

        # apply births
        e[q] = child_e
        self.energy = e
        if q.numel():
            self._birth(q, p)
            if Ly.bonds and (Ly.bond_at_birth or Ly.bond_mode in ("birth_only", "colony")):
                bp, bq, bk = p, q, kw[q]
                if Ly.bond_mode == "colony":   # the newborn's own adhesion gene decides whether it stays
                    stay = torch.rand(q.numel(), device=self.dev, generator=self.rng) < torch.sigmoid(self.genome["A"][q, 0])
                    bp, bq, bk = bp[stay], bq[stay], bk[stay]
                    self.st["attached_births"] = self.st["attached_births"] + stay.sum()
                self.bond[bp, bk] = True
                self.bond[bq, self._opp[bk]] = True

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
        self.reward[q] = 0
        self.baseline[q] = 0
        self._express(q)

    # ---------------------------------------------------------------- colony
    @torch.no_grad()
    def _fragment(self):
        """Large groups split under strain: each bond breaks with a chance that grows with its group's size.
        Breaking a bond in a tree-shaped group splits it into two offspring groups."""
        Ly = self.cfg.layers
        if not self.bond.any():
            return
        lab = self.find_groups()
        size = torch.bincount(lab[self.alive], minlength=self.N + 1).float()
        s = size[lab.clamp(max=self.N)]                                     # group size seen by each cell
        p_break = 1 - torch.exp(-Ly.frag_strength * (s / Ly.frag_size) ** 2)
        cut = (torch.rand(self.N, K, device=self.dev, generator=self.rng) < p_break.unsqueeze(1)) & self.bond
        # a bond is one link seen from both ends: cut it if either end drew a break
        mirror = cut[self.nbr, self._opp.view(1, K).expand(self.N, K)]
        cut = cut | mirror
        self.st["splits"] = self.st["splits"] + (cut.sum() // 2)
        self.bond &= ~cut

    # ---------------------------------------------------------------- groups
    @torch.no_grad()
    def find_groups(self, max_iter: int = 1024):
        """Label each living cell with its bonded group (the smallest site index in it)."""
        big = self.N
        lab = torch.where(self.alive, torch.arange(self.N, device=self.dev), torch.full((self.N,), big, device=self.dev))
        for _ in range(max_iter):
            nb = grid.gather(lab.unsqueeze(1).float(), self.G).squeeze(-1).long()
            nb = torch.where(self.bond, nb, torch.full_like(nb, big))
            new = torch.minimum(lab, nb.min(1).values)
            new = torch.where(self.alive, new, lab)
            if torch.equal(new, lab):
                break
            lab = new
        self.labels = lab
        return lab

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
