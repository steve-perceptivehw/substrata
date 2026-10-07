"""
Observers watch the world; the engine never knows who is watching.

Every viewer (Rerun now, a browser dashboard later) implements the same three
calls, so a new viewer plugs in without touching the engine.
"""

from __future__ import annotations

import csv
import os

import numpy as np
import torch


class Observer:
    def start(self, world, run_dir: str): ...
    def metrics(self, tick: int, m: dict): ...
    def frame(self, tick: int, frames: dict): ...
    def close(self): ...


# ---------------------------------------------------------------- frames
_ANCHORS = np.array([  # perceptual dark-to-bright ramp (inferno-like)
    [0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99],
    [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164]], dtype=np.float32)
EMPTY = np.array([18, 18, 22], dtype=np.uint8)


def colormap(v: np.ndarray) -> np.ndarray:
    v = np.clip(v, 0, 1) * (len(_ANCHORS) - 1)
    i = np.minimum(v.astype(int), len(_ANCHORS) - 2)
    f = (v - i)[..., None]
    return (_ANCHORS[i] * (1 - f) + _ANCHORS[i + 1] * f).astype(np.uint8)


@torch.no_grad()
def render(w, channels) -> dict:
    """Return {channel: HxWx3 uint8}. Empty sites are dark gray."""
    G = w.G
    alive = w.alive.view(G, G).cpu().numpy()
    out = {}

    def scalar(t, lo, hi):
        v = ((t.float() - lo) / max(hi - lo, 1e-9)).view(G, G).cpu().numpy()
        return colormap(v)

    for ch in channels:
        if ch == "energy":
            img = scalar(w.energy, 0, w.cfg.energy.max)
        elif ch == "genome":
            img = (w.genome_color().view(G, G, 3).cpu().numpy() * 255).astype(np.uint8)
        elif ch == "age":
            img = scalar(w.age.float() / w.lifespan.float(), 0, 1)
        elif ch == "active_params":
            img = scalar(w.active / w.p_max, 0, 1)
        elif ch == "messages":
            img = scalar(w.msg.abs().mean(1), 0, 1)
        elif ch == "transfers":
            img = scalar(w.give_out, 0, w.cfg.energy.transfer_rate * w.cfg.energy.max)
        elif ch == "inflow":
            v = w.inflow.view(G, G).cpu().numpy()
            out[ch] = colormap(v / max(float(w.field.base.max()) * w.cfg.substrate.inflow * (1 + w.cfg.substrate.season_amp), 1e-9))
            continue          # inflow is shown everywhere, occupied or not
        elif ch == "bonds":
            img = scalar(w.bond.float().sum(1), 0, 4)
        elif ch == "groups":
            if w.labels is None or not w.cfg.layers.bonds:
                img = np.zeros((G, G, 3), dtype=np.uint8) + 60
            else:
                lab = w.labels.view(G, G).cpu().numpy().astype(np.int64)
                nb = w.bond.any(1).view(G, G).cpu().numpy()
                h = (lab * 2654435761) % (2 ** 32)
                img = np.stack([(h >> 8) & 255, (h >> 16) & 255, (h >> 24) & 255], -1).astype(np.uint8) // 2 + 100
                img[~nb] = 70     # unbonded cells in gray
        elif ch == "generation":
            a = w.alive
            hi = float(w.gen[a].max()) if a.any() else 1.0
            lo = float(w.gen[a].min()) if a.any() else 0.0
            img = scalar(w.gen.float(), lo, hi if hi > lo else lo + 1)
        else:
            continue
        img[~alive] = EMPTY
        out[ch] = img
    return out


# ------------------------------------------------------------------- CSV
class CSVObserver(Observer):
    def start(self, world, run_dir):
        self.path = os.path.join(run_dir, "metrics.csv")
        self.f = None
        self.cols = None

    def metrics(self, tick, m):
        if self.f is None:
            new = not os.path.exists(self.path)
            self.f = open(self.path, "a", newline="")
            self.cols = list(m.keys())
            self.wr = csv.DictWriter(self.f, fieldnames=self.cols, extrasaction="ignore")
            if new:
                self.wr.writeheader()
        self.wr.writerow(m)
        self.f.flush()

    def close(self):
        if self.f:
            self.f.close()


# ----------------------------------------------------------------- Rerun
CHART_GROUPS = {  # each group shares an axis, so keep similar scales together
    "population": ["population"],
    "turnover": ["births_per_tick", "deaths_per_tick"],
    "energy_budget": ["inflow_in", "cost_out", "transfer", "death_loss"],
    "cell_energy": ["energy_mean"],
    "brains": ["active_params_frac", "expressed_hidden_frac"],
    "bandwidth": ["msg_bandwidth"],
    "learning": ["weight_drift"],
    "generations": ["generation_mean"],
    "habitat": ["habitat_match"],
    "bonds": ["bonds_per_cell", "bonded_frac"],
    "groups": ["group_size_mean", "group_size_max"],
    "group_sharing": ["bond_flow_per_cell", "transfer_per_cell"],
    "diversity": ["div_weights", "div_learning", "div_masks"],
}


class RerunObserver(Observer):
    def __init__(self, mode="spawn", port=9090):
        self.mode, self.port = mode, port

    def start(self, world, run_dir):
        import rerun as rr
        import rerun.blueprint as rrb
        self.rr = rr
        chans = list(world.cfg.observe.channels)
        bp = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Grid(*[rrb.Spatial2DView(origin=f"grid/{c}", name=c) for c in chans]),
                rrb.Vertical(*[rrb.TimeSeriesView(origin=f"metrics/{g}", name=g) for g in CHART_GROUPS]),
                column_shares=[3, 2],
            ),
            collapse_panels=True,
        )
        rr.init(f"substrata-{world.cfg.name}", spawn=False)
        if self.mode == "spawn":
            rr.spawn(default_blueprint=bp)
        elif self.mode == "save":
            rr.save(os.path.join(run_dir, "recording.rrd"), default_blueprint=bp)
        elif self.mode == "serve":
            uri = rr.serve_grpc(default_blueprint=bp, server_memory_limit="4GiB")
            rr.serve_web_viewer(web_port=self.port, open_browser=False, connect_to=uri)
            print(f"Rerun web viewer on port {self.port} (open http://<this-pc-ip>:{self.port} from another device)")
        rr.log("config", rr.TextDocument("```toml\n" + _toml(world.cfg.to_dict()) + "\n```",
                                         media_type="text/markdown"), static=True)

    def metrics(self, tick, m):
        rr = self.rr
        rr.set_time("tick", sequence=tick)
        for g, keys in CHART_GROUPS.items():
            for k in keys:
                if k in m:
                    rr.log(f"metrics/{g}/{k}", rr.Scalars(float(m[k])))

    def frame(self, tick, frames):
        rr = self.rr
        rr.set_time("tick", sequence=tick)
        for ch, img in frames.items():
            rr.log(f"grid/{ch}", rr.Image(img, magnification_filter="Nearest"))


def _toml(d, prefix=""):
    lines, tables = [], []
    for k, v in d.items():
        if isinstance(v, dict):
            tables.append((k, v))
        else:
            lines.append(f"{k} = {v!r}".replace("'", '"').replace("True", "true").replace("False", "false"))
    for k, v in tables:
        lines.append(f"\n[{prefix}{k}]")
        lines.append(_toml(v, prefix=f"{prefix}{k}."))
    return "\n".join(lines)
