"""
Run configuration. Every number here is an assumption; each one is meant to be
changed from a config file, never by editing engine code.

Load order: dataclass defaults -> TOML file -> command-line overrides.
"""

from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SubstrateCfg:
    grid: int = 128                 # grid is grid x grid sites, torus wraparound
    rewiring: bool = False          # reserved; topology is fixed for now
    inflow: float = 0.5             # mean of the inflow field (energy / site / tick), over space and time
    field: str = "uniform"          # spatial shape: "uniform", "patches", "gradient", "file" (see inflow.py)
    contrast: float = 0.0           # 0 = flat, 1 = poorest sites get almost nothing
    patch_scale: float = 12.0       # typical patch size in sites ("patches")
    field_seed: int = 1             # which random patch layout
    inflow_field_file: str = ""     # .npy (grid x grid) for field = "file"
    season: str = "none"            # "none", "global", "wave", "drift"
    season_period: int = 2000       # ticks per season cycle ("global", "wave")
    season_amp: float = 0.5         # 0 to 1, size of the seasonal swing
    drift_every: int = 50           # ticks per one-site slide of the pattern ("drift")


@dataclass
class CellCfg:
    hidden: int = 32                # maximum hidden units per cell
    msg: int = 8                    # message width (bandwidth)
    state: int = 8                  # hidden state carried tick to tick (memory)
    group_ctx: int = 8              # width of the Layer 2 downward channel
    w_max: float = 4.0              # phenotype weight clamp


@dataclass
class EnergyCfg:
    initial: float = 10.0           # energy of seeded cells
    max: float = 50.0               # storage cap per cell
    base_cost: float = 0.1          # fixed metabolism per tick (state memory, upkeep)
    param_cost_full: float = 0.6    # per-tick cost of a fully expressed network; scales with active params
    msg_cost: float = 0.01          # per unit of |message| sent (bandwidth)
    transfer_rate: float = 0.1      # max fraction of own energy a cell can give away per tick
    transfer_loss: float = 0.05     # fraction lost in transit


@dataclass
class LifeCfg:
    initial_density: float = 0.3    # fraction of sites seeded
    seed_mode: str = "clone"        # "clone": one random genome everywhere; "diverse": independent random genomes
    repro_threshold: float = 20.0   # energy needed to reproduce
    child_fraction: float = 0.5     # share of parent energy given to the child
    lifespan: int = 300             # mean maximum age (ticks); senescence keeps turnover going
    lifespan_jitter: float = 0.2    # +/- fraction


@dataclass
class LayersCfg:
    plasticity: bool = True         # Layer 1 learning within a lifetime
    plasticity_every: int = 1
    eta_scale: float = 0.01         # bound on the genome's learning rate
    expression: str = "context"     # "context" or "fixed"
    express_every: int = 10
    heritable_marks: bool = False   # reserved
    bonds: bool = False             # Layer 2 (next build step)
    layer3: str = "on"              # "off" (no mutation), "frozen_learning", "on", "neutral" (not yet built)


@dataclass
class MutationCfg:
    rate: float = 0.02              # probability each gene is perturbed in a child
    sigma_w: float = 0.1            # weight genes
    sigma_mask: float = 0.5         # mask logits (sign flip = gene switched on/off)
    sigma_learn: float = 0.1        # plasticity and expression genes


@dataclass
class ObserveCfg:
    metrics_every: int = 10
    image_every: int = 100
    channels: list = field(default_factory=lambda: [
        "energy", "genome", "inflow", "age", "active_params", "messages", "transfers", "generation"])
    rerun: str = "spawn"            # "spawn" (local viewer), "save" (.rrd file), "serve" (web viewer on LAN), "none"
    serve_port: int = 9090
    checkpoint_every: int = 20000
    color_scale: float = 1.0        # genome color sensitivity to genetic drift


@dataclass
class Config:
    name: str = "run"
    seed: int = 0
    device: str = "auto"
    ticks: int = 300000
    substrate: SubstrateCfg = field(default_factory=SubstrateCfg)
    cell: CellCfg = field(default_factory=CellCfg)
    energy: EnergyCfg = field(default_factory=EnergyCfg)
    life: LifeCfg = field(default_factory=LifeCfg)
    layers: LayersCfg = field(default_factory=LayersCfg)
    mutation: MutationCfg = field(default_factory=MutationCfg)
    observe: ObserveCfg = field(default_factory=ObserveCfg)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _merge(obj: Any, data: dict, path: str = "") -> None:
    for k, v in data.items():
        if not hasattr(obj, k):
            raise KeyError(f"unknown config key: {path}{k}")
        cur = getattr(obj, k)
        if dataclasses.is_dataclass(cur):
            _merge(cur, v, f"{path}{k}.")
        else:
            setattr(obj, k, type(cur)(v) if cur is not None and not isinstance(cur, list) else v)


def load(path: str | None = None, overrides: list[str] | None = None) -> Config:
    cfg = Config()
    if path:
        with open(path, "rb") as f:
            _merge(cfg, tomllib.load(f))
    for ov in overrides or []:
        key, val = ov.split("=", 1)
        *parents, leaf = key.split(".")
        node = cfg
        for p in parents:
            node = getattr(node, p)
        cur = getattr(node, leaf)
        if isinstance(cur, bool):
            val = val.lower() in ("1", "true", "yes", "on")
        setattr(node, leaf, type(cur)(val))
    return cfg


def from_dict(d: dict) -> Config:
    cfg = Config()
    _merge(cfg, d)
    return cfg
