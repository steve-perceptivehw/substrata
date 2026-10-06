"""
The inflow field: how much energy arrives at each site on each tick.

The field is always normalized so its average over the grid and over a full
season equals `substrate.inflow`. Changing its shape moves energy around in space
and time without changing the total, so runs with different shapes stay comparable.

Spatial shape (`substrate.field`):
  uniform    every site gets the same
  patches    smooth random rich and poor regions; `patch_scale` sets their size in sites
  gradient   one smooth band of rich to poor across x (a cosine, so it wraps cleanly on the torus)
  file       a grid x grid .npy file (`inflow_field_file`), rescaled to the mean

Contrast (`substrate.contrast`, 0 to 1): 0 is flat; 1 means the poorest sites get nearly nothing.

Time (`substrate.season`):
  none       fixed
  global     the whole field swells and shrinks together:  1 + amp * sin(2 pi t / period)
  wave       a season that travels across x, so rich times arrive at different places at different times
  drift      the spatial pattern itself slides across the grid, one site every `drift_every` ticks

Preview a config's field without running a world:
    python -m substrata.inflow --config configs/trial_b.toml
"""

from __future__ import annotations

import math

import torch


class InflowField:
    def __init__(self, cfg, device):
        s = cfg.substrate
        self.s, self.G, self.dev = s, s.grid, device
        self.base = self._spatial().reshape(-1)                 # mean 1
        ys, xs = torch.meshgrid(torch.arange(self.G, device=device), torch.arange(self.G, device=device),
                                indexing="ij")
        self.x_phase = (xs.float() / self.G).reshape(-1)
        self.static = s.season == "none"
        self._cache = self.base * s.inflow

    # ----------------------------------------------------------- spatial
    def _spatial(self) -> torch.Tensor:
        s, G = self.s, self.G
        c = max(0.0, min(1.0, s.contrast))
        if s.field == "uniform":
            f = torch.ones(G, G, device=self.dev)
        elif s.field == "patches":
            g = torch.Generator(device="cpu").manual_seed(s.field_seed)
            noise = torch.randn(G, G, generator=g)
            # smooth on the torus with a Gaussian filter in Fourier space
            k = torch.fft.fftfreq(G)
            kk = k[:, None] ** 2 + k[None, :] ** 2
            sigma = max(s.patch_scale, 1.0) / 2.0
            z = torch.fft.ifft2(torch.fft.fft2(noise) * torch.exp(-2 * (math.pi * sigma) ** 2 * kk)).real
            z = (z - z.mean()) / (z.std() + 1e-9)
            f = (1 + c * torch.tanh(z * 1.5)).to(self.dev)
        elif s.field == "gradient":
            x = torch.arange(G, device=self.dev).float()
            f = (1 + c * torch.cos(2 * math.pi * x / G)).expand(G, G).clone()
        elif s.field == "file":
            import numpy as np
            f = torch.as_tensor(np.load(s.inflow_field_file), dtype=torch.float32, device=self.dev)
            assert f.shape == (G, G), "inflow field file must be grid x grid"
        else:
            raise ValueError(f"unknown substrate.field: {s.field}")
        f = f.clamp(min=0)
        return f / f.mean()

    # -------------------------------------------------------------- time
    def at(self, tick: int) -> torch.Tensor:
        s = self.s
        if self.static:
            return self._cache
        ph = 2 * math.pi * tick / max(s.season_period, 1)
        if s.season == "global":
            mult = 1 + s.season_amp * math.sin(ph)
            return self.base * (s.inflow * mult)
        if s.season == "wave":
            mult = 1 + s.season_amp * torch.sin(ph - 2 * math.pi * self.x_phase)
            return self.base * mult * s.inflow
        if s.season == "drift":
            shift = (tick // max(s.drift_every, 1)) % self.G
            b = torch.roll(self.base.view(self.G, self.G), shifts=int(shift), dims=1).reshape(-1)
            return b * s.inflow
        raise ValueError(f"unknown substrate.season: {s.season}")


def describe(cfg, ticks=None) -> str:
    """Text summary of a field: spread, and how much of the grid sits below a typical cell's cost."""
    f = InflowField(cfg, torch.device("cpu"))
    E = cfg.energy
    typical = E.base_cost + E.param_cost_full * 0.2 + E.msg_cost * 4      # ~ what an early cell spends
    period = cfg.substrate.season_period if cfg.substrate.season != "none" else 1
    samples = [f.at(t) for t in torch.linspace(0, period, 9).long().tolist()[:-1]] if period > 1 else [f.at(0)]
    allv = torch.stack(samples)
    lines = [
        f"field={cfg.substrate.field} contrast={cfg.substrate.contrast} season={cfg.substrate.season}",
        f"mean inflow {float(allv.mean()):.3f}   min {float(allv.min()):.3f}   max {float(allv.max()):.3f}",
        f"typical early cell cost ~{typical:.3f} per tick",
        f"share of site-ticks below that cost: {float((allv < typical).float().mean()):.0%}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    from . import config as C

    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--png", help="optional path to save a picture of the field at tick 0")
    a = ap.parse_args()
    cfg = C.load(a.config, a.set)
    print(describe(cfg))
    if a.png:
        from .observers import colormap
        import numpy as np
        from PIL import Image
        f = InflowField(cfg, torch.device("cpu")).at(0).view(cfg.substrate.grid, cfg.substrate.grid)
        img = colormap((f / f.max()).numpy())
        Image.fromarray(np.kron(img, np.ones((4, 4, 1), dtype=np.uint8))).save(a.png)
        print(f"saved {a.png}")
