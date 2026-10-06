"""
Run a world.

    python -m substrata.run --config configs/trial_a.toml
    python -m substrata.run --config configs/trial_a.toml --set layers.plasticity=false --name trial_a_noplast
    python -m substrata.run --resume runs/trial_a_.../checkpoint.pt
    python -m substrata.run --config configs/trial_a.toml --rerun serve      # watch from another device

Ctrl-C stops cleanly and writes a checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

from . import config as C
from . import metrics as MX
from .observers import CSVObserver, RerunObserver, render
from .world import World

SAVE_KEYS = ["alive", "energy", "age", "lifespan", "gen", "state", "msg", "group", "give_out",
             "W1", "W2", "gate", "active"]   # M1, M2 are rebuilt from the genome


def save_checkpoint(w: World, path: str):
    torch.save({
        "config": w.cfg.to_dict(),
        "tick": w.tick,
        "genome": w.genome,
        "tensors": {k: getattr(w, k) for k in SAVE_KEYS},
        "rng": w.rng.get_state(),
    }, path + ".tmp")
    os.replace(path + ".tmp", path)


def load_checkpoint(path: str, device) -> World:
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = C.from_dict(ck["config"])
    w = World(cfg, device)
    w.genome = {k: v.to(device) for k, v in ck["genome"].items()}
    for k, v in ck["tensors"].items():
        setattr(w, k, v.to(device))
    w.M1 = (w.genome["M1l"] > 0).float()
    w.M2 = (w.genome["M2l"] > 0).float()
    w.tick = ck["tick"]
    w.inflow = w.field.at(w.tick)
    w.rng.set_state(ck["rng"])
    return w


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config value, e.g. layers.plasticity=false")
    ap.add_argument("--name")
    ap.add_argument("--ticks", type=int)
    ap.add_argument("--rerun", choices=["spawn", "save", "serve", "none"])
    ap.add_argument("--resume", help="path to a checkpoint.pt")
    ap.add_argument("--runs-dir", default="runs")
    args = ap.parse_args(argv)

    if args.resume:
        dev = pick_device("auto")
        w = load_checkpoint(args.resume, dev)
        cfg = w.cfg
        run_dir = os.path.dirname(os.path.abspath(args.resume))
    else:
        overrides = list(args.set)
        if args.name:
            overrides.append(f"name={args.name}")
        if args.ticks:
            overrides.append(f"ticks={args.ticks}")
        cfg = C.load(args.config, overrides)
        dev = pick_device(cfg.device)
        torch.manual_seed(cfg.seed)
        w = World(cfg, dev)
        run_dir = os.path.join(args.runs_dir, f"{cfg.name}_{time.strftime('%Y%m%d-%H%M%S')}")
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(cfg.to_dict(), f, indent=2)
    if args.ticks:
        cfg.ticks = args.ticks
    if args.rerun:
        cfg.observe.rerun = args.rerun

    ob = cfg.observe
    observers = [CSVObserver()]
    if ob.rerun != "none":
        observers.append(RerunObserver(ob.rerun, ob.serve_port))
    for o in observers:
        o.start(w, run_dir)

    gpu = torch.cuda.get_device_name(0) if dev.type == "cuda" else "CPU"
    print(f"Substrata | {cfg.name} | {cfg.substrate.grid}x{cfg.substrate.grid} | {gpu} | "
          f"layer3={cfg.layers.layer3} plasticity={cfg.layers.plasticity} | run dir: {run_dir}")

    ckpt = os.path.join(run_dir, "checkpoint.pt")
    t_wall, t_tick = time.perf_counter(), w.tick
    try:
        while w.tick < cfg.ticks:
            w.step()
            if w.tick % ob.metrics_every == 0:
                m = MX.compute(w)
                now = time.perf_counter()
                m["ticks_per_s"] = (w.tick - t_tick) / max(now - t_wall, 1e-9)
                w.reset_stats()
                for o in observers:
                    o.metrics(w.tick, m)
                if w.tick % (ob.metrics_every * 50) == 0:
                    print(f"tick {w.tick:>8}  pop {m['population']:>6}  gen {m.get('generation_mean', 0):7.1f}  "
                          f"E {m.get('energy_mean', 0):5.1f}  active {m.get('active_params_frac', 0):.2f}  "
                          f"{m['ticks_per_s']:.0f} t/s")
                    t_wall, t_tick = now, w.tick
                if m["population"] == 0:
                    print(f"Extinct at tick {w.tick}.")
                    break
            if w.tick % ob.image_every == 0:
                frames = render(w, ob.channels)
                for o in observers:
                    o.frame(w.tick, frames)
            if ob.checkpoint_every and w.tick % ob.checkpoint_every == 0:
                save_checkpoint(w, ckpt)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        save_checkpoint(w, ckpt)
        for o in observers:
            o.close()
        print(f"Checkpoint at tick {w.tick}: {ckpt}")


if __name__ == "__main__":
    main()
