"""
Substrata GPU benchmark: how big can the per-cell networks and messages be,
and how many ticks per second does the 4070 Super deliver at each size?

This times a stand-in for one simulation tick that has the same cost shape as
the real engine will:
  1. gather neighbor messages on a fixed grid (Moore neighborhood, 8 neighbors)
  2. run every cell's own two-layer network (batched, genome-masked weights)
  3. apply a genome-parameterized plasticity rule to every weight
     (generalized Hebbian:  dW = eta * (A*pre*post + B*pre + C*post + D))
  4. energy bookkeeping (inflow, compute cost per active parameter, transfers)

The numbers are random. Nothing here is the model; it only measures cost.

Usage (on the SkyTech, from the repo root):
    python bench/bench_tick.py                 # full sweep, writes bench/results.csv
    python bench/bench_tick.py --quick         # small sweep for a smoke test
    python bench/bench_tick.py --grids 128 --hidden 32 --msg 8
"""

import argparse
import csv
import itertools
import os
import platform
import time

import torch


# --------------------------------------------------------------------------- #
# Sizes held fixed in this benchmark (placeholders, not design decisions)
# --------------------------------------------------------------------------- #
STATE = 16        # per-cell hidden state carried tick to tick
GROUP_CTX = 8     # pooled group state fed back down (Layer 2 downward channel)
N_NEIGH = 8       # Moore neighborhood
ACTIONS = N_NEIGH * 2 + 2  # energy transfer per neighbor, bond per neighbor, reproduce, idle


def make_world(grid, hidden, msg, device, dtype):
    n = grid * grid
    d_in = STATE + N_NEIGH * msg + 2 + GROUP_CTX   # +2: own energy, local inflow
    d_out = STATE + msg + ACTIONS
    g = torch.Generator(device=device).manual_seed(0)

    def rnd(*shape, scale=1.0):
        return torch.randn(*shape, generator=g, device=device, dtype=dtype) * scale

    w = {
        "W1": rnd(n, d_in, hidden, scale=d_in ** -0.5),
        "W2": rnd(n, hidden, d_out, scale=hidden ** -0.5),
        # genome masks: which weights are expressed (about 60% active)
        "M1": (torch.rand(n, d_in, hidden, generator=g, device=device) < 0.6).to(dtype),
        "M2": (torch.rand(n, hidden, d_out, generator=g, device=device) < 0.6).to(dtype),
        # plasticity genome: eta, A, B, C, D per cell per layer
        "P1": rnd(n, 5, scale=0.01),
        "P2": rnd(n, 5, scale=0.01),
        "state": rnd(n, STATE),
        "msg": rnd(n, msg),
        "energy": torch.ones(n, device=device, dtype=dtype),
        "inflow": torch.full((n,), 0.01, device=device, dtype=dtype),
        "group": rnd(n, GROUP_CTX),
    }
    return w, d_in, d_out


def neighbor_messages(msg, grid):
    """Gather the 8 neighbors' outgoing messages with wraparound (torus)."""
    m = msg.view(grid, grid, -1)
    shifts = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    return torch.cat([torch.roll(m, s, dims=(0, 1)) for s in shifts], dim=-1).view(grid * grid, -1)


def plasticity(W, mask, pre, post, P):
    eta, A, B, C, D = (P[:, i].view(-1, 1, 1) for i in range(5))
    pre_ = pre.unsqueeze(2)
    post_ = post.unsqueeze(1)
    dW = eta * (A * pre_ * post_ + B * pre_ + C * post_ + D)
    W.add_(dW * mask).clamp_(-4.0, 4.0)


def tick(w, grid, msg_w):
    n = grid * grid
    x = torch.cat(
        [w["state"], neighbor_messages(w["msg"], grid),
         w["energy"].unsqueeze(1), w["inflow"].unsqueeze(1), w["group"]],
        dim=1,
    )
    h = torch.tanh(torch.bmm(x.unsqueeze(1), w["W1"] * w["M1"]).squeeze(1))
    out = torch.bmm(h.unsqueeze(1), w["W2"] * w["M2"]).squeeze(1)

    w["state"] = torch.tanh(out[:, :STATE])
    w["msg"] = torch.tanh(out[:, STATE:STATE + msg_w])
    actions = out[:, STATE + msg_w:]

    plasticity(w["W1"], w["M1"], x, h, w["P1"])
    plasticity(w["W2"], w["M2"], h, torch.tanh(out), w["P2"])

    # energy: inflow minus compute cost (active params) minus bandwidth cost
    active = w["M1"].sum(dim=(1, 2)) + w["M2"].sum(dim=(1, 2))
    cost = 1e-6 * active + 1e-4 * w["msg"].abs().sum(1)
    give = torch.sigmoid(actions[:, :N_NEIGH]) * 0.01 * w["energy"].unsqueeze(1)
    received = neighbor_messages(give, grid).sum(1)  # stand-in for routing transfers
    w["energy"] = (w["energy"] + w["inflow"] - cost - give.sum(1) + received).clamp_(min=0)

    # group context: stand-in for pooling over bonded groups (3x3 mean)
    g = w["state"][:, :GROUP_CTX].view(1, grid, grid, GROUP_CTX).permute(0, 3, 1, 2)
    g = torch.nn.functional.avg_pool2d(g, 3, stride=1, padding=1, count_include_pad=False)
    w["group"] = g.permute(0, 2, 3, 1).reshape(n, GROUP_CTX)


def run_case(grid, hidden, msg, device, dtype, warmup, iters):
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    w, d_in, d_out = make_world(grid, hidden, msg, device, dtype)
    params_per_cell = d_in * hidden + hidden * d_out
    with torch.no_grad():
        for _ in range(warmup):
            tick(w, grid, msg)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            tick(w, grid, msg)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else float("nan")
    del w
    return {
        "grid": grid, "cells": grid * grid, "hidden": hidden, "msg": msg,
        "params_per_cell": params_per_cell,
        "ticks_per_s": iters / dt, "ms_per_tick": 1000 * dt / iters,
        "peak_gb": peak, "status": "ok",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", type=int, nargs="+", default=[64, 128, 192, 256])
    ap.add_argument("--hidden", type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument("--msg", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    ap.add_argument("--cpu", action="store_true", help="force CPU (for smoke tests only)")
    ap.add_argument("--quick", action="store_true", help="tiny sweep to check it runs")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results.csv"))
    args = ap.parse_args()

    if args.quick:
        args.grids, args.hidden, args.msg, args.iters, args.warmup = [32, 64], [8, 16], [4], 10, 2

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = getattr(torch, args.dtype)
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor() or "cpu"
    print(f"torch {torch.__version__} | device: {gpu} | dtype: {args.dtype}")
    if device.type == "cpu":
        print("WARNING: running on CPU. Numbers are only a smoke test, not the real benchmark.")

    rows = []
    header = f"{'grid':>5} {'cells':>7} {'hid':>4} {'msg':>4} {'params/cell':>11} {'ticks/s':>9} {'ms/tick':>8} {'peak GB':>8}"
    print(header)
    print("-" * len(header))
    for grid, hidden, msg in itertools.product(args.grids, args.hidden, args.msg):
        try:
            r = run_case(grid, hidden, msg, device, dtype, args.warmup, args.iters)
        except torch.cuda.OutOfMemoryError:
            r = {"grid": grid, "cells": grid * grid, "hidden": hidden, "msg": msg,
                 "params_per_cell": "", "ticks_per_s": "", "ms_per_tick": "",
                 "peak_gb": "", "status": "OOM"}
            torch.cuda.empty_cache()
        rows.append(r)
        if r["status"] == "ok":
            print(f"{grid:>5} {grid*grid:>7} {hidden:>4} {msg:>4} {r['params_per_cell']:>11} "
                  f"{r['ticks_per_s']:>9.1f} {r['ms_per_tick']:>8.2f} {r['peak_gb']:>8.2f}")
        else:
            print(f"{grid:>5} {grid*grid:>7} {hidden:>4} {msg:>4} {'':>11} {'OOM':>9}")

    with open(args.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\nWrote {args.out}")
    print("Rule of thumb: ticks/s x 3600 x 8 = ticks in an 8-hour overnight run.")


if __name__ == "__main__":
    main()
