"""
Are bonded neighbors relatives?

Reads a run's checkpoint and compares how genetically similar cells are when they are
(a) bonded to each other, (b) neighbors but not bonded, and (c) random pairs anywhere.
If bonds mostly join kin, (a) is much closer than (b) and (c): the condition under
which kin selection can protect sharing inside groups.

    python analysis/kinship.py runs/trial_b_bonds_<time>/checkpoint.pt

Genetic distance: mean absolute difference over a fixed sample of weight genes plus all
learning genes (the same genes the genome view is colored by).
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from substrata import grid  # noqa: E402
from substrata.run import load_checkpoint  # noqa: E402


@torch.no_grad()
def genes(w):
    g = w.genome
    return torch.cat([g["W1g"].reshape(w.N, -1)[:, w._col_idx], g["P"].reshape(w.N, -1), g["Eb"]], dim=1)


@torch.no_grad()
def kinship(w, samples=20000):
    a = w.alive
    X = genes(w)
    nbr = grid.neighbor_index(w.G, w.dev)
    both = a.unsqueeze(1) & a[nbr]
    out = {}
    for name, mask in [("bonded neighbors", w.bond & both), ("unbonded neighbors", ~w.bond & both)]:
        p, k = mask.nonzero(as_tuple=True)
        if p.numel() == 0:
            out[name] = (float("nan"), 0)
            continue
        if p.numel() > samples:
            sel = torch.randperm(p.numel(), device=w.dev)[:samples]
            p, k = p[sel], k[sel]
        d = (X[p] - X[nbr[p, k]]).abs().mean(1)
        out[name] = (float(d.mean()), int(mask.sum()) // 2)
    idx = a.nonzero().squeeze(1)
    i = idx[torch.randint(0, idx.numel(), (samples,), device=w.dev)]
    j = idx[torch.randint(0, idx.numel(), (samples,), device=w.dev)]
    out["random pairs"] = (float((X[i] - X[j]).abs().mean(1).mean()), samples)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    w = load_checkpoint(args.checkpoint, dev)
    lab = w.find_groups()
    a = w.alive
    sizes = torch.bincount(lab[a]).float()
    sizes = sizes[sizes > 0]
    print(f"{args.checkpoint}  tick {w.tick}  cells {int(a.sum())}  bonds {int(w.bond.sum()) // 2}  "
          f"groups of 5+: {int((sizes >= 5).sum())}  largest group: {int(sizes.max())}")
    res = kinship(w)
    rnd = res["random pairs"][0]
    print(f"{'pair type':<20} {'genetic distance':>17} {'vs. random':>11} {'pairs':>9}")
    for name, (d, n) in res.items():
        print(f"{name:<20} {d:>17.4f} {d / rnd:>10.0%} {n:>9}")


if __name__ == "__main__":
    main()
