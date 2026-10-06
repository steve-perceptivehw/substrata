"""
Fixed 2D torus with a Moore neighborhood (8 neighbors).

Direction k has offset DIRS[k] = (dy, dx). OPP[k] is the opposite direction.
All helpers take flat per-site tensors of shape [N, ...] with N = G * G.
"""

import torch

DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
OPP = [7 - k for k in range(8)]
K = len(DIRS)


def _grid(t: torch.Tensor, G: int) -> torch.Tensor:
    return t.view(G, G, *t.shape[1:])


def gather(t: torch.Tensor, G: int) -> torch.Tensor:
    """out[p, k, ...] = t[p + DIRS[k]]  (what each site sees in each direction)."""
    g = _grid(t, G)
    outs = [torch.roll(g, shifts=(-dy, -dx), dims=(0, 1)) for dy, dx in DIRS]
    return torch.stack(outs, dim=2).reshape(G * G, K, *t.shape[1:])


def send(t: torch.Tensor, G: int) -> torch.Tensor:
    """t[p, k] is sent from p toward p + DIRS[k].
    out[q, k] = t[q - DIRS[k], k]  (what q received from its neighbor in direction OPP[k])."""
    g = t.view(G, G, K, *t.shape[2:])
    outs = [torch.roll(g[:, :, k], shifts=(dy, dx), dims=(0, 1)) for k, (dy, dx) in enumerate(DIRS)]
    return torch.stack(outs, dim=2).reshape(G * G, K, *t.shape[2:])


def neighbor_index(G: int, device) -> torch.Tensor:
    """idx[p, k] = flat index of p + DIRS[k]."""
    ys, xs = torch.meshgrid(torch.arange(G, device=device), torch.arange(G, device=device), indexing="ij")
    cols = [(((ys + dy) % G) * G + (xs + dx) % G).reshape(-1) for dy, dx in DIRS]
    return torch.stack(cols, dim=1)
