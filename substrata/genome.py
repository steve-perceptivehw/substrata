"""
Genome layout, random initialization and mutation.

A genome is a dict of per-site tensors (first dim = site). The genome is never
changed by learning; the phenotype (the weights a cell actually uses) starts as a
copy of the genome at birth and is then reshaped by the plasticity rule.

Genes:
  W1g, W2g   initial weights of the two layers
  M1l, M2l   mask logits; a weight exists only where its logit > 0 (structure)
  P          plasticity rule per layer: raw (eta, A, B, C, D)
  E, Eb      expression: which hidden units are switched on, as a function of context
"""

from __future__ import annotations

import torch

LEARNING_GENES = ("P", "E", "Eb")
STRUCTURE_GENES = ("W1g", "W2g", "M1l", "M2l")
ALL_GENES = STRUCTURE_GENES + LEARNING_GENES


def shapes(d_in: int, hidden: int, d_out: int, d_ctx: int) -> dict:
    return {
        "W1g": (d_in, hidden),
        "W2g": (hidden, d_out),
        "M1l": (d_in, hidden),
        "M2l": (hidden, d_out),
        "P": (2, 5),
        "E": (d_ctx, hidden),
        "Eb": (hidden,),
    }


def random(n: int, shp: dict, device, gen: torch.Generator) -> dict:
    def r(*s):
        return torch.randn(n, *s, device=device, generator=gen)

    d_in, hidden = shp["W1g"]
    d_out = shp["W2g"][1]
    d_ctx = shp["E"][0]
    return {
        "W1g": r(*shp["W1g"]) * d_in ** -0.5,
        "W2g": r(*shp["W2g"]) * hidden ** -0.5,
        "M1l": r(*shp["M1l"]),
        "M2l": r(*shp["M2l"]),
        "P": r(*shp["P"]),
        "E": r(*shp["E"]) * d_ctx ** -0.5,
        "Eb": r(*shp["Eb"]),
    }


def mutable_genes(layer3: str) -> tuple:
    if layer3 == "off":
        return ()
    if layer3 == "frozen_learning":
        return STRUCTURE_GENES
    if layer3 == "on":
        return ALL_GENES
    if layer3 == "neutral":
        raise NotImplementedError("Neutral drift control (ladder E) is not built yet.")
    raise ValueError(f"unknown layer3 mode: {layer3}")


def mutate(child: dict, genes: tuple, rate: float, sigma: dict, gen: torch.Generator) -> None:
    """In-place: each gene value is perturbed with probability `rate`."""
    for name in genes:
        t = child[name]
        hit = torch.rand(t.shape, device=t.device, generator=gen) < rate
        noise = torch.randn(t.shape, device=t.device, generator=gen) * sigma[name]
        t.add_(noise * hit)
