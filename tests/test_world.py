"""Sanity checks for the engine. Run with:  python -m pytest tests -q   (or  python tests/test_world.py)"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from substrata import config as C
from substrata import grid
from substrata.world import World


def small(**over):
    sets = [f"{k}={v}" for k, v in over.items()]
    cfg = C.load("configs/smoke.toml", sets)
    return World(cfg, torch.device("cpu"))


def test_grid_send_gather_are_consistent():
    G = 5
    t = torch.arange(G * G).float().unsqueeze(1)
    seen = grid.gather(t, G)                       # seen[p, k] = t[p + d_k]
    nbr = grid.neighbor_index(G, "cpu")
    assert torch.equal(seen.squeeze(-1), t.squeeze(1)[nbr])
    # sending p -> p + d_k, then looking back from the receiver along OPP[k] finds the sender
    x = torch.zeros(G * G, grid.K)
    x[7, 2] = 1.0
    r = grid.send(x, G)
    q = nbr[7, 2]
    assert r[q, 2] == 1.0 and r.sum() == 1.0
    assert nbr[q, grid.OPP[2]] == 7


def test_energy_is_conserved_except_for_accounted_flows():
    w = small()
    for _ in range(50):
        before = w.total_energy()
        w.reset_stats()
        w.step()
        st = {k: float(v) for k, v in w.st.items()}
        expected = before + st["inflow_in"] - st["cost_out"] - st["transit_loss"] - st["overflow"] - st["death_loss"]
        assert abs(w.total_energy() - expected) < 1e-2 * max(1.0, abs(before)), (w.tick, w.total_energy(), expected)


def test_dead_sites_hold_nothing():
    w = small()
    for _ in range(120):
        w.step()
    dead = ~w.alive
    assert float(w.energy[dead].abs().sum()) == 0
    assert float(w.msg[dead].abs().sum()) == 0


def test_layer3_off_keeps_genome_identical():
    w = small(**{"layers.layer3": "off"})
    for _ in range(200):
        w.step()
    a = w.alive
    P = w.genome["P"][a]
    assert int(w.gen[a].max()) > 0, "expected some reproduction"
    assert torch.allclose(P, P[:1].expand_as(P))


def test_frozen_learning_mutates_only_structure():
    w = small(**{"layers.layer3": "frozen_learning", "mutation.rate": 0.5})
    for _ in range(200):
        w.step()
    a = w.alive
    P, W = w.genome["P"][a], w.genome["W1g"][a]
    assert torch.allclose(P, P[:1].expand_as(P))
    assert not torch.allclose(W, W[:1].expand_as(W))


def test_plasticity_off_leaves_weights_at_genome():
    w = small(**{"layers.plasticity": "false"})
    for _ in range(50):
        w.step()
    a = w.alive
    assert torch.allclose(w.W1[a], w.genome["W1g"][a])


def test_inflow_fields_keep_their_mean():
    from substrata.inflow import InflowField
    for field in ["uniform", "patches", "gradient"]:
        for season in ["none", "global", "wave", "drift"]:
            cfg = C.load("configs/smoke.toml", [f"substrate.field={field}", f"substrate.season={season}",
                                                "substrate.contrast=0.8", "substrate.inflow=0.35",
                                                "substrate.season_period=100"])
            f = InflowField(cfg, torch.device("cpu"))
            mean = torch.stack([f.at(t) for t in range(0, 100, 5)]).mean()
            assert abs(float(mean) - 0.35) < 0.01, (field, season, float(mean))
            assert float(f.at(37).min()) >= 0


def test_energy_conserved_with_moving_seasons():
    w = small(**{"substrate.field": "patches", "substrate.contrast": 0.8, "substrate.season": "wave",
                 "substrate.season_period": 50})
    for _ in range(60):
        before = w.total_energy()
        w.reset_stats()
        w.step()
        st = {k: float(v) for k, v in w.st.items()}
        expected = before + st["inflow_in"] - st["cost_out"] - st["transit_loss"] - st["overflow"] - st["death_loss"]
        assert abs(w.total_energy() - expected) < 1e-2 * max(1.0, abs(before))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
