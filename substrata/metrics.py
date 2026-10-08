"""
Cheap metrics computed every `metrics_every` ticks. These cover the trial floor
(alive and not trivial) and the raw signals behind the other criteria.
Assembly and causal emergence come later, once there is structure to measure.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def compute(w) -> dict:
    a = w.alive
    n = int(a.sum())
    st = {k: float(v) for k, v in w.st.items()}
    t = max(st["ticks"], 1)
    m = {
        "tick": w.tick,
        "population": n,
        "occupancy": n / w.N,
        "births_per_tick": st["births"] / t,
        "deaths_per_tick": st["deaths"] / t,
        "starved_share": st["starved"] / max(st["deaths"], 1),
        # energy budget, per tick, whole world
        "inflow_in": st["inflow_in"] / t,
        "cost_out": st["cost_out"] / t,
        "transfer": st["transfer"] / t,
        "transit_loss": st["transit_loss"] / t,
        "overflow": st["overflow"] / t,
        "death_loss": st["death_loss"] / t,
        "leak": st.get("leak", 0.0) / t,
        "washed_per_tick": st.get("washed", 0.0) / t,
    }
    if n == 0:
        m.update(extinct=1)
        return m

    g = w.genome
    P = torch.tanh(g["P"][a])
    m.update(
        extinct=0,
        energy_mean=float(w.energy[a].mean()),
        age_mean=float(w.age[a].float().mean()),
        generation_mean=float(w.gen[a].float().mean()),
        generation_max=int(w.gen[a].max()),
        active_params_frac=float((w.active[a] / w.p_max).mean()),
        expressed_hidden_frac=float(w.gate[a].mean()),
        msg_bandwidth=float(w.msg[a].abs().sum(1).mean()),
        transfer_per_cell=st["transfer"] / t / n,
        eta_mean=float((w.cfg.layers.eta_scale * P[:, :, 0]).abs().mean()),
        # genetic diversity: mean per-gene std across the living population
        div_weights=float(g["W1g"][a].reshape(n, -1)[:, w._col_idx].std(0).mean()) if n > 1 else 0.0,
        div_learning=float(g["P"][a].reshape(n, -1).std(0).mean()) if n > 1 else 0.0,
        div_masks=float((g["M1l"][a] > 0).float().reshape(n, -1)[:, w._col_idx].std(0).mean()) if n > 1 else 0.0,
        # how far learning has moved the weights from the genome
        weight_drift=float((w.W1[a] - g["W1g"][a]).abs().mean()),
        # habitat match: inflow where cells live vs. the grid average (1 = no preference)
        habitat_match=float(w.inflow[a].mean() / w.inflow.mean().clamp(min=1e-9)),
        # Layer 2
        bonds_per_cell=float(w.bond[a].float().sum(1).mean()),
        bonded_frac=float((w.bond[a].any(1)).float().mean()),
        bond_flow_per_cell=st["bond_flow"] / t / n,
        reward_gate_mean=float(torch.sigmoid(g["R"][a]).mean()),   # 0 = plain correlation learning, 1 = reward-driven
        adhesion_mean=float(torch.sigmoid(g["A"][a]).mean()),      # chance a newborn stays attached (colony mode)
        splits_per_tick=st.get("splits", 0.0) / t,
        attached_birth_share=st.get("attached_births", 0.0) / max(st["births"], 1),
    )
    if w.cfg.layers.bonds:
        if w.labels is None or w.tick % w.cfg.observe.group_every == 0:
            w.find_groups()
        lab = w.labels[a]
        sizes = torch.bincount(lab, minlength=w.N + 1).float()
        sizes = sizes[sizes > 0]
        m.update(
            group_size_mean=float((sizes ** 2).sum() / sizes.sum()),   # the size of the group a typical cell is in
            group_size_max=int(sizes.max()),
            groups_5plus=int((sizes >= 5).sum()),
            in_groups_5plus=float(sizes[sizes >= 5].sum() / n),
        )
    return m
