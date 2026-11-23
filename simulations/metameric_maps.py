"""Exact 12-mode Clements coherence and coincidence maps (D_mn, Gamma_qr, Delta Gamma_qr),
passive preparation unitaries, modular-bridge Fiedler overlap, and representative
Anderson disorder traces (Sec. II.C, III.C, Figs. 3, 5).

Reads data/clements_fsw.json and data/core_networks.json; writes data/metameric_maps.json.
Usage: python -I simulations/metameric_maps.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import (
    DATA_DIR,
    annihilation_sym2,
    clements_layer,
    dephasing_matrix,
    gamma_ss,
    graph_chain,
    graph_modular_bridge,
    graph_small_world,
    hbt_observables,
    lift_sym2,
    rdm_diff_shot,
    rdm_from_rho,
    state_b1,
    state_b2,
    state_cz4,
    sym2_basis,
    two_boson_model,
    wf_to_fock,
)


def _round_mat(mat: np.ndarray, ndigits: int = 8) -> list[list[float]]:
    return [[round(float(x), ndigits) for x in row] for row in mat]


def compute_clements_metameric_maps(rng: np.random.Generator) -> dict:
    """Full 12x12 D_{mn}(d), Gamma_{qr}(d), and Delta Gamma_{qr}(d) maps at d = 0..8 (Fig. 3)."""
    N, D_depths, theta, gamma_layer = 12, 8, 0.44, 0.36
    sigma_phi = np.sqrt(gamma_layer)
    ports_2 = (4, 5)          # 0-based (4, 5) <-> ports (5, 6)
    ports_4 = (4, 5, 6, 7)    # 0-based (4, 5, 6, 7) <-> ports (5, 6, 7, 8)

    pairs, _, n_ops = sym2_basis(N)
    B_ann = annihilation_sym2(N, pairs)
    D2 = len(pairs)
    deph_damp = np.exp(-dephasing_matrix(n_ops, np.full(N, gamma_layer)))

    U1_e, U1_o = clements_layer(N, 0, theta), clements_layer(N, 1, theta)
    U2_e, U2_o = lift_sym2(U1_e, pairs), lift_sym2(U1_o, pairs)
    G_ss = gamma_ss(N)
    off_mask = ~np.eye(N, dtype=bool)

    W_hom, W_np = state_b1(N, *ports_2, noon_sign=+1.0)
    _, W_nm = state_b1(N, *ports_2, noon_sign=-1.0)
    W_adj, W_cz4 = state_cz4(N, *ports_4)

    # Passive preparation unitaries (Sec. II.C):
    # U_DC^(a,b) |1_a, 1_b> = i |Psi_NOON^+>, U_H^(a,b) |1_a, 1_b> = |Psi_NOON^->.
    a0, b0 = ports_2
    U_prep_np = np.eye(N, dtype=np.complex128)
    U_prep_np[np.ix_([a0, b0], [a0, b0])] = np.array([[1.0, 1j], [1j, 1.0]]) / np.sqrt(2.0)
    err_np = float(np.max(np.abs((-1j) * (U_prep_np @ W_hom @ U_prep_np.T) - W_np)))
    assert err_np < 1e-14

    U_prep_nm = np.eye(N, dtype=np.complex128)
    U_prep_nm[np.ix_([a0, b0], [a0, b0])] = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    err_nm = float(np.max(np.abs(U_prep_nm @ W_hom @ U_prep_nm.T - W_nm)))
    assert err_nm < 1e-14

    # B2+/B2- synthesis: U_H^(b,c) diag(1, e^{i phi/2}, 1) U_DC^(a,b) on (4, 5, 6), phi in {0, pi}.
    Wb2_1p, Wb2_2 = state_b2(N, 4, 5, 6, sign=+1.0)
    Wb2_1m, _ = state_b2(N, 4, 5, 6, sign=-1.0)
    U_dc1 = np.eye(N, dtype=np.complex128)
    U_dc1[np.ix_([4, 5], [4, 5])] = np.array([[1.0, 1j], [1j, 1.0]]) / np.sqrt(2.0)
    U_ph_half = np.eye(N, dtype=np.complex128)
    U_ph_half[5, 5] = 1j
    U_h2 = np.eye(N, dtype=np.complex128)
    U_h2[np.ix_([5, 6], [5, 6])] = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    U_b2_p, U_b2_m = U_h2 @ U_dc1, U_h2 @ U_ph_half @ U_dc1
    err_b2_p = float(np.max(np.abs((-1j) * (U_b2_p @ W_hom @ U_b2_p.T) - Wb2_1p)))
    err_b2_m = float(np.max(np.abs((-1j) * (U_b2_m @ W_hom @ U_b2_m.T) - Wb2_1m)))
    err_b2_2 = float(np.max(np.abs(U_h2 @ W_hom @ U_h2.T - Wb2_2)))
    assert err_b2_p < 1e-14 and err_b2_m < 1e-14 and err_b2_2 < 1e-14

    # O(4) switch U_sw = (I_2 (+) U_H^(c,e)) SWAP_{b,c} mapping |Phi_adj> -> |Phi_CZ4>.
    ca, cb, cc, ce = ports_4
    U_swap = np.eye(N, dtype=np.complex128)
    U_swap[cb, cb] = U_swap[cc, cc] = 0.0
    U_swap[cb, cc] = U_swap[cc, cb] = 1.0
    U_H_ce = np.eye(N, dtype=np.complex128)
    U_H_ce[np.ix_([cc, ce], [cc, ce])] = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    U_sw = U_H_ce @ U_swap
    err_cz4 = float(np.max(np.abs(U_sw @ W_adj @ U_sw.T - W_cz4)))
    assert err_cz4 < 1e-14

    def evolve_pair(W_a: np.ndarray, W_b: np.ndarray, snap_depths: list[int]):
        def run_traj(W0: np.ndarray):
            psi0 = wf_to_fock(W0, pairs)
            r = np.outer(psi0, psi0.conj())
            tr = np.empty((D_depths + 1, D2, D2), dtype=np.complex128)
            tr[0] = r
            for d in range(D_depths):
                U2 = U2_e if (d % 2 == 0) else U2_o
                r = deph_damp * (U2 @ r @ U2.conj().T)
                tr[d + 1] = r
            return tr

        tr_a, tr_b = run_traj(W_a), run_traj(W_b)
        _, G_a = hbt_observables(tr_a, n_ops, pairs, N)
        _, G_b = hbt_observables(tr_b, n_ops, pairs, N)
        da_full = np.linalg.norm(G_a - G_ss[None], axis=(1, 2))
        db_full = np.linalg.norm(G_b - G_ss[None], axis=(1, 2))
        da_off = np.linalg.norm((G_a - G_ss[None])[:, off_mask], axis=1)
        db_off = np.linalg.norm((G_b - G_ss[None])[:, off_mask], axis=1)

        snapshots = {}
        max_ens_entry, max_ens_frob = 0.0, 0.0
        for d in snap_depths:
            Da, Db = rdm_from_rho(tr_a[d], B_ann), rdm_from_rho(tr_b[d], B_ann)
            diff_D = Db - Da
            entry_err = float(np.max(np.abs(diff_D)))
            frob_err = float(np.linalg.norm(diff_D, "fro"))
            max_ens_entry = max(max_ens_entry, entry_err)
            max_ens_frob = max(max_ens_frob, frob_err)

            dGamma = G_b[d] - G_a[d]
            snapshots[f"d_{d}"] = {
                "depth": int(d),
                "D_abs": _round_mat(np.abs(Da), 8),
                "D_real": _round_mat(np.real(Da), 8),
                "D_imag": _round_mat(np.imag(Da), 8),
                "I_diag": [round(float(np.real(Da[m, m])), 8) for m in range(N)],
                "max_abs_Delta_D": entry_err,
                "frob_Delta_D": frob_err,
                "Gamma_state1": _round_mat(G_a[d], 8),
                "Gamma_state2": _round_mat(G_b[d], 8),
                "Delta_Gamma_2_minus_1": _round_mat(dGamma, 8),
                "max_abs_Delta_Gamma": round(float(np.max(np.abs(dGamma))), 8),
                "max_abs_Delta_Gamma_offdiag": round(float(np.max(np.abs(dGamma[off_mask]))), 8),
                "d_HBT_state1": round(float(da_full[d]), 6),
                "d_HBT_state2": round(float(db_full[d]), 6),
                "d_neq_state1": round(float(da_off[d]), 6),
                "d_neq_state2": round(float(db_off[d]), 6),
                "Delta_d_neq_2_minus_1": round(float(db_off[d] - da_off[d]), 6),
            }

        max_shot_frob, max_shot_entry = 0.0, 0.0
        for _ in range(30):
            U_cum = np.eye(N, dtype=np.complex128)
            for d in range(D_depths + 1):
                if d > 0:
                    U1 = U1_e if ((d - 1) % 2 == 0) else U1_o
                    ph = np.exp(-1j * rng.normal(0.0, sigma_phi, size=N))
                    U_cum = ph[:, None] * (U1 @ U_cum)
                frob_w = rdm_diff_shot(U_cum, W_a, W_b)
                Wa_o, Wb_o = U_cum @ W_a @ U_cum.T, U_cum @ W_b @ U_cum.T
                D1_w = 2.0 * (Wa_o @ Wa_o.conj().T).conj()
                D2_w = 2.0 * (Wb_o @ Wb_o.conj().T).conj()
                max_shot_frob = max(max_shot_frob, frob_w)
                max_shot_entry = max(max_shot_entry, float(np.max(np.abs(D1_w - D2_w))))

        assert max_ens_entry < 1e-14 and max_shot_entry < 1e-14
        return {
            "depths": list(range(D_depths + 1)),
            "d_HBT_state1": [round(float(x), 6) for x in da_full],
            "d_HBT_state2": [round(float(x), 6) for x in db_full],
            "d_neq_state1": [round(float(x), 6) for x in da_off],
            "d_neq_state2": [round(float(x), 6) for x in db_off],
            "max_ensemble_abs_Delta_D": max_ens_entry,
            "max_ensemble_frob_Delta_D": max_ens_frob,
            "max_shot_by_shot_abs_Delta_D_30masks": max_shot_entry,
            "max_shot_by_shot_frob_Delta_D_30masks": max_shot_frob,
            "snapshots": snapshots,
        }

    hom_noonp = evolve_pair(W_hom, W_np, [0, 1, 2, 3])
    cz4_pair = evolve_pair(W_adj, W_cz4, [0, 2, 3])

    with open(os.path.join(DATA_DIR, "clements_fsw.json"), "r", encoding="utf-8") as f:
        cref = json.load(f)
    ref_hom = cref["clements_12mode_exact"]["families"]["HOM_vs_NOONplus_matched"]
    err_d1 = float(np.max(np.abs(np.array(hom_noonp["d_neq_state1"]) - np.array(ref_hom["d1_offdiag"][:9]))))
    err_d2 = float(np.max(np.abs(np.array(hom_noonp["d_neq_state2"]) - np.array(ref_hom["d2_offdiag"][:9]))))
    assert err_d1 < 1e-5 and err_d2 < 1e-5

    return {
        "N_modes": N,
        "D2": D2,
        "theta_rad": theta,
        "gamma_layer": gamma_layer,
        "sigma_phi_rad": round(float(sigma_phi), 6),
        "Gamma_ss_diag": round(float(G_ss[0, 0]), 8),
        "Gamma_ss_offdiag": round(float(G_ss[0, 1]), 8),
        "passive_prep_verification": {
            "HOM_to_NOONplus_1MZI_max_err": err_np,
            "HOM_to_NOONminus_1MZI_max_err": err_nm,
            "B2_plus_Psi1_UDC_zero_phase_max_err": err_b2_p,
            "B2_minus_Psi1_UDC_pi_over_2_max_err": err_b2_m,
            "B2_plus_Psi2_1splitter_max_err": err_b2_2,
            "B2_CZ4_O4_two_gate_Usw_max_err": err_cz4,
        },
        "cross_check_vs_clements_fsw": {"max_d_neq_err_HOM": err_d1, "max_d_neq_err_NOONplus": err_d2},
        "HOM_vs_NOONplus_aligned_5_6": {
            "state1_label": "HOM", "state2_label": "NOON+",
            "ports_0based": list(ports_2), "ports_1based": [p + 1 for p in ports_2],
            **hom_noonp,
        },
        "B2_CZ4_5_6_7_8": {
            "state1_label": "Phi_adj", "state2_label": "Phi_CZ4",
            "ports_0based": list(ports_4), "ports_1based": [p + 1 for p in ports_4],
            **cz4_pair,
        },
    }


def compute_network_topology_and_disorder(rng: np.random.Generator) -> dict:
    """Modular-bridge Fiedler overlap and ensemble-averaged Anderson disorder trajectories (Fig. 5)."""
    with open(os.path.join(DATA_DIR, "core_networks.json"), "r", encoding="utf-8") as f:
        core = json.load(f)

    J_bridge = 0.15
    adj_mod = graph_modular_bridge(16, J_bridge=J_bridge)
    lap = np.diag(np.sum(adj_mod, axis=1)) - adj_mod
    evals, evecs = np.linalg.eigh(lap)
    fiedler_val = float(evals[1])
    v_F = -evecs[:, 1] if np.mean(evecs[:8, 1]) > 0.0 else evecs[:, 1].copy()

    # Two-octagon layout for the N = 16 modular bridge (ports 8 and 9 face across the bridge).
    R, cx_L, cx_R = 0.96, -1.52, 1.52
    coords = [
        [round(float(cx_L + R * np.cos((7 - i) * np.pi / 4.0)), 6), round(float(R * np.sin((7 - i) * np.pi / 4.0)), 6)]
        for i in range(8)
    ] + [
        [round(float(cx_R + R * np.cos(np.pi - i * np.pi / 4.0)), 6), round(float(R * np.sin(np.pi - i * np.pi / 4.0)), 6)]
        for i in range(8)
    ]
    edges = []
    for i in range(16):
        for j in range(i + 1, 16):
            w = float(adj_mod[i, j])
            if w > 0.0:
                etype = "bridge" if (i, j) == (7, 8) else ("ring" if abs(w - 1.0) < 1e-9 else "chord")
                edges.append({
                    "u_0based": i, "v_0based": j, "u_1based": i + 1, "v_1based": j + 1,
                    "weight": round(w, 4), "type": etype,
                })

    # Inter-module pair correlation <s_1 s_2> = (1/2) sum_{m,n} s_m s_n Gamma_{mn}, s_m = -1 (left), +1 (right).
    s_vec = np.array([-1.0] * 8 + [1.0] * 8, dtype=np.float64)
    W1_br, W2_br = state_b2(16, 6, 7, 8, sign=+1.0)
    G1_br0, G2_br0 = 2.0 * np.abs(W1_br) ** 2, 2.0 * np.abs(W2_br) ** 2
    s1s2_psi1 = float(0.5 * (s_vec @ G1_br0 @ s_vec))
    s1s2_psi2 = float(0.5 * (s_vec @ G2_br0 @ s_vec))
    vf_psi1 = float(0.5 * (v_F @ G1_br0 @ v_F) * 16.0)
    vf_psi2 = float(0.5 * (v_F @ G2_br0 @ v_F) * 16.0)

    mb_core = core["N_scaling_and_topologies"]["modular_bridge_N16"]

    adj_sw16 = graph_small_world(16)
    sw16_core = core["N_scaling_and_topologies"]["small_world_WS_gamma0.45"]["N_16"]
    sw16_coords = [
        [round(float(np.cos(0.5 * np.pi - i * np.pi / 8.0)), 6), round(float(np.sin(0.5 * np.pi - i * np.pi / 8.0)), 6)]
        for i in range(16)
    ]
    sw16_edges = [
        {
            "u_0based": i, "v_0based": j, "u_1based": i + 1, "v_1based": j + 1,
            "weight": round(float(adj_sw16[i, j]), 4),
            "type": "ring" if abs(adj_sw16[i, j] - 1.0) < 1e-9 else "chord",
        }
        for i in range(16) for j in range(i + 1, 16) if adj_sw16[i, j] > 0.0
    ]

    # Ensemble-averaged Anderson disorder trajectories at W/J in {0.0, 1.5, 3.0} on N = 16 chain.
    N_and, J, gamma_0, dz, n_steps = 16, 1.0, 0.45, 0.025, 600
    z_grid = np.linspace(0.0, dz * n_steps, n_steps + 1)
    idx_store = np.arange(0, n_steps + 1, 4)
    G_ss_16 = gamma_ss(N_and)
    W1_a, W2_a = state_b2(N_and, 8, 7, 9, sign=1.0)
    _, _, _, H2_clean, _, M_clean, n_ops_16, pairs_16, _, _ = two_boson_model(
        graph_chain(N_and), J, np.full(N_and, gamma_0)
    )
    D2_16 = len(pairs_16)
    p1_16, p2_16 = wf_to_fock(W1_a, pairs_16), wf_to_fock(W2_a, pairs_16)
    rho1_0, rho2_0 = np.outer(p1_16, p1_16.conj()), np.outer(p2_16, p2_16.conj())

    def batch_rk4_hbt(H2_batch: np.ndarray, rho0: np.ndarray):
        R = H2_batch.shape[0]
        r = np.broadcast_to(rho0[None], (R, D2_16, D2_16)).copy()
        pops_save = np.empty((R, len(idx_store), D2_16), dtype=np.float64)
        pops_save[:, 0] = np.real(np.diagonal(r, axis1=1, axis2=2))
        ptr = 1

        def rhs(x):
            return -1j * (H2_batch @ x - x @ H2_batch) - M_clean[None] * x

        for s in range(1, n_steps + 1):
            k1 = rhs(r)
            k2 = rhs(r + 0.5 * dz * k1)
            k3 = rhs(r + 0.5 * dz * k2)
            k4 = rhs(r + dz * k3)
            r = r + (dz / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            if ptr < len(idx_store) and s == idx_store[ptr]:
                pops_save[:, ptr] = np.real(np.diagonal(r, axis1=1, axis2=2))
                ptr += 1

        I_arr = np.einsum("rtk,jk->rtj", pops_save, n_ops_16)
        G_arr = np.zeros((R, len(idx_store), N_and, N_and), dtype=np.float64)
        for k_idx, (m, n) in enumerate(pairs_16):
            if m == n:
                G_arr[:, :, m, m] += 2.0 * pops_save[:, :, k_idx]
            else:
                G_arr[:, :, m, n] += pops_save[:, :, k_idx]
                G_arr[:, :, n, m] += pops_save[:, :, k_idx]
        return I_arr, G_arr

    rep_curves = {}
    for W_val in [0.0, 1.5, 3.0]:
        n_real = 1 if W_val == 0.0 else 20
        H2_batch = np.broadcast_to(H2_clean[None], (n_real, D2_16, D2_16)).copy()
        if W_val > 0.0:
            eps_mat = rng.uniform(-W_val, W_val, size=(n_real, N_and))
            diag_idx = np.arange(D2_16)
            H2_batch[:, diag_idx, diag_idx] += eps_mat @ n_ops_16
        I1, G1 = batch_rk4_hbt(H2_batch, rho1_0)
        I2, G2 = batch_rk4_hbt(H2_batch, rho2_0)
        max_I_err = float(np.max(np.abs(I1 - I2)))
        assert max_I_err < 1e-13
        d1_all = np.linalg.norm(G1 - G_ss_16[None, None], axis=(2, 3))
        d2_all = np.linalg.norm(G2 - G_ss_16[None, None], axis=(2, 3))
        rep_curves[f"W_{W_val:.1f}"] = {
            "W_over_J": W_val,
            "n_realizations": n_real,
            "max_intensity_diff": max_I_err,
            "Jz": [round(float(x), 4) for x in z_grid[idx_store]],
            "d1_mean": [round(float(x), 6) for x in np.mean(d1_all, axis=0)],
            "d2_mean": [round(float(x), 6) for x in np.mean(d2_all, axis=0)],
            "d1_std": [round(float(x), 6) for x in (np.std(d1_all, axis=0, ddof=1) if n_real > 1 else np.zeros_like(d1_all[0]))],
            "d2_std": [round(float(x), 6) for x in (np.std(d2_all, axis=0, ddof=1) if n_real > 1 else np.zeros_like(d2_all[0]))],
        }

    return {
        "modular_bridge_N16": {
            "N": 16,
            "J_bridge_over_J": J_bridge,
            "adjacency_matrix": _round_mat(adj_mod, 4),
            "edges": edges,
            "node_coords_2d": coords,
            "fiedler_eigenvalue_lambda2": round(fiedler_val, 8),
            "fiedler_eigenvector_vF": [round(float(x), 6) for x in v_F],
            "bridge_B2_ports_0based": [6, 7, 8],
            "bridge_B2_ports_1based": [7, 8, 9],
            "intramodule_B2_ports_0based": [1, 0, 2],
            "intramodule_B2_ports_1based": [2, 1, 3],
            "bridge_intermodule_correlation_s1s2": {
                "Psi1_bunched": round(s1s2_psi1, 6),
                "Psi2_antibunched": round(s1s2_psi2, 6),
                "Psi1_vF_scaled": round(vf_psi1, 6),
                "Psi2_vF_scaled": round(vf_psi2, 6),
            },
            "source_core_networks": {
                "intramodule_gamma0.45": mb_core["intramodule_gamma0.45"],
                "bridge_straddling_gamma0.45": mb_core["bridge_straddling_gamma0.45"],
                "bridge_straddling_gamma1.50": mb_core["bridge_straddling_gamma1.50"],
            },
        },
        "small_world_N16": {
            "N": 16,
            "chord_weight_over_J": 0.85,
            "adjacency_matrix": _round_mat(adj_sw16, 4),
            "edges": sw16_edges,
            "node_coords_2d": sw16_coords,
            "B2_ports_0based": sw16_core["ports_3_0based"],
            "B2_ports_1based": sw16_core["ports_3_1based"],
            "B1_ports_0based": sw16_core["ports_2_0based"],
            "B1_ports_1based": sw16_core["ports_2_1based"],
        },
        "spectral_gaps_from_core": core["spectral_gaps_Delta_cum_vs_Delta_1"],
        "anderson_disorder_N16": {
            "source_core_networks_scan": core["anderson_disorder_N16"],
            "representative_curves": rep_curves,
        },
    }


def main():
    required = ("core_networks.json", "clements_fsw.json")
    missing = [name for name in required if not os.path.isfile(os.path.join(DATA_DIR, name))]
    if missing:
        raise FileNotFoundError(
            f"Missing {', '.join(missing)} in {DATA_DIR}; "
            "run simulations/core_networks.py and simulations/clements_fsw.py first."
        )

    t0 = time.perf_counter()
    seed = 20261124
    rng = np.random.default_rng(seed)

    clements_maps = compute_clements_metameric_maps(rng)
    net_data = compute_network_topology_and_disorder(rng)
    elapsed = time.perf_counter() - t0

    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, "metameric_maps.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {"seed": seed, "runtime_seconds": round(elapsed, 2)},
            "clements_metameric_maps": clements_maps,
            "network_topology_and_disorder": net_data,
        }, f, indent=2)
    print(f"metameric_maps.json ({os.path.getsize(out_path) / 1024:.1f} KB, {elapsed:.2f} s)")


if __name__ == "__main__":
    main()
