"""Three-photon dephasing cascade (9*gamma : 4*gamma : gamma), simultaneously g^(1)- and g^(2)-blind
pair (|Xi_1>, |Xi_2>), and Bose-Hubbard U/J nonlinearity witness for B2-CZ4 (Sec. VI.A, Fig. 9).

Writes data/higher_order.json.
Usage: python -I simulations/higher_order.py
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time
import numpy as np
from scipy.linalg import expm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import (
    DATA_DIR,
    crossing_stats,
    gamma_ss,
    graph_chain,
    graph_small_world,
    hbt_observables,
    k_boson_model,
    rk4,
    two_boson_model,
)


def run_three_photon_sector(adj: np.ndarray, ports_trio: tuple[int, int, int], label: str):
    """Exact N_ph = 3 dynamics on Sym^3(C^N): 9*gamma : 4*gamma : gamma cascade and (|Xi_1>, |Xi_2>)."""
    N, J, gamma_0 = adj.shape[0], 1.0, 0.45
    H3, M3, n3, tuples3, occ3_to_idx = k_boson_model(adj, J, np.full(N, gamma_0), k_photons=3)
    D3 = len(tuples3)
    unique_rates = sorted({round(float(x / gamma_0), 6) for x in M3.ravel()})

    def occ_idx(counts: dict[int, int]) -> int:
        v = [0] * N
        for m, c in counts.items():
            v[m] = c
        return occ3_to_idx[tuple(v)]

    # One-body operators E_{mn} = b_m^dag b_n on Sym^3(C^N).
    E1_3 = np.zeros((N, N, D3, D3), dtype=np.complex128)
    for col in range(D3):
        occ = n3[:, col]
        for n in range(N):
            if occ[n] == 0:
                continue
            for m in range(N):
                if m == n:
                    E1_3[m, n, col, col] = float(occ[m])
                else:
                    new_occ = list(occ)
                    new_occ[n] -= 1
                    new_occ[m] += 1
                    E1_3[m, n, occ3_to_idx[tuple(new_occ)], col] = np.sqrt(float(occ[n] * (occ[m] + 1)))

    def observables_3ph(traj3: np.ndarray):
        pops = np.real(np.diagonal(traj3, axis1=1, axis2=2))
        I_1p = pops @ n3.T
        D_1p = np.einsum("tab,mnba->tmn", traj3, E1_3)
        T = pops.shape[0]
        G2 = np.zeros((T, N, N), dtype=np.float64)
        G3 = np.zeros((T, N, N, N), dtype=np.float64)
        for k_idx in range(D3):
            occ = n3[:, k_idx]
            pk = pops[:, k_idx]
            nz = np.nonzero(occ)[0]
            for m in nz:
                if occ[m] >= 2:
                    G2[:, m, m] += float(occ[m] * (occ[m] - 1)) * pk
                for r in nz:
                    if r != m:
                        G2[:, m, r] += float(occ[m] * occ[r]) * pk
            for q in nz:
                if occ[q] == 3:
                    G3[:, q, q, q] += 6.0 * pk
                elif occ[q] == 2:
                    for s in nz:
                        if s != q:
                            val = 2.0 * float(occ[s]) * pk
                            G3[:, q, q, s] += val
                            G3[:, q, s, q] += val
                            G3[:, s, q, q] += val
            if len(nz) == 3:
                q, r, s = int(nz[0]), int(nz[1]), int(nz[2])
                for perm in itertools.permutations((q, r, s)):
                    G3[:, perm[0], perm[1], perm[2]] += pk
        return I_1p, D_1p, G2, G3

    _, _, _, G3_ss = observables_3ph(np.eye(D3, dtype=np.complex128)[None] / float(D3))

    # First-order-metameric 3-photon states sharing D(0) = |a><a| + |b><b| + |c><c|:
    # |Phi_9g>, |Phi_4g_exact>, |Phi_sym_mid>, |Phi_1g> = |1_a, 1_b, 1_c>.
    a, b, c = ports_trio
    omega = np.exp(2j * np.pi / 3.0)

    psi_9g = np.zeros(D3, dtype=np.complex128)
    psi_9g[occ_idx({a: 3})] = 1.0 / np.sqrt(3.0)
    psi_9g[occ_idx({b: 3})] = omega / np.sqrt(3.0)
    psi_9g[occ_idx({c: 3})] = (omega**2) / np.sqrt(3.0)

    psi_4g = np.zeros(D3, dtype=np.complex128)
    psi_4g[occ_idx({a: 2, c: 1})] = psi_4g[occ_idx({b: 2, c: 1})] = 1.0 / np.sqrt(2.0)

    psi_mid = np.zeros(D3, dtype=np.complex128)
    psi_mid[occ_idx({a: 2, b: 1})] = psi_mid[occ_idx({b: 2, c: 1})] = psi_mid[occ_idx({c: 2, a: 1})] = 1.0 / np.sqrt(3.0)

    psi_1g = np.zeros(D3, dtype=np.complex128)
    psi_1g[occ_idx({a: 1, b: 1, c: 1})] = 1.0

    psi_coh1g = np.zeros(D3, dtype=np.complex128)
    psi_coh1g[occ_idx({a: 2, b: 1})] = psi_coh1g[occ_idx({a: 1, b: 2})] = 1.0 / np.sqrt(2.0)

    dz, n_steps = 0.02, 300
    z_grid = np.linspace(0.0, dz * n_steps, n_steps + 1)
    tr_9g = rk4(H3, M3, np.outer(psi_9g, psi_9g.conj()), dz, n_steps)
    tr_4g = rk4(H3, M3, np.outer(psi_4g, psi_4g.conj()), dz, n_steps)
    tr_mid = rk4(H3, M3, np.outer(psi_mid, psi_mid.conj()), dz, n_steps)
    tr_1g = rk4(H3, M3, np.outer(psi_1g, psi_1g.conj()), dz, n_steps)
    tr_c1g = rk4(H3, M3, np.outer(psi_coh1g, psi_coh1g.conj()), dz, n_steps)

    i_3a, i_3b = occ_idx({a: 3}), occ_idx({b: 3})
    i_2ac, i_2bc = occ_idx({a: 2, c: 1}), occ_idx({b: 2, c: 1})
    i_2ab, i_1a2b = occ_idx({a: 2, b: 1}), occ_idx({a: 1, b: 2})
    assert abs(M3[i_3a, i_3b] / gamma_0 - 9.0) < 1e-12
    assert abs(M3[i_2ac, i_2bc] / gamma_0 - 4.0) < 1e-12
    assert abs(M3[i_2ab, i_1a2b] / gamma_0 - 1.0) < 1e-12

    coh_9g = np.abs(tr_9g[:, i_3a, i_3b]) / abs(tr_9g[0, i_3a, i_3b])
    coh_4g = np.abs(tr_4g[:, i_2ac, i_2bc]) / abs(tr_4g[0, i_2ac, i_2bc])
    coh_1g = np.abs(tr_c1g[:, i_2ab, i_1a2b]) / abs(tr_c1g[0, i_2ab, i_1a2b])

    _, D_9g, _, G3_9g = observables_3ph(tr_9g)
    _, D_4g, _, G3_4g = observables_3ph(tr_4g)
    _, D_mid, _, G3_mid = observables_3ph(tr_mid)
    _, D_1g, _, G3_1g = observables_3ph(tr_1g)

    max_D_diff = float(max(
        np.max(np.linalg.norm(D_9g - D_1g, axis=(1, 2))),
        np.max(np.linalg.norm(D_4g - D_1g, axis=(1, 2))),
        np.max(np.linalg.norm(D_mid - D_1g, axis=(1, 2))),
    ))
    assert max_D_diff < 1e-12

    d3_9g = np.linalg.norm((G3_9g - G3_ss).reshape(n_steps + 1, -1), axis=1)
    d3_4g = np.linalg.norm((G3_4g - G3_ss).reshape(n_steps + 1, -1), axis=1)
    d3_mid = np.linalg.norm((G3_mid - G3_ss).reshape(n_steps + 1, -1), axis=1)
    d3_1g = np.linalg.norm((G3_1g - G3_ss).reshape(n_steps + 1, -1), axis=1)

    triples = list(itertools.combinations(range(N), 3))

    def off3_dist(G3_t):
        arr = np.stack([G3_t[:, q, r, s] for q, r, s in triples], axis=1)
        ref = np.array([G3_ss[0, q, r, s] for q, r, s in triples])[None, :]
        return np.linalg.norm(arr - ref, axis=1)

    doff3_9g, doff3_4g, doff3_mid, doff3_1g = off3_dist(G3_9g), off3_dist(G3_4g), off3_dist(G3_mid), off3_dist(G3_1g)

    # Simultaneously g^(1)- and g^(2)-blind pair: |Xi_{1,2}> = Sym^3(U_prep) (|3_0> +/- |3_1>) / sqrt(2).
    U1_prep = expm(-1j * (-J * adj) * 0.35)

    def lift_3_mode(U1: np.ndarray, m_src: int) -> np.ndarray:
        vec = np.zeros(D3, dtype=np.complex128)
        u_col = U1[:, m_src]
        for k_idx in range(D3):
            occ = n3[:, k_idx]
            denom, prod_u = 1.0, 1.0 + 0.0j
            for q in np.nonzero(occ)[0]:
                cq = int(occ[q])
                denom *= 2.0 if cq == 2 else (6.0 if cq == 3 else 1.0)
                prod_u *= u_col[q] ** cq
            vec[k_idx] = np.sqrt(6.0 / denom) * prod_u
        return vec

    v3_0, v3_1 = lift_3_mode(U1_prep, 0), lift_3_mode(U1_prep, 1)
    psi_xi1, psi_xi2 = (v3_0 + v3_1) / np.sqrt(2.0), (v3_0 - v3_1) / np.sqrt(2.0)
    _, D_xi1, G2_xi1, G3_xi1 = observables_3ph(rk4(H3, M3, np.outer(psi_xi1, psi_xi1.conj()), dz, n_steps))
    _, D_xi2, G2_xi2, G3_xi2 = observables_3ph(rk4(H3, M3, np.outer(psi_xi2, psi_xi2.conj()), dz, n_steps))
    max_D_xi = float(np.max(np.linalg.norm(D_xi1 - D_xi2, axis=(1, 2))))
    max_G2_xi = float(np.max(np.linalg.norm(G2_xi1 - G2_xi2, axis=(1, 2))))
    assert max_D_xi < 1e-12 and max_G2_xi < 1e-12

    d3_xi1 = np.linalg.norm((G3_xi1 - G3_ss).reshape(n_steps + 1, -1), axis=1)
    d3_xi2 = np.linalg.norm((G3_xi2 - G3_ss).reshape(n_steps + 1, -1), axis=1)
    doff3_xi1, doff3_xi2 = off3_dist(G3_xi1), off3_dist(G3_xi2)

    idx_s = np.arange(0, n_steps + 1, 2)
    return {
        "label": label,
        "N": N,
        "D3": D3,
        "gamma_over_J": gamma_0,
        "ports_trio_0based": list(ports_trio),
        "ports_trio_1based": [p + 1 for p in ports_trio],
        "M_deph_distinct_rates_over_gamma": unique_rates,
        "cascade_g1_blindness_max_D_Frob_diff": max_D_diff,
        "coherence_decay_cascade_9g_4g_1g": {
            "Jz_grid": [round(float(x), 4) for x in z_grid[:101]],
            "exact_pure_deph_exp_9g": [round(float(np.exp(-9.0 * gamma_0 * x)), 6) for x in z_grid[:101]],
            "exact_pure_deph_exp_4g": [round(float(np.exp(-4.0 * gamma_0 * x)), 6) for x in z_grid[:101]],
            "exact_pure_deph_exp_1g": [round(float(np.exp(-gamma_0 * x)), 6) for x in z_grid[:101]],
            "coupled_network_coh_9g": [round(float(x), 6) for x in coh_9g[:101]],
            "coupled_network_coh_4g": [round(float(x), 6) for x in coh_4g[:101]],
            "coupled_network_coh_1g": [round(float(x), 6) for x in coh_1g[:101]],
        },
        "cascade_trio_crossings": {
            "full_G3_9g_vs_4g_exact": crossing_stats(d3_9g, d3_4g, z_grid),
            "full_G3_9g_vs_sym_mid": crossing_stats(d3_9g, d3_mid, z_grid),
            "full_G3_9g_vs_1g": crossing_stats(d3_9g, d3_1g, z_grid),
            "full_G3_4g_exact_vs_1g": crossing_stats(d3_4g, d3_1g, z_grid),
            "full_G3_sym_mid_vs_1g": crossing_stats(d3_mid, d3_1g, z_grid),
            "distinct_3click_1g_vs_9g": crossing_stats(doff3_1g, doff3_9g, z_grid),
            "distinct_3click_1g_vs_4g_exact": crossing_stats(doff3_1g, doff3_4g, z_grid),
            "distinct_3click_1g_vs_sym_mid": crossing_stats(doff3_1g, doff3_mid, z_grid),
        },
        "cascade_trajectories": {
            "Jz_grid": [round(float(x), 4) for x in z_grid[idx_s]],
            "d3_9g": [round(float(x), 6) for x in d3_9g[idx_s]],
            "d3_4g_exact": [round(float(x), 6) for x in d3_4g[idx_s]],
            "d3_sym_mid": [round(float(x), 6) for x in d3_mid[idx_s]],
            "d3_1g": [round(float(x), 6) for x in d3_1g[idx_s]],
            "doff3_9g": [round(float(x), 6) for x in doff3_9g[idx_s]],
            "doff3_4g_exact": [round(float(x), 6) for x in doff3_4g[idx_s]],
            "doff3_1g": [round(float(x), 6) for x in doff3_1g[idx_s]],
        },
        "g1_and_g2_blind_pair_Xi1_Xi2": {
            "prep_z_J": 0.35,
            "max_1body_RDM_Frob_diff_all_z": max_D_xi,
            "max_2body_HBT_Frob_diff_all_z": max_G2_xi,
            "full_G3_crossing": crossing_stats(d3_xi1, d3_xi2, z_grid),
            "distinct_3click_crossing": crossing_stats(doff3_xi1, doff3_xi2, z_grid),
            "trajectories": {
                "Jz_grid": [round(float(x), 4) for x in z_grid[idx_s]],
                "d3_Xi1": [round(float(x), 6) for x in d3_xi1[idx_s]],
                "d3_Xi2": [round(float(x), 6) for x in d3_xi2[idx_s]],
                "doff3_Xi1": [round(float(x), 6) for x in doff3_xi1[idx_s]],
                "doff3_Xi2": [round(float(x), 6) for x in doff3_xi2[idx_s]],
            },
        },
    }


def run_bose_hubbard_witness_scan():
    """Bose-Hubbard U/J in [0, 25] nonlinearity witness ||Delta D(t)||_F on N = 8 chain (Eq. (16), Fig. 9c)."""
    N, J, gamma_0 = 8, 1.0, 0.45
    adj = graph_chain(N)
    off_mask = ~np.eye(N, dtype=bool)
    G_ss_sym = gamma_ss(N)
    G_ss_hc = np.full((N, N), 2.0 / (N * (N - 1)), dtype=np.float64)
    np.fill_diagonal(G_ss_hc, 0.0)

    U_grid = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 18.0, 25.0]
    store_U = {0.0, 0.3, 1.0, 5.0, 25.0}
    scan, trajectories = [], {}

    for U_val in U_grid:
        _, _, _, H2, _, M2, n2, pairs, idx_map, B_ann = two_boson_model(adj, J, np.full(N, gamma_0), U=U_val * J)
        D2 = len(pairs)

        psi_adj = np.zeros(D2, dtype=np.complex128)
        psi_adj[idx_map[(2, 3)]] = psi_adj[idx_map[(4, 5)]] = 1.0 / np.sqrt(2.0)

        psi_cross = np.zeros(D2, dtype=np.complex128)
        psi_cross[idx_map[(2, 4)]] = psi_cross[idx_map[(3, 5)]] = 1.0 / np.sqrt(2.0)

        psi_cz4 = np.zeros(D2, dtype=np.complex128)
        psi_cz4[idx_map[(2, 4)]] = psi_cz4[idx_map[(2, 5)]] = psi_cz4[idx_map[(3, 4)]] = 0.5
        psi_cz4[idx_map[(3, 5)]] = -0.5

        psi_b2_1 = np.zeros(D2, dtype=np.complex128)
        psi_b2_1[idx_map[(4, 4)]] = 1.0 / np.sqrt(2.0)
        psi_b2_1[idx_map[(3, 3)]] = psi_b2_1[idx_map[(5, 5)]] = 0.5 / np.sqrt(2.0)
        psi_b2_1[idx_map[(3, 5)]] = 0.5
        psi_b2_2 = np.zeros(D2, dtype=np.complex128)
        psi_b2_2[idx_map[(4, 3)]] = psi_b2_2[idx_map[(4, 5)]] = 1.0 / np.sqrt(2.0)

        dt, stride = (0.0025, 20) if U_val >= 12.0 else ((0.005, 10) if U_val >= 5.0 else ((0.01, 5) if U_val >= 2.0 else (0.025, 2)))
        t_max = 10.0
        n_st = int(round(t_max / dt))
        t_grid = np.linspace(0.0, t_max, n_st // stride + 1)

        tr_adj = rk4(H2, M2, np.outer(psi_adj, psi_adj.conj()), dt, n_st)[::stride]
        tr_crs = rk4(H2, M2, np.outer(psi_cross, psi_cross.conj()), dt, n_st)[::stride]
        tr_cz4 = rk4(H2, M2, np.outer(psi_cz4, psi_cz4.conj()), dt, n_st)[::stride]
        tr_b2_1 = rk4(H2, M2, np.outer(psi_b2_1, psi_b2_1.conj()), dt, n_st)[::stride]
        tr_b2_2 = rk4(H2, M2, np.outer(psi_b2_2, psi_b2_2.conj()), dt, n_st)[::stride]

        B_dag_B = np.array([[B_ann[m].conj().T @ B_ann[n] for n in range(N)] for m in range(N)])
        D_adj = np.einsum("tab,mnba->tmn", tr_adj, B_dag_B)
        D_crs = np.einsum("tab,mnba->tmn", tr_crs, B_dag_B)
        D_cz4 = np.einsum("tab,mnba->tmn", tr_cz4, B_dag_B)
        D_b2_1 = np.einsum("tab,mnba->tmn", tr_b2_1, B_dag_B)
        D_b2_2 = np.einsum("tab,mnba->tmn", tr_b2_2, B_dag_B)

        leak_cz4 = np.linalg.norm(D_adj - D_cz4, axis=(1, 2))
        leak_crs = np.linalg.norm(D_adj - D_crs, axis=(1, 2))
        leak_b2 = np.linalg.norm(D_b2_1 - D_b2_2, axis=(1, 2))

        I_adj, G_adj = hbt_observables(tr_adj, n2, pairs, N)
        I_cz4, G_cz4 = hbt_observables(tr_cz4, n2, pairs, N)
        G_ref = G_ss_sym if U_val < 10.0 else G_ss_hc
        d_adj_off = np.linalg.norm((G_adj - G_ref[None])[:, off_mask], axis=1)
        d_cz4_off = np.linalg.norm((G_cz4 - G_ref[None])[:, off_mask], axis=1)

        scan.append({
            "U_over_J": U_val,
            "initial_RDM_diff_Frob": float(leak_cz4[0]),
            "max_RDM_leakage_B2_CZ4_Frob": round(float(np.max(leak_cz4)), 6),
            "Jt_at_max_leakage_B2_CZ4": round(float(t_grid[int(np.argmax(leak_cz4))]), 4),
            "max_local_intensity_leakage_B2_CZ4": round(float(np.max(np.abs(I_adj - I_cz4))), 6),
            "max_RDM_leakage_adj_vs_cross_Frob": round(float(np.max(leak_crs)), 6),
            "max_RDM_leakage_B2_3port_Frob": round(float(np.max(leak_b2)), 6),
            "reference_plateau_regime": "Sym2_uniform" if U_val < 10.0 else "hard_core_uniform",
            "B2_CZ4_offdiag_crossing": crossing_stats(d_adj_off, d_cz4_off, t_grid),
        })
        if U_val in store_U:
            trajectories[f"U_{U_val:.1f}"] = {
                "U_over_J": U_val,
                "Jt_grid": [round(float(x), 4) for x in t_grid],
                "delta_D_Frob_B2_CZ4": [round(float(x), 6) for x in leak_cz4],
                "delta_D_Frob_adj_vs_cross": [round(float(x), 6) for x in leak_crs],
                "d_offdiag_Phi_adj": [round(float(x), 6) for x in d_adj_off],
                "d_offdiag_Phi_CZ4": [round(float(x), 6) for x in d_cz4_off],
            }

    assert scan[0]["max_RDM_leakage_B2_CZ4_Frob"] < 1e-12
    return {
        "N": N,
        "topology": "open_1D_chain",
        "ports_B2_CZ4_0based": [2, 3, 4, 5],
        "ports_B2_CZ4_1based": [3, 4, 5, 6],
        "gamma_over_J": gamma_0,
        "Jt_max": 10.0,
        "scan_by_U_over_J": scan,
        "representative_trajectories": trajectories,
    }


def main():
    t0 = time.perf_counter()
    res_3ph_8 = run_three_photon_sector(graph_small_world(8), (0, 1, 3), "small_world_N8")
    res_3ph_6 = run_three_photon_sector(graph_chain(6), (1, 2, 3), "chain_1D_N6")
    res_bh = run_bose_hubbard_witness_scan()
    elapsed = time.perf_counter() - t0

    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, "higher_order.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {"seed": None, "deterministic": True, "runtime_seconds": round(elapsed, 2)},
            "three_photon_N8_D120": res_3ph_8,
            "three_photon_N6_D56": res_3ph_6,
            "bose_hubbard_U_scan_B2_CZ4": res_bh,
        }, f, indent=2)
    print(f"higher_order.json ({os.path.getsize(out_path) / 1024:.1f} KB, {elapsed:.2f} s)")


if __name__ == "__main__":
    main()
