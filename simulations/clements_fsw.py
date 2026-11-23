"""Clements MZI mesh Kraus evolution, finite-ensemble SNSPD shot-noise Monte Carlo,
coupling-angle sweep, and continuous-z segmented-waveguide scan (Sec. IV.A, Figs. 1, 6, Table II).

Writes data/clements_fsw.json.
Usage: python -I simulations/clements_fsw.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import (  # noqa: E402
    DATA_DIR,
    clements_layer,
    crossing_stats,
    dephasing_matrix,
    gamma_ss,
    graph_small_world,
    hbt_observables,
    lift_sym2,
    rdm_diff_shot,
    rk4,
    state_b1,
    state_b2,
    state_cz4,
    sym2_basis,
    two_boson_model,
    wf_to_fock,
)


def run_clements_exact_mesh(
    N_modes: int,
    D_depths: int,
    theta: float,
    gamma_layer: float,
    matched_ports: tuple[int, int],
    mismatched_ports: tuple[int, int],
    b2_ports: tuple[int, int, int],
    b2_ports_aligned: tuple[int, int, int],
    cz4_ports: tuple[int, int, int, int],
    rng: np.random.Generator,
):
    """Exact discrete Kraus evolution on Sym^2(C^N) and single-realization g^(1)-invisibility check."""
    pairs, _, n_ops = sym2_basis(N_modes)
    D2 = len(pairs)
    deph_damp = np.exp(-dephasing_matrix(n_ops, np.full(N_modes, gamma_layer)))

    U1_e, U1_o = clements_layer(N_modes, 0, theta), clements_layer(N_modes, 1, theta)
    U2_e, U2_o = lift_sym2(U1_e, pairs), lift_sym2(U1_o, pairs)

    G_ss = gamma_ss(N_modes)
    off_mask = ~np.eye(N_modes, dtype=bool)
    d_grid = np.arange(D_depths + 1, dtype=np.float64)

    def evolve_pair(W_a: np.ndarray, W_b: np.ndarray):
        def run_one(W0):
            psi0 = wf_to_fock(W0, pairs)
            r = np.outer(psi0, psi0.conj())
            tr = np.empty((D_depths + 1, D2, D2), dtype=np.complex128)
            tr[0] = r
            for d in range(D_depths):
                U2 = U2_e if (d % 2 == 0) else U2_o
                r = deph_damp * (U2 @ r @ U2.conj().T)
                tr[d + 1] = r
            return hbt_observables(tr, n_ops, pairs, N_modes)[1]

        Ga, Gb = run_one(W_a), run_one(W_b)
        da_full = np.linalg.norm(Ga - G_ss[None], axis=(1, 2))
        db_full = np.linalg.norm(Gb - G_ss[None], axis=(1, 2))
        da_off = np.linalg.norm((Ga - G_ss[None])[:, off_mask], axis=1)
        db_off = np.linalg.norm((Gb - G_ss[None])[:, off_mask], axis=1)

        # Theorem 2 check over 15 independent Gaussian phase-mask realizations (phi_j ~ N(0, gamma_layer)).
        sigma_phi = np.sqrt(gamma_layer)
        max_rdm_frob = 0.0
        for _ in range(15):
            U_cum = np.eye(N_modes, dtype=np.complex128)
            max_rdm_frob = max(max_rdm_frob, rdm_diff_shot(U_cum, W_a, W_b))
            for d in range(D_depths):
                U1 = U1_e if (d % 2 == 0) else U1_o
                ph = np.exp(-1j * rng.normal(0.0, sigma_phi, size=N_modes))
                U_cum = ph[:, None] * (U1 @ U_cum)
                max_rdm_frob = max(max_rdm_frob, rdm_diff_shot(U_cum, W_a, W_b))

        return {
            "shot_by_shot_max_RDM_Frob_diff": max_rdm_frob,
            "d1_full_HBT": [round(float(x), 6) for x in da_full],
            "d2_full_HBT": [round(float(x), 6) for x in db_full],
            "d1_offdiag": [round(float(x), 6) for x in da_off],
            "d2_offdiag": [round(float(x), 6) for x in db_off],
            "crossing_full_HBT": crossing_stats(da_full, db_full, d_grid),
            "crossing_offdiag": crossing_stats(da_off, db_off, d_grid),
        }

    Wh_m, Wnp_m = state_b1(N_modes, *matched_ports, noon_sign=+1.0)
    _, Wnm_m = state_b1(N_modes, *matched_ports, noon_sign=-1.0)
    Wh_mm, Wnp_mm = state_b1(N_modes, *mismatched_ports, noon_sign=+1.0)
    _, Wnm_mm = state_b1(N_modes, *mismatched_ports, noon_sign=-1.0)
    Wb2p, Wb2_ref = state_b2(N_modes, *b2_ports, sign=+1.0)
    Wb2m, _ = state_b2(N_modes, *b2_ports, sign=-1.0)
    Wb2p_al, Wb2_ref_al = state_b2(N_modes, *b2_ports_aligned, sign=+1.0)
    Wcz_adj, Wcz_graph = state_cz4(N_modes, *cz4_ports)

    # Passive preparation check for B2+: U_H^(b,c) U_DC^(a,b) |1_a, 1_b> = i |Psi_1^+>.
    a3, b3, c3 = b2_ports
    U_dc1 = np.eye(N_modes, dtype=np.complex128)
    U_dc1[np.ix_([a3, b3], [a3, b3])] = np.array([[1.0, 1j], [1j, 1.0]]) / np.sqrt(2.0)
    U_bs2 = np.eye(N_modes, dtype=np.complex128)
    U_bs2[np.ix_([b3, c3], [b3, c3])] = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    W_in_ab, _ = state_b1(N_modes, a3, b3, -1.0)
    U_p1 = U_bs2 @ U_dc1
    prep_err_1 = float(np.max(np.abs((U_p1 @ W_in_ab @ U_p1.T) / 1j - Wb2p)))
    prep_err_2 = float(np.max(np.abs(U_bs2 @ W_in_ab @ U_bs2.T - Wb2_ref)))
    assert prep_err_1 < 1e-14 and prep_err_2 < 1e-14

    # Passive U(4) switch mapping |Phi_adj> -> |Phi_CZ4> on cz4_ports (a, b, c, e).
    ca, cb, cc, ce = cz4_ports
    U_diag = np.eye(N_modes, dtype=np.complex128)
    U_diag[cc, cc] = U_diag[ce, ce] = 1j
    H2x2 = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / np.sqrt(2.0)
    U_ac_be = np.eye(N_modes, dtype=np.complex128)
    U_ac_be[np.ix_([ca, cc], [ca, cc])] = H2x2
    U_ac_be[np.ix_([cb, ce], [cb, ce])] = H2x2
    U_ce = np.eye(N_modes, dtype=np.complex128)
    U_ce[np.ix_([cc, ce], [cc, ce])] = np.array([[1.0, 1.0], [-1.0, 1.0]], dtype=np.complex128) / np.sqrt(2.0)
    U_sw = U_ce @ U_ac_be @ U_diag
    cz4_u_sw_err = float(np.max(np.abs(U_sw @ Wcz_adj @ U_sw.T - Wcz_graph)))
    assert cz4_u_sw_err < 1e-14

    res_cz4 = evolve_pair(Wcz_adj, Wcz_graph)
    dA_off = np.array(res_cz4["d1_offdiag"])
    dB_off = np.array(res_cz4["d2_offdiag"])
    res_cz4["regime_by_depth"] = [
        {"d": 0, "regime": "input", "gap_dB_minus_dA": round(float(dB_off[0] - dA_off[0]), 6)},
        {
            "d": 1,
            "regime": "unitary_phase_insensitive",
            "gap_dB_minus_dA": round(float(dB_off[1] - dA_off[1]), 6),
            "rel_gap_pct": round(100.0 * float(dB_off[1] - dA_off[1]) / float(dB_off[1]), 2),
        },
    ] + [
        {
            "d": int(d),
            "regime": "dissipative_dephasing",
            "gap_dB_minus_dA": round(float(dB_off[d] - dA_off[d]), 6),
            "rel_gap_pct": round(100.0 * float(dB_off[d] - dA_off[d]) / max(float(dB_off[d]), 1e-12), 2),
        }
        for d in range(2, D_depths + 1)
    ]

    families = {
        "HOM_vs_NOONplus_matched": {
            "ports_0based": list(matched_ports), "ports_1based": [p + 1 for p in matched_ports],
            "alignment": "layer0_matched", **evolve_pair(Wh_m, Wnp_m),
        },
        "HOM_vs_NOONminus_matched": {
            "ports_0based": list(matched_ports), "ports_1based": [p + 1 for p in matched_ports],
            "alignment": "layer0_matched", **evolve_pair(Wh_m, Wnm_m),
        },
        "HOM_vs_NOONplus_mismatched": {
            "ports_0based": list(mismatched_ports), "ports_1based": [p + 1 for p in mismatched_ports],
            "alignment": "odd_layer_mismatched", **evolve_pair(Wh_mm, Wnp_mm),
        },
        "HOM_vs_NOONminus_mismatched": {
            "ports_0based": list(mismatched_ports), "ports_1based": [p + 1 for p in mismatched_ports],
            "alignment": "odd_layer_mismatched", **evolve_pair(Wh_mm, Wnm_mm),
        },
        "B2_plus": {
            "ports_0based": list(b2_ports), "ports_1based": [p + 1 for p in b2_ports],
            "passive_prep_unitary_err_psi1": prep_err_1, "passive_prep_unitary_err_psi2": prep_err_2,
            **evolve_pair(Wb2p, Wb2_ref),
        },
        "B2_minus": {
            "ports_0based": list(b2_ports), "ports_1based": [p + 1 for p in b2_ports],
            **evolve_pair(Wb2m, Wb2_ref),
        },
        "B2_plus_layer0_aligned": {
            "ports_0based": list(b2_ports_aligned), "ports_1based": [p + 1 for p in b2_ports_aligned],
            **evolve_pair(Wb2p_al, Wb2_ref_al),
        },
        "B2_CZ4_doublon_free": {
            "ports_0based": list(cz4_ports), "ports_1based": [p + 1 for p in cz4_ports],
            "U_sw_unitary_switch_err": cz4_u_sw_err, **res_cz4,
        },
    }

    max_rdm_all = max(v["shot_by_shot_max_RDM_Frob_diff"] for v in families.values())
    assert max_rdm_all < 1e-13

    def off_gap(name: str):
        c = families[name]["crossing_offdiag"]
        return float(c["max_inversion_gap"]) if c is not None else None

    g_np_m, g_np_mm = off_gap("HOM_vs_NOONplus_matched"), off_gap("HOM_vs_NOONplus_mismatched")
    g_nm_m, g_nm_mm = off_gap("HOM_vs_NOONminus_matched"), off_gap("HOM_vs_NOONminus_mismatched")
    return {
        "N_modes": N_modes,
        "D2": D2,
        "theta_rad": theta,
        "gamma_layer": gamma_layer,
        "depths": list(range(D_depths + 1)),
        "max_shot_by_shot_RDM_Frob_diff_all_families": max_rdm_all,
        "port_alignment_contrast_ratios": {
            "HOM_NOONplus_matched_gap": g_np_m,
            "HOM_NOONplus_mismatched_gap": g_np_mm,
            "HOM_NOONminus_matched_gap": g_nm_m,
            "HOM_NOONminus_mismatched_gap": g_nm_mm,
            "pure_port_misalignment_HOM_NOONplus_ratio": round(g_np_m / g_np_mm, 4) if (g_np_m and g_np_mm) else None,
            "combined_misalignment_and_sign_NOONplus_vs_NOONminus_ratio": round(g_np_m / g_nm_mm, 4) if (g_np_m and g_nm_mm) else None,
            "pure_phase_sign_matched_NOONplus_over_NOONminus_ratio": round(g_np_m / g_nm_m, 4) if (g_np_m and g_nm_m) else None,
        },
        "families": families,
    }


def run_clements_monte_carlo(
    N_modes: int,
    D_depths: int,
    theta: float,
    gamma_layer: float,
    matched_ports: tuple[int, int],
    mismatched_ports: tuple[int, int],
    b2_ports: tuple[int, int, int],
    cz4_ports: tuple[int, int, int, int],
    N_real_list: list[int],
    n_pool: int,
    n_trials: int,
    pairs_per_sec: int,
    rng: np.random.Generator,
):
    """Finite-ensemble random-phase Monte Carlo with Poissonian SNSPD coincidence counting."""
    sigma_phi = np.sqrt(gamma_layer)
    U1_e, U1_o = clements_layer(N_modes, 0, theta), clements_layer(N_modes, 1, theta)
    G_ss = gamma_ss(N_modes)
    off_mask = ~np.eye(N_modes, dtype=bool)
    iu, ju = np.triu_indices(N_modes, k=1)

    Wh_m, Wnp_m = state_b1(N_modes, *matched_ports, noon_sign=+1.0)
    Wh_mm, Wnm_mm = state_b1(N_modes, *mismatched_ports, noon_sign=-1.0)
    Wb2_1, Wb2_2 = state_b2(N_modes, *b2_ports, sign=+1.0)
    Wcz_A, Wcz_B = state_cz4(N_modes, *cz4_ports)
    state_Ws = {
        "HOM_m": Wh_m, "NOONp_m": Wnp_m, "HOM_mm": Wh_mm, "NOONm_mm": Wnm_mm,
        "B2_1": Wb2_1, "B2_2": Wb2_2, "CZ4_A": Wcz_A, "CZ4_B": Wcz_B,
    }

    U_pool = np.broadcast_to(np.eye(N_modes, dtype=np.complex128), (n_pool, N_modes, N_modes)).copy()
    G_pools = {}
    for key, W0 in state_Ws.items():
        arr = np.empty((n_pool, D_depths + 1, N_modes, N_modes), dtype=np.float64)
        arr[:, 0] = 2.0 * np.abs(W0) ** 2
        G_pools[key] = arr

    for d in range(D_depths):
        U1 = U1_e if (d % 2 == 0) else U1_o
        ph = np.exp(-1j * rng.normal(0.0, sigma_phi, size=(n_pool, N_modes)))
        U_pool = ph[:, :, None] * (U1[None] @ U_pool)
        UT = np.transpose(U_pool, (0, 2, 1))
        for key, W0 in state_Ws.items():
            G_pools[key][:, d + 1] = 2.0 * np.abs(U_pool @ W0[None] @ UT) ** 2

    def add_shot_noise(G_mean: np.ndarray, n_tot: int, include_diag: bool = False) -> np.ndarray:
        # P(1_q, 1_r) = Gamma_{qr} (q < r); P(2_q) = (1/2) Gamma_{qq}.
        G_noisy = G_mean.copy()
        c_off = rng.poisson(np.maximum(G_mean[:, iu, ju], 0.0) * n_tot) / float(n_tot)
        G_noisy[:, iu, ju] = G_noisy[:, ju, iu] = c_off
        if include_diag:
            idx_d = np.arange(N_modes)
            G_noisy[:, idx_d, idx_d] = 2.0 * rng.poisson(np.maximum(0.5 * G_mean[:, idx_d, idx_d], 0.0) * n_tot) / float(n_tot)
        return G_noisy

    def summarize(vals: np.ndarray, ok_count: int):
        mean_v, std_v = float(np.mean(vals)), float(np.std(vals, ddof=1))
        lo16, hi84 = np.percentile(vals, [16.0, 84.0])
        return {
            "mean_gap": round(mean_v, 6),
            "std_gap": round(std_v, 6),
            "ci68_low": round(float(lo16), 6),
            "ci68_high": round(float(hi84), 6),
            "SNR": round(mean_v / max(std_v, 1e-12), 2) if std_v > 1e-12 else None,
            "success_rate": round(ok_count / float(len(vals)), 4),
        }

    scaling, depth_bands = {}, {}
    for n_real in N_real_list:
        n_tot = n_real * pairs_per_sec
        rec = {k: [] for k in [
            "hom_np_d3_phase", "hom_np_d3_shot", "hom_nm_d4_phase", "hom_nm_d4_shot",
            "b2_full_d3_phase", "b2_full_d3_shot", "cz4_d1_unitary_phase", "cz4_d1_unitary_shot",
            "cz4_d2_diss_phase", "cz4_d2_diss_shot", "cz4_d3_diss_phase", "cz4_d3_diss_shot",
        ]}
        ok = {k: 0 for k in rec}
        store_bands = n_real in (30, 100)
        if store_bands:
            band_store = {k: np.empty((n_trials, D_depths + 1), dtype=np.float64)
                          for k in ["HOM_m", "NOONp_m", "B2_1", "B2_2", "CZ4_A", "CZ4_B"]}

        for t_idx in range(n_trials):
            sel = rng.choice(n_pool, size=n_real, replace=False)
            Gm_h = np.mean(G_pools["HOM_m"][sel], axis=0)
            Gm_np = np.mean(G_pools["NOONp_m"][sel], axis=0)
            Gm_hmm = np.mean(G_pools["HOM_mm"][sel], axis=0)
            Gm_nmm = np.mean(G_pools["NOONm_mm"][sel], axis=0)
            Gm_b1 = np.mean(G_pools["B2_1"][sel], axis=0)
            Gm_b2 = np.mean(G_pools["B2_2"][sel], axis=0)
            Gm_cA = np.mean(G_pools["CZ4_A"][sel], axis=0)
            Gm_cB = np.mean(G_pools["CZ4_B"][sel], axis=0)

            Gs_h = add_shot_noise(Gm_h, n_tot, False)
            Gs_np = add_shot_noise(Gm_np, n_tot, False)
            Gs_hmm = add_shot_noise(Gm_hmm, n_tot, False)
            Gs_nmm = add_shot_noise(Gm_nmm, n_tot, False)
            Gs_b1 = add_shot_noise(Gm_b1, n_tot, True)
            Gs_b2 = add_shot_noise(Gm_b2, n_tot, True)
            Gs_cA = add_shot_noise(Gm_cA, n_tot, False)
            Gs_cB = add_shot_noise(Gm_cB, n_tot, False)

            dist_off = lambda G: np.linalg.norm((G - G_ss[None])[:, off_mask], axis=1)
            dist_full = lambda G: np.linalg.norm(G - G_ss[None], axis=(1, 2))

            dh_p, dnp_p = dist_off(Gm_h), dist_off(Gm_np)
            dh_s, dnp_s = dist_off(Gs_h), dist_off(Gs_np)
            dhmm_p, dnmm_p = dist_off(Gm_hmm), dist_off(Gm_nmm)
            dhmm_s, dnmm_s = dist_off(Gs_hmm), dist_off(Gs_nmm)
            db1_p, db2_p = dist_full(Gm_b1), dist_full(Gm_b2)
            db1_s, db2_s = dist_full(Gs_b1), dist_full(Gs_b2)
            dcA_p, dcB_p = dist_off(Gm_cA), dist_off(Gm_cB)
            dcA_s, dcB_s = dist_off(Gs_cA), dist_off(Gs_cB)

            if store_bands:
                band_store["HOM_m"][t_idx], band_store["NOONp_m"][t_idx] = dh_s, dnp_s
                band_store["B2_1"][t_idx], band_store["B2_2"][t_idx] = db1_s, db2_s
                band_store["CZ4_A"][t_idx], band_store["CZ4_B"][t_idx] = dcA_s, dcB_s

            for name, gap_val, init_ok in [
                ("hom_np_d3_phase", dnp_p[3] - dh_p[3], dh_p[0] > dnp_p[0]),
                ("hom_np_d3_shot", dnp_s[3] - dh_s[3], dh_s[0] > dnp_s[0]),
                ("hom_nm_d4_phase", dnmm_p[4] - dhmm_p[4], dhmm_p[0] > dnmm_p[0]),
                ("hom_nm_d4_shot", dnmm_s[4] - dhmm_s[4], dhmm_s[0] > dnmm_s[0]),
                ("b2_full_d3_phase", db2_p[3] - db1_p[3], db1_p[0] > db2_p[0]),
                ("b2_full_d3_shot", db2_s[3] - db1_s[3], db1_s[0] > db2_s[0]),
                ("cz4_d1_unitary_phase", dcB_p[1] - dcA_p[1], dcA_p[0] > dcB_p[0]),
                ("cz4_d1_unitary_shot", dcB_s[1] - dcA_s[1], dcA_s[0] > dcB_s[0]),
                ("cz4_d2_diss_phase", dcB_p[2] - dcA_p[2], dcA_p[0] > dcB_p[0]),
                ("cz4_d2_diss_shot", dcB_s[2] - dcA_s[2], dcA_s[0] > dcB_s[0]),
                ("cz4_d3_diss_phase", dcB_p[3] - dcA_p[3], dcA_p[0] > dcB_p[0]),
                ("cz4_d3_diss_shot", dcB_s[3] - dcA_s[3], dcA_s[0] > dcB_s[0]),
            ]:
                rec[name].append(float(gap_val))
                if init_ok and gap_val > 0.0:
                    ok[name] += 1

        scaling[f"Nreal_{n_real}"] = {
            "N_real": n_real,
            "pairs_per_mask": pairs_per_sec,
            "total_pairs_per_state_depth": n_tot,
            "est_wall_time_sec_8depths_2states_1s_mask": 8 * 2 * n_real,
            "HOM_NOONplus_matched_d3_phase_only": summarize(np.array(rec["hom_np_d3_phase"]), ok["hom_np_d3_phase"]),
            "HOM_NOONplus_matched_d3_with_SNSPD": summarize(np.array(rec["hom_np_d3_shot"]), ok["hom_np_d3_shot"]),
            "HOM_NOONminus_mismatched_d4_phase_only": summarize(np.array(rec["hom_nm_d4_phase"]), ok["hom_nm_d4_phase"]),
            "HOM_NOONminus_mismatched_d4_with_SNSPD": summarize(np.array(rec["hom_nm_d4_shot"]), ok["hom_nm_d4_shot"]),
            "B2_plus_fullHBT_d3_phase_only": summarize(np.array(rec["b2_full_d3_phase"]), ok["b2_full_d3_phase"]),
            "B2_plus_fullHBT_d3_with_SNSPD": summarize(np.array(rec["b2_full_d3_shot"]), ok["b2_full_d3_shot"]),
            "B2_CZ4_d1_unitary_phase_only": summarize(np.array(rec["cz4_d1_unitary_phase"]), ok["cz4_d1_unitary_phase"]),
            "B2_CZ4_d1_unitary_with_SNSPD": summarize(np.array(rec["cz4_d1_unitary_shot"]), ok["cz4_d1_unitary_shot"]),
            "B2_CZ4_d2_dissipative_phase_only": summarize(np.array(rec["cz4_d2_diss_phase"]), ok["cz4_d2_diss_phase"]),
            "B2_CZ4_d2_dissipative_with_SNSPD": summarize(np.array(rec["cz4_d2_diss_shot"]), ok["cz4_d2_diss_shot"]),
            "B2_CZ4_d3_dissipative_phase_only": summarize(np.array(rec["cz4_d3_diss_phase"]), ok["cz4_d3_diss_phase"]),
            "B2_CZ4_d3_dissipative_with_SNSPD": summarize(np.array(rec["cz4_d3_diss_shot"]), ok["cz4_d3_diss_shot"]),
        }
        if store_bands:
            depth_bands[f"Nreal_{n_real}"] = {
                k: {
                    "mean": [round(float(x), 6) for x in np.mean(arr, axis=0)],
                    "std": [round(float(x), 6) for x in np.std(arr, axis=0, ddof=1)],
                }
                for k, arr in band_store.items()
            }

    return {
        "N_modes": N_modes,
        "n_pool": n_pool,
        "n_trials": n_trials,
        "SNSPD_pair_rate_Hz": pairs_per_sec,
        "integration_sec_per_mask": 1.0,
        "scaling_by_N_real": scaling,
        "depth_bands_with_SNSPD": depth_bands,
    }


def run_fsw_segmentation_scan(rng: np.random.Generator):
    """Piecewise-constant segmented-waveguide (FSW) scan on N = 8 small-world graph vs Lindblad limit (Fig. 6d)."""
    N, J, gamma_0 = 8, 1.0, 0.45
    adj = graph_small_world(N)
    H1, _, _, H2, _, M2, n_ops, pairs, _, _ = two_boson_model(adj, J, np.full(N, gamma_0))
    H1_real = np.real(H1)
    G_ss = gamma_ss(N)
    off_mask = ~np.eye(N, dtype=bool)

    W1_b2, W2_b2 = state_b2(N, 7, 0, 6, sign=+1.0)
    Wh_2p, Wnp_2p = state_b1(N, 4, 5, noon_sign=+1.0)

    dz_ode, z_max, dz_eval = 0.005, 2.40, 0.025
    n_ode = int(round(z_max / dz_ode))
    stride_ode = int(round(dz_eval / dz_ode))
    n_eval = int(round(z_max / dz_eval))
    z_eval = np.linspace(0.0, z_max, n_eval + 1)

    def ode_hbt(W0):
        psi = wf_to_fock(W0, pairs)
        tr = rk4(H2, M2, np.outer(psi, psi.conj()), dz_ode, n_ode)[::stride_ode]
        return hbt_observables(tr, n_ops, pairs, N)[1]

    G1_ode, G2_ode = ode_hbt(W1_b2), ode_hbt(W2_b2)
    Gh_ode, Gnp_ode = ode_hbt(Wh_2p), ode_hbt(Wnp_2p)
    d1_ode = np.linalg.norm(G1_ode - G_ss[None], axis=(1, 2))
    d2_ode = np.linalg.norm(G2_ode - G_ss[None], axis=(1, 2))
    dh_ode = np.linalg.norm((Gh_ode - G_ss[None])[:, off_mask], axis=1)
    dnp_ode = np.linalg.norm((Gnp_ode - G_ss[None])[:, off_mask], axis=1)
    delta_b2_ode, delta_2p_ode = d2_ode - d1_ode, dnp_ode - dh_ode

    c_b2_ode = crossing_stats(d1_ode, d2_ode, z_eval)
    c_2p_ode = crossing_stats(dh_ode, dnp_ode, z_eval)
    lindblad_gap_b2 = float(c_b2_ode["max_inversion_gap"])
    lindblad_gap_2p = float(c_2p_ode["max_inversion_gap"])

    # Piecewise-constant detuning delta_beta_j ~ N(0, gamma_0 / dz_seg) evaluated continuously at dz_eval.
    n_large = 4000
    eye_N = np.eye(N, dtype=np.float64)
    scan = []
    for dz_seg in [0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00, 1.50, 2.00]:
        sub_per_seg = int(round(dz_seg / dz_eval))
        n_segs = int(np.ceil(n_eval / sub_per_seg))
        sigma_beta = np.sqrt(gamma_0 / dz_seg)

        U_seg_start = np.broadcast_to(np.eye(N, dtype=np.complex128), (n_large, N, N)).copy()
        G1_m = np.zeros((n_eval + 1, N, N), dtype=np.float64)
        G2_m = np.zeros((n_eval + 1, N, N), dtype=np.float64)
        Gh_m = np.zeros((n_eval + 1, N, N), dtype=np.float64)
        Gnp_m = np.zeros((n_eval + 1, N, N), dtype=np.float64)
        G1_m[0], G2_m[0] = 2.0 * np.abs(W1_b2) ** 2, 2.0 * np.abs(W2_b2) ** 2
        Gh_m[0], Gnp_m[0] = 2.0 * np.abs(Wh_2p) ** 2, 2.0 * np.abs(Wnp_2p) ** 2

        max_shot_rdm = 0.0
        for seg_idx in range(n_segs):
            dbeta = rng.normal(0.0, sigma_beta, size=(n_large, N))
            evals, evecs = np.linalg.eigh(H1_real[None, :, :] + dbeta[:, :, None] * eye_N[None, :, :])
            k_start = seg_idx * sub_per_seg
            k_end = min((seg_idx + 1) * sub_per_seg, n_eval)
            for k in range(k_start + 1, k_end + 1):
                tau = (k - k_start) * dz_eval
                U_sub = np.einsum("rik,rk,rjk->rij", evecs, np.exp(-1j * evals * tau), evecs) @ U_seg_start
                UT = np.transpose(U_sub, (0, 2, 1))
                W1_r = U_sub @ W1_b2[None] @ UT
                W2_r = U_sub @ W2_b2[None] @ UT
                if k == 1 or k == k_end:
                    D1_0 = 2.0 * (W1_r[0] @ W1_r[0].conj().T).conj()
                    D2_0 = 2.0 * (W2_r[0] @ W2_r[0].conj().T).conj()
                    max_shot_rdm = max(max_shot_rdm, float(np.linalg.norm(D1_0 - D2_0, "fro")))
                G1_m[k] = np.mean(2.0 * np.abs(W1_r) ** 2, axis=0)
                G2_m[k] = np.mean(2.0 * np.abs(W2_r) ** 2, axis=0)
                Gh_m[k] = np.mean(2.0 * np.abs(U_sub @ Wh_2p[None] @ UT) ** 2, axis=0)
                Gnp_m[k] = np.mean(2.0 * np.abs(U_sub @ Wnp_2p[None] @ UT) ** 2, axis=0)
            U_seg_start = np.einsum("rik,rk,rjk->rij", evecs, np.exp(-1j * evals * dz_seg), evecs) @ U_seg_start

        d1_s = np.linalg.norm(G1_m - G_ss[None], axis=(1, 2))
        d2_s = np.linalg.norm(G2_m - G_ss[None], axis=(1, 2))
        dh_s = np.linalg.norm((Gh_m - G_ss[None])[:, off_mask], axis=1)
        dnp_s = np.linalg.norm((Gnp_m - G_ss[None])[:, off_mask], axis=1)
        delta_b2_s, delta_2p_s = d2_s - d1_s, dnp_s - dh_s
        c_b2 = crossing_stats(d1_s, d2_s, z_eval)
        c_2p = crossing_stats(dh_s, dnp_s, z_eval)
        gap_b2 = float(c_b2["max_inversion_gap"]) if c_b2 else 0.0
        gap_2p = float(c_2p["max_inversion_gap"]) if c_2p else 0.0

        scan.append({
            "J_dz": dz_seg,
            "dz_eval_continuous": dz_eval,
            "substeps_per_segment": sub_per_seg,
            "n_segments_covering_zmax": n_segs,
            "z_max": z_max,
            "sigma_beta_over_J": round(float(sigma_beta), 5),
            "n_realizations": n_large,
            "single_realization_RDM_Frob_diff": max_shot_rdm,
            "B2_max_curve_deviation_from_Lindblad": round(float(max(np.max(np.abs(d1_s - d1_ode)), np.max(np.abs(d2_s - d2_ode)))), 6),
            "B2_max_gap_deviation_from_Lindblad": round(float(np.max(np.abs(delta_b2_s - delta_b2_ode))), 6),
            "B2_crossing": c_b2,
            "B2_peak_contrast_reduction_pct_vs_Lindblad": round(100.0 * (lindblad_gap_b2 - gap_b2) / lindblad_gap_b2, 2),
            "HOM_NOONplus_offdiag_crossing": c_2p,
            "HOM_NOONplus_max_gap_deviation_from_Lindblad": round(float(np.max(np.abs(delta_2p_s - delta_2p_ode))), 6),
            "HOM_NOONplus_peak_contrast_reduction_pct_vs_Lindblad": round(100.0 * (lindblad_gap_2p - gap_2p) / lindblad_gap_2p, 2),
            "curves_B2": {
                "Jz": [round(float(x), 4) for x in z_eval],
                "d1": [round(float(x), 6) for x in d1_s],
                "d2": [round(float(x), 6) for x in d2_s],
                "delta_d": [round(float(x), 6) for x in delta_b2_s],
                "delta_d_minus_Lindblad": [round(float(x), 6) for x in (delta_b2_s - delta_b2_ode)],
            },
        })

    return {
        "N": N,
        "topology": "small_world_8",
        "gamma_over_J": gamma_0,
        "continuous_eval_grid": {"dz_eval": dz_eval, "z_max": z_max, "n_points": len(z_eval)},
        "Lindblad_exact_baseline": {
            "B2_crossing": c_b2_ode,
            "HOM_NOONplus_offdiag_crossing": c_2p_ode,
            "curves_B2": {
                "Jz": [round(float(x), 4) for x in z_eval],
                "d1": [round(float(x), 6) for x in d1_ode],
                "d2": [round(float(x), 6) for x in d2_ode],
                "delta_d": [round(float(x), 6) for x in delta_b2_ode],
            },
        },
        "segmentation_scan": scan,
    }


def main():
    t0 = time.perf_counter()
    seed = 20261120
    rng = np.random.default_rng(seed)

    mzi12_exact = run_clements_exact_mesh(
        12, 10, 0.44, 0.36, (4, 5), (5, 6), (5, 6, 4), (4, 5, 6), (4, 5, 6, 7), rng
    )
    mzi20_exact = run_clements_exact_mesh(
        20, 10, 0.44, 0.36, (8, 9), (9, 10), (9, 10, 8), (8, 9, 10), (8, 9, 10, 11), rng
    )
    mc12 = run_clements_monte_carlo(
        12, 10, 0.44, 0.36, (4, 5), (5, 6), (5, 6, 4), (4, 5, 6, 7),
        [10, 20, 30, 50, 100, 300], 5000, 250, 25000, rng,
    )
    mc20 = run_clements_monte_carlo(
        20, 10, 0.44, 0.36, (8, 9), (9, 10), (9, 10, 8), (8, 9, 10, 11),
        [30, 100], 1500, 150, 25000, rng,
    )

    theta_sweep = []
    for th in [0.15, 0.20, 0.22, 0.25, 0.30, 0.35, 0.44]:
        res_u = run_clements_exact_mesh(12, 10, th, 0.0, (4, 5), (5, 6), (5, 6, 4), (4, 5, 6), (4, 5, 6, 7), rng)
        res_d = run_clements_exact_mesh(12, 10, th, 0.36, (4, 5), (5, 6), (5, 6, 4), (4, 5, 6), (4, 5, 6, 7), rng)
        entry = {"theta_rad": th, "gamma_layer_0.00": {}, "gamma_layer_0.36": {}}
        for tag, r_obj in (("gamma_layer_0.00", res_u), ("gamma_layer_0.36", res_d)):
            for fname, metric in (
                ("HOM_vs_NOONplus_matched", "crossing_offdiag"),
                ("B2_plus", "crossing_full_HBT"),
                ("B2_CZ4_doublon_free", "crossing_offdiag"),
            ):
                c = r_obj["families"][fname][metric]
                d1_k = "d1_offdiag" if "offdiag" in metric else "d1_full_HBT"
                d2_k = "d2_offdiag" if "offdiag" in metric else "d2_full_HBT"
                entry[tag][fname] = {
                    "d1_at_unitary_d1": r_obj["families"][fname][d1_k][1],
                    "d2_at_unitary_d1": r_obj["families"][fname][d2_k][1],
                    "crossing_strictly_dissipative_d_ge_2": bool(c is not None and c["first_crossing"] >= 2.0),
                    "crossing": c,
                }
        theta_sweep.append(entry)

    mzi12_th025_exact = run_clements_exact_mesh(
        12, 10, 0.25, 0.36, (4, 5), (5, 6), (5, 6, 4), (4, 5, 6), (4, 5, 6, 7), rng
    )
    # Finite-ensemble SNSPD Monte Carlo at peak depth d = 5 for theta = 0.25 rad (Fig. 6b, Table II).
    n_pool_025, sigma_phi_025 = 3000, np.sqrt(0.36)
    U1e_025, U1o_025 = clements_layer(12, 0, 0.25), clements_layer(12, 1, 0.25)
    Gss_12, off_12 = gamma_ss(12), ~np.eye(12, dtype=bool)
    iu12, ju12 = np.triu_indices(12, k=1)
    Wh_025, Wnp_025 = state_b1(12, 4, 5, +1.0)
    Wb1_025, Wb2_025 = state_b2(12, 5, 6, 4, +1.0)
    WcA_025, WcB_025 = state_cz4(12, 4, 5, 6, 7)

    Up = np.broadcast_to(np.eye(12, dtype=np.complex128), (n_pool_025, 12, 12)).copy()
    for d_s in range(5):
        U1s = U1e_025 if (d_s % 2 == 0) else U1o_025
        ph = np.exp(-1j * rng.normal(0.0, sigma_phi_025, size=(n_pool_025, 12)))
        Up = ph[:, :, None] * (U1s[None] @ Up)
    UT = np.transpose(Up, (0, 2, 1))
    Gh_d5 = 2.0 * np.abs(Up @ Wh_025[None] @ UT) ** 2
    Gnp_d5 = 2.0 * np.abs(Up @ Wnp_025[None] @ UT) ** 2
    Gb1_d5 = 2.0 * np.abs(Up @ Wb1_025[None] @ UT) ** 2
    Gb2_d5 = 2.0 * np.abs(Up @ Wb2_025[None] @ UT) ** 2
    GcA_d5 = 2.0 * np.abs(Up @ WcA_025[None] @ UT) ** 2
    GcB_d5 = 2.0 * np.abs(Up @ WcB_025[None] @ UT) ** 2

    mc12_th025_stats = {}
    for n_real in [10, 20, 30, 50, 100, 300]:
        n_tot = n_real * 25000
        g_np, g_b2, g_cz = [], [], []
        for _ in range(200):
            sel = rng.choice(n_pool_025, size=n_real, replace=False)

            def noisy_off(G_arr):
                Gm = np.mean(G_arr[sel], axis=0)
                Gn = Gm.copy()
                c = rng.poisson(np.maximum(Gm[iu12, ju12], 0.0) * n_tot) / float(n_tot)
                Gn[iu12, ju12] = Gn[ju12, iu12] = c
                return float(np.linalg.norm((Gn - Gss_12)[off_12]))

            def noisy_full(G_arr):
                Gm = np.mean(G_arr[sel], axis=0)
                Gn = Gm.copy()
                c = rng.poisson(np.maximum(Gm[iu12, ju12], 0.0) * n_tot) / float(n_tot)
                Gn[iu12, ju12] = Gn[ju12, iu12] = c
                idx_d = np.arange(12)
                Gn[idx_d, idx_d] = 2.0 * rng.poisson(np.maximum(0.5 * Gm[idx_d, idx_d], 0.0) * n_tot) / float(n_tot)
                return float(np.linalg.norm(Gn - Gss_12))

            g_np.append(noisy_off(Gnp_d5) - noisy_off(Gh_d5))
            g_b2.append(noisy_full(Gb2_d5) - noisy_full(Gb1_d5))
            g_cz.append(noisy_off(GcB_d5) - noisy_off(GcA_d5))

        def pack_mc(arr):
            a = np.array(arr)
            m, s = float(np.mean(a)), float(np.std(a, ddof=1))
            lo, hi = np.percentile(a, [16.0, 84.0])
            return {
                "mean_gap": round(m, 6), "std_gap": round(s, 6),
                "ci68_low": round(float(lo), 6), "ci68_high": round(float(hi), 6),
                "SNR": round(m / max(s, 1e-12), 2),
                "success_rate": round(float(np.mean(a > 0.0)), 4),
            }

        mc12_th025_stats[f"Nreal_{n_real}"] = {
            "N_real": n_real,
            "HOM_NOONplus_matched_d5_with_SNSPD": pack_mc(g_np),
            "B2_plus_fullHBT_d5_with_SNSPD": pack_mc(g_b2),
            "B2_CZ4_d5_with_SNSPD": pack_mc(g_cz),
        }

    fsw = run_fsw_segmentation_scan(rng)
    elapsed = time.perf_counter() - t0

    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, "clements_fsw.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {"seed": seed, "runtime_seconds": round(elapsed, 2)},
            "clements_12mode_exact": mzi12_exact,
            "clements_20mode_exact": mzi20_exact,
            "clements_12mode_monte_carlo": mc12,
            "clements_20mode_monte_carlo": mc20,
            "clements_12mode_theta_sweep": theta_sweep,
            "clements_12mode_theta025_exact": mzi12_th025_exact,
            "clements_12mode_theta025_monte_carlo_d5": mc12_th025_stats,
            "fsw_segmentation_scan": fsw,
        }, f, indent=2)
    print(f"clements_fsw.json ({os.path.getsize(out_path) / 1024:.1f} KB, {elapsed:.2f} s)")


if __name__ == "__main__":
    main()
