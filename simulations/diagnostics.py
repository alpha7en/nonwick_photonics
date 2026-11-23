"""First-order-blind chip diagnostics: 12-mode local dephasing tomography {gamma_j},
4x sensitivity comparison against single-photon interferometry, and finite-depth
Kraus-calibrated Hong-Ou-Mandel visibility readout (Sec. IV.B, Appendix C, Fig. 7).

Writes data/diagnostics.json.
Usage: python -I simulations/diagnostics.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import (
    DATA_DIR,
    clements_layer,
    dephasing_matrix,
    lift_sym2,
    rk4,
    state_b1,
    sym2_basis,
    wf_to_fock,
)


def _offdiag_indices(pairs: list[tuple[int, int]]):
    return np.array([k for k, (m, n) in enumerate(pairs) if m < n], dtype=np.int32)


def run_dephasing_reconstruction(gamma_true: np.ndarray, V_hom: float, M_pairs: int, rng: np.random.Generator):
    """12-mode local dephasing reconstruction from overlapping-triangle pair sums S_{ab} = gamma_a + gamma_b."""
    N = len(gamma_true)
    pi_plus, pi_minus = 0.5 * (1.0 + V_hom), 0.5 * (1.0 - V_hom)

    edges_adj = [(j, j + 1) for j in range(N - 1)]
    edges_nnn = [(j, j + 2) for j in range(N - 2)]
    edges_full = edges_adj + edges_nnn
    M_edges = len(edges_full)

    def incidence(edges):
        B = np.zeros((len(edges), N), dtype=np.float64)
        for p, (a, b) in enumerate(edges):
            B[p, a] = B[p, b] = 1.0
        return B

    B_adj = incidence(edges_adj)
    B_1tri = incidence(edges_adj + [(0, 2)])
    B_full = incidence(edges_full)

    # Single-realization g^(1)-invisibility check across all 21 probe pairs (Theorem 2).
    max_g1_diff = 0.0
    for a, b in edges_full:
        W_hom, W_np = state_b1(N, a, b, noon_sign=+1.0)
        _, W_nm = state_b1(N, a, b, noon_sign=-1.0)
        U_bs = np.eye(N, dtype=np.complex128)
        U_bs[np.ix_([a, b], [a, b])] = np.array([[1.0, 1j], [1j, 1.0]]) / np.sqrt(2.0)
        for _ in range(5):
            phi = rng.normal(0.0, np.sqrt(gamma_true))
            U_tot = U_bs @ np.diag(np.exp(-1j * phi))
            D_np = 2.0 * ((U_tot @ W_np @ U_tot.T) @ (U_tot @ W_np @ U_tot.T).conj().T).conj()
            D_nm = 2.0 * ((U_tot @ W_nm @ U_tot.T) @ (U_tot @ W_nm @ U_tot.T).conj().T).conj()
            D_hm = 2.0 * ((U_tot @ W_hom @ U_tot.T) @ (U_tot @ W_hom @ U_tot.T).conj().T).conj()
            max_g1_diff = max(max_g1_diff, float(np.max(np.abs(D_np - D_nm))), float(np.max(np.abs(D_hm - D_np))))

    S_true = B_full @ gamma_true
    norm_true = float(np.linalg.norm(gamma_true))
    d_hold, n_trials = 1, 200
    bg_idx = [j for j in range(N) if j not in (3, 8)]
    hs_idx = [3, 8]

    def summarize(samples: np.ndarray, rep_est: np.ndarray, rep_se: np.ndarray):
        rmses = np.linalg.norm(samples - gamma_true[None, :], axis=1) / np.sqrt(N)
        rel_rmses = np.linalg.norm(samples - gamma_true[None, :], axis=1) / norm_true
        hs_ok = int(np.sum(np.min(samples[:, hs_idx], axis=1) > np.max(samples[:, bg_idx], axis=1)))
        return {
            "n_mc_trials": int(samples.shape[0]),
            "gamma_mean": [round(float(x), 5) for x in np.mean(samples, axis=0)],
            "gamma_std": [round(float(x), 5) for x in np.std(samples, axis=0, ddof=1)],
            "gamma_p10": [round(float(x), 5) for x in np.percentile(samples, 10, axis=0)],
            "gamma_p90": [round(float(x), 5) for x in np.percentile(samples, 90, axis=0)],
            "rmse_abs_mean": round(float(np.mean(rmses)), 5),
            "rmse_abs_std": round(float(np.std(rmses, ddof=1)), 5),
            "rel_rmse_mean": round(float(np.mean(rel_rmses)), 5),
            "rel_rmse_std": round(float(np.std(rel_rmses, ddof=1)), 5),
            "hotspot_identification_rate": round(hs_ok / float(samples.shape[0]), 4),
            "representative_single_experiment": {
                "gamma_hat": [round(float(x), 5) for x in rep_est],
                "gamma_se": [round(float(x), 5) for x in rep_se],
                "rmse_abs": round(float(np.linalg.norm(rep_est - gamma_true) / np.sqrt(N)), 5),
                "rel_rmse": round(float(np.linalg.norm(rep_est - gamma_true) / norm_true), 5),
            },
        }

    # Coincidence probabilities after a 50:50 readout beam splitter U_DC:
    # P_ab^(HOM) = (1 - eta)/2, P_ab^(NOON+/-)(d) = (3 - eta)/4 +/- ((1 + eta)/4) exp(-2 S_ab d).
    P_hom = 0.5 * (1.0 - V_hom)
    P_p0, P_m0 = 1.0, 0.5 * (1.0 - V_hom)
    P_p1 = (3.0 - V_hom) / 4.0 + ((1.0 + V_hom) / 4.0) * np.exp(-2.0 * S_true * d_hold)
    P_m1 = (3.0 - V_hom) / 4.0 - ((1.0 + V_hom) / 4.0) * np.exp(-2.0 * S_true * d_hold)

    samples_A1 = np.zeros((n_trials, N), dtype=np.float64)
    samples_A2 = np.zeros((n_trials, N), dtype=np.float64)
    rep_A1 = rep_se_A1 = rep_A2 = rep_se_A2 = None

    for t in range(n_trials):
        c_p0 = rng.binomial(M_pairs, np.full(M_edges, P_p0)) / float(M_pairs)
        c_m0 = rng.binomial(M_pairs, np.full(M_edges, P_m0)) / float(M_pairs)
        c_p1 = rng.binomial(M_pairs, P_p1) / float(M_pairs)
        c_m1 = rng.binomial(M_pairs, P_m1) / float(M_pairs)
        c_hm = rng.binomial(M_pairs, np.full(M_edges, P_hom)) / float(M_pairs)

        # Two-depth log-ratio estimator S_hat = -(1/(2 d)) ln(Delta Gamma(d) / Delta Gamma(0)).
        dg0 = np.maximum(c_p0 - c_m0, 1e-4)
        dg1 = np.maximum(c_p1 - c_m1, 1e-4)
        S_A1 = -np.log(np.clip(dg1 / dg0, 1e-4, 0.9999)) / (2.0 * d_hold)
        var_dg0 = (c_p0 * (1.0 - c_p0) + c_m0 * (1.0 - c_m0)) / float(M_pairs)
        var_dg1 = (c_p1 * (1.0 - c_p1) + c_m1 * (1.0 - c_m1)) / float(M_pairs)
        var_S_A1 = np.maximum((var_dg1 / (dg1**2) + var_dg0 / (dg0**2)) / (4.0 * d_hold**2), 1e-12)
        W1 = np.diag(1.0 / var_S_A1)
        cov_A1 = np.linalg.inv(B_full.T @ W1 @ B_full)
        g_A1 = cov_A1 @ (B_full.T @ W1 @ S_A1)
        samples_A1[t] = g_A1
        if t == 0:
            rep_A1, rep_se_A1 = g_A1.copy(), np.sqrt(np.diag(cov_A1))

        # Single-depth 3-state self-calibrated ratio R_ab = (P+ - P-) / (P+ + P- - 2 P_HOM) = exp(-2 S_ab d).
        num_A2 = np.maximum(c_p1 - c_m1, 1e-4)
        den_A2 = np.maximum(c_p1 + c_m1 - 2.0 * c_hm, 1e-4)
        S_A2 = -np.log(np.clip(num_A2 / den_A2, 1e-4, 0.9999)) / (2.0 * d_hold)
        vp1 = c_p1 * (1.0 - c_p1) / float(M_pairs)
        vm1 = c_m1 * (1.0 - c_m1) / float(M_pairs)
        vhm = c_hm * (1.0 - c_hm) / float(M_pairs)
        var_lnR = (
            ((1.0 / num_A2 - 1.0 / den_A2) ** 2) * vp1
            + ((-1.0 / num_A2 - 1.0 / den_A2) ** 2) * vm1
            + ((2.0 / den_A2) ** 2) * vhm
        )
        W2 = np.diag(1.0 / np.maximum(var_lnR / (4.0 * d_hold**2), 1e-12))
        cov_A2 = np.linalg.inv(B_full.T @ W2 @ B_full)
        g_A2 = cov_A2 @ (B_full.T @ W2 @ S_A2)
        samples_A2[t] = g_A2
        if t == 0:
            rep_A2, rep_se_A2 = g_A2.copy(), np.sqrt(np.diag(cov_A2))

    # Controlled discrete Gaussian phase masks Phi ~ N(0, S_ab d):
    # P^(NOON+)(Phi) = pi_+ cos^2(Phi) + pi_-, P^(NOON-)(Phi) = pi_+ sin^2(Phi) + pi_-.
    synth_results = {}
    for n_real in [30, 100, 300]:
        m_per_mask = max(1, int(round(M_pairs / n_real)))
        samples_syn = np.zeros((n_trials, N), dtype=np.float64)
        rep_syn = rep_se_syn = None
        for t in range(n_trials):
            S_hat = np.zeros(M_edges, dtype=np.float64)
            S_var = np.zeros(M_edges, dtype=np.float64)
            for p in range(M_edges):
                phi = rng.normal(0.0, np.sqrt(S_true[p] * d_hold), size=n_real)
                pp = np.clip(pi_plus * (np.cos(phi) ** 2) + pi_minus, 0.0, 1.0)
                pm = np.clip(pi_plus * (np.sin(phi) ** 2) + pi_minus, 0.0, 1.0)
                cp = rng.binomial(m_per_mask, pp) / float(m_per_mask)
                cm = rng.binomial(m_per_mask, pm) / float(m_per_mask)
                ch = rng.binomial(M_pairs, P_hom) / float(M_pairs)
                diff_masks = cp - cm
                num = float(np.mean(diff_masks))
                den = float(np.mean(cp + cm) - 2.0 * ch)
                S_est = -np.log(np.clip(num / max(den, 1e-4), 1e-3, 0.999)) / (2.0 * d_hold)
                S_hat[p] = S_est
                var_num = float(np.var(diff_masks, ddof=1) / n_real)
                x_est = max(2.0 * S_est * d_hold, 0.05)
                var_S_ens = (S_est**2) * (2.0 / n_real) * ((np.sinh(x_est) / x_est) ** 2)
                var_S_emp = var_num / max(4.0 * (d_hold**2) * (max(num, 0.05) ** 2), 1e-8)
                S_var[p] = max(0.5 * (var_S_emp + var_S_ens), 1e-8)

            W_s = np.diag(1.0 / S_var)
            cov_s = np.linalg.inv(B_full.T @ W_s @ B_full)
            g_s = np.clip(cov_s @ (B_full.T @ W_s @ S_hat), 0.0, None)
            samples_syn[t] = g_s
            if t == 0:
                rep_syn, rep_se_syn = g_s.copy(), np.sqrt(np.diag(cov_s))
        synth_results[f"synthetic_masks_Nreal_{n_real}"] = summarize(samples_syn, rep_syn, rep_se_syn)

    # Coupled dimer integrated-area identity: int_0^inf Delta Gamma_12(z; eta) dz = (1 + eta) / (6 (gamma_1 + gamma_2)).
    g1_d, g2_d, J_d = 0.20, 0.55, 1.0
    _, _, n_ops_2 = sym2_basis(2)
    H2_d = np.array([
        [0.0, -np.sqrt(2.0) * J_d, 0.0],
        [-np.sqrt(2.0) * J_d, 0.0, -np.sqrt(2.0) * J_d],
        [0.0, -np.sqrt(2.0) * J_d, 0.0],
    ], dtype=np.complex128)
    M2_d = dephasing_matrix(n_ops_2, np.array([g1_d, g2_d]))
    drho0_d = np.zeros((3, 3), dtype=np.complex128)
    drho0_d[0, 2] = drho0_d[2, 0] = pi_plus
    dz_d, n_st_d = 0.01, 4000
    z_d = np.linspace(0.0, dz_d * n_st_d, n_st_d + 1)
    dGamma12_d = np.real(rk4(H2_d, M2_d, drho0_d, dz_d, n_st_d)[:, 1, 1])
    area_num = float(np.trapezoid(dGamma12_d, z_d))
    area_exact = (1.0 + V_hom) / (6.0 * (g1_d + g2_d))

    d_trace = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    s_bg, s_hs = float(gamma_true[0] + gamma_true[1]), float(gamma_true[3] + gamma_true[4])
    return max_g1_diff, {
        "modes_1based": list(range(1, N + 1)),
        "modes_0based": list(range(N)),
        "hotspot_modes_1based": [4, 9],
        "hotspot_modes_0based": [3, 8],
        "gamma_true": [round(float(x), 5) for x in gamma_true],
        "edges_1based": [[a + 1, b + 1] for a, b in edges_full],
        "edges_0based": [list(e) for e in edges_full],
        "design_matrix_conditioning": {
            "n_edges_adj_only": len(edges_adj),
            "rank_B_adj_only": int(np.linalg.matrix_rank(B_adj)),
            "cond_B_adj_only": "inf",
            "n_edges_1tri": len(edges_adj) + 1,
            "rank_B_1tri": int(np.linalg.matrix_rank(B_1tri)),
            "cond_B_1tri": round(float(np.linalg.cond(B_1tri)), 4),
            "n_edges_full_overlapping_triangles": M_edges,
            "rank_B_full": int(np.linalg.matrix_rank(B_full)),
            "cond_B_full": round(float(np.linalg.cond(B_full)), 4),
        },
        "native_shot_limited_A1": summarize(samples_A1, rep_A1, rep_se_A1),
        "native_shot_limited_A2_3state": summarize(samples_A2, rep_A2, rep_se_A2),
        **synth_results,
        "coupled_dimer_area_check": {
            "gamma_1_over_J": g1_d,
            "gamma_2_over_J": g2_d,
            "V_HOM": V_hom,
            "Jz_max": 40.0,
            "integral_numerical": round(area_num, 7),
            "integral_exact_analytic": round(area_exact, 7),
            "abs_error": float(f"{abs(area_num - area_exact):.3e}"),
            "rel_error": float(f"{abs(area_num - area_exact) / area_exact:.3e}"),
            "Jz_curve": [round(float(x), 3) for x in z_d[::40][:26]],
            "Delta_Gamma12_curve": [round(float(x), 6) for x in dGamma12_d[::40][:26]],
        },
        "background_vs_hotspot_depth_traces": {
            "depths": d_trace.tolist(),
            "bg_pair_1based": [1, 2],
            "bg_pair_sum": round(s_bg, 4),
            "bg_2ph_exact": [round(float(pi_plus * np.exp(-2.0 * s_bg * d)), 5) for d in d_trace],
            "bg_1ph_exact": [round(float(np.exp(-0.5 * s_bg * d)), 5) for d in d_trace],
            "hs_pair_1based": [4, 5],
            "hs_pair_sum": round(s_hs, 4),
            "hs_2ph_exact": [round(float(pi_plus * np.exp(-2.0 * s_hs * d)), 5) for d in d_trace],
            "hs_1ph_exact": [round(float(np.exp(-0.5 * s_hs * d)), 5) for d in d_trace],
        },
    }


def run_coupled_clements_insitu(gamma_true: np.ndarray, V_hom: float, rng: np.random.Generator):
    """In-situ weighted nonlinear least-squares tomography on a coupled 12-mode Clements mesh (theta = 0.44 rad)."""
    N, theta, D_max, M_pairs = len(gamma_true), 0.44, 4, 25000
    pi_plus, pi_minus = 0.5 * (1.0 + V_hom), 0.5 * (1.0 - V_hom)

    pairs, _, n_ops = sym2_basis(N)
    off_k = _offdiag_indices(pairs)
    D2, n_off = len(pairs), len(off_k)

    U1_e, U1_o = clements_layer(N, 0, theta), clements_layer(N, 1, theta)
    U2_e, U2_o = lift_sym2(U1_e, pairs), lift_sym2(U1_o, pairs)
    U2_e_dag, U2_o_dag = U2_e.conj().T, U2_o.conj().T

    probe_pairs = [(j, j + 1) for j in range(11)] + [(0, 2), (3, 5), (6, 8), (9, 11)]
    M_p = len(probe_pairs)
    drho_pm = np.zeros((M_p, D2, D2), dtype=np.complex128)
    drho_hn = np.zeros((M_p, D2, D2), dtype=np.complex128)
    W_np_arr = np.zeros((M_p, N, N), dtype=np.complex128)
    W_nm_arr = np.zeros((M_p, N, N), dtype=np.complex128)
    W_hm_arr = np.zeros((M_p, N, N), dtype=np.complex128)

    for p, (a, b) in enumerate(probe_pairs):
        W_hm_arr[p], W_np_arr[p] = state_b1(N, a, b, noon_sign=+1.0)
        _, W_nm_arr[p] = state_b1(N, a, b, noon_sign=-1.0)
        p_np, p_nm, p_hm = wf_to_fock(W_np_arr[p], pairs), wf_to_fock(W_nm_arr[p], pairs), wf_to_fock(W_hm_arr[p], pairs)
        drho_pm[p] = pi_plus * (np.outer(p_np, p_np.conj()) - np.outer(p_nm, p_nm.conj()))
        drho_hn[p] = pi_plus * (np.outer(p_hm, p_hm.conj()) - np.outer(p_np, p_np.conj()))

    def forward_offdiag(g_vec: np.ndarray, drho0: np.ndarray) -> np.ndarray:
        damp = np.exp(-dephasing_matrix(n_ops, g_vec))
        rho = drho0.copy()
        out = np.empty((M_p, D_max, n_off), dtype=np.float64)
        for d in range(D_max):
            U2, U2_d = (U2_e, U2_e_dag) if (d % 2 == 0) else (U2_o, U2_o_dag)
            rho = (U2 @ rho @ U2_d) * damp[None, :, :]
            out[:, d, :] = np.real(np.diagonal(rho, axis1=1, axis2=2))[:, off_k]
        return out

    def exact_probs(W_arr):
        rho = np.zeros((M_p, D2, D2), dtype=np.complex128)
        for p in range(M_p):
            psi = wf_to_fock(W_arr[p], pairs)
            rho[p] = np.outer(psi, psi.conj())
        damp = np.exp(-dephasing_matrix(n_ops, gamma_true))
        out = np.empty((M_p, D_max, n_off), dtype=np.float64)
        for d in range(D_max):
            U2, U2_d = (U2_e, U2_e_dag) if (d % 2 == 0) else (U2_o, U2_o_dag)
            rho = (U2 @ rho @ U2_d) * damp[None, :, :]
            out[:, d, :] = np.clip(np.real(np.diagonal(rho, axis1=1, axis2=2))[:, off_k], 0.0, 1.0)
        return out

    P_np_s, P_nm_s, P_hm_s = exact_probs(W_np_arr), exact_probs(W_nm_arr), exact_probs(W_hm_arr)
    P_an_bg = np.full_like(P_np_s, 1.0 / n_off)
    P_np_tot = np.clip(pi_plus * P_np_s + pi_minus * P_an_bg, 0.0, 1.0)
    P_nm_tot = np.clip(pi_plus * P_nm_s + pi_minus * P_an_bg, 0.0, 1.0)
    P_hm_tot = np.clip(pi_plus * P_hm_s + pi_minus * P_an_bg, 0.0, 1.0)

    norm_true = float(np.linalg.norm(gamma_true))
    mask_pm = np.abs(forward_offdiag(np.full(N, 0.22), drho_pm)[:, 1:, :].ravel()) > 2e-4
    mask_hn = np.abs(forward_offdiag(np.full(N, 0.22), drho_hn)[:, 1:, :].ravel()) > 2e-4

    def run_wnls(P_A, P_B, drho0, mask):
        n_tr = 10
        ests = np.zeros((n_tr, N), dtype=np.float64)
        for t in range(n_tr):
            cA = rng.binomial(M_pairs, P_A) / float(M_pairs)
            cB = rng.binomial(M_pairs, P_B) / float(M_pairs)
            y = (cA - cB)[:, 1:, :].ravel()[mask]
            sig = np.sqrt(np.maximum((P_A * (1.0 - P_A) + P_B * (1.0 - P_B))[:, 1:, :].ravel()[mask] / M_pairs, 1e-10))
            sol = least_squares(
                lambda g_v: (forward_offdiag(g_v, drho0)[:, 1:, :].ravel()[mask] - y) / sig,
                x0=np.full(N, 0.20), bounds=(0.0, 2.0), xtol=1e-6, ftol=1e-6,
            )
            ests[t] = sol.x
        rel_rmses = np.linalg.norm(ests - gamma_true[None, :], axis=1) / norm_true
        return {
            "n_mc_trials": n_tr,
            "gamma_mean": [round(float(x), 5) for x in np.mean(ests, axis=0)],
            "gamma_std": [round(float(x), 5) for x in np.std(ests, axis=0, ddof=1)],
            "rel_rmse_mean": round(float(np.mean(rel_rmses)), 5),
        }

    U_b = np.eye(N, dtype=np.complex128)
    max_g1_coupled = 0.0
    for d in range(D_max):
        U1 = U1_e if (d % 2 == 0) else U1_o
        U_b = np.exp(-1j * rng.normal(0.0, np.sqrt(gamma_true)))[:, None] * (U1 @ U_b)
        D_mats = [2.0 * ((U_b @ W @ U_b.T) @ (U_b @ W @ U_b.T).conj().T).conj() for W in (W_np_arr[0], W_nm_arr[0], W_hm_arr[0])]
        max_g1_coupled = max(max_g1_coupled, float(np.max(np.abs(D_mats[0] - D_mats[1]))), float(np.max(np.abs(D_mats[2] - D_mats[0]))))

    return {
        "N_modes": N,
        "theta_rad": theta,
        "D_max": D_max,
        "V_HOM": V_hom,
        "M_pairs_per_state": M_pairs,
        "single_realization_max_g1_diff": max_g1_coupled,
        "NOON_plus_vs_minus_insitu_WNLS": run_wnls(P_np_tot, P_nm_tot, drho_pm, mask_pm),
        "HOM_vs_NOON_plus_insitu_WNLS": run_wnls(P_hm_tot, P_np_tot, drho_hn, mask_hn),
    }


def run_sensitivity_vs_1photon(V_hom: float):
    """4x dephasing-rate sensitivity of two-photon NOON coherence (4 gamma_bar) vs single-photon (gamma_bar)."""
    pi_plus, d_probe, M_det = 0.5 * (1.0 + V_hom), 1, 25000
    sweep = []
    for g_bar in [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50]:
        y2, y1 = pi_plus * np.exp(-4.0 * g_bar * d_probe), np.exp(-g_bar * d_probe)
        slope_2ph, slope_1ph = 4.0 * d_probe * y2, d_probe * y1
        p2p, p2m = (3.0 - V_hom) / 4.0 + 0.5 * y2, (3.0 - V_hom) / 4.0 - 0.5 * y2
        p1p, p1m = 0.5 * (1.0 + y1), 0.5 * (1.0 - y1)
        sig_2ph = np.sqrt((p2p * (1.0 - p2p) + p2m * (1.0 - p2m)) / M_det) / slope_2ph
        sig_1ph = np.sqrt((p1p * (1.0 - p1p) + p1m * (1.0 - p1m)) / M_det) / slope_1ph
        shot_gain = sig_1ph / sig_2ph

        y2_id = np.exp(-4.0 * g_bar * d_probe)
        p2_id = 0.5 * (1.0 + y2_id)
        shot_gain_id = sig_1ph / (np.sqrt(2.0 * p2_id * (1.0 - p2_id) / M_det) / (4.0 * d_probe * y2_id))
        sweep.append({
            "gamma_bar": g_bar,
            "decay_rate_2ph": round(4.0 * g_bar, 5),
            "decay_rate_1ph": round(g_bar, 5),
            "exact_decay_rate_ratio": 4.0,
            "slope_sensitivity_ratio_VHOM": round(float(slope_2ph / slope_1ph), 4),
            "slope_sensitivity_ratio_ideal": round(float(4.0 * np.exp(-3.0 * g_bar * d_probe)), 4),
            "shot_sigma_2ph_VHOM": round(float(sig_2ph), 6),
            "shot_sigma_1ph": round(float(sig_1ph), 6),
            "shot_precision_gain_VHOM": round(float(shot_gain), 4),
            "shot_fisher_info_ratio_VHOM": round(float(shot_gain**2), 4),
            "shot_precision_gain_ideal": round(float(shot_gain_id), 4),
            "shot_fisher_info_ratio_ideal": round(float(shot_gain_id**2), 4),
        })

    d_fine = np.linspace(0.0, 5.0, 101)
    return {
        "V_HOM": V_hom,
        "d_probe": d_probe,
        "M_det_per_state": M_det,
        "exact_rate_ratio": 4.0,
        "weak_dephasing_slope_ratio_at_VHOM": round(2.0 * (1.0 + V_hom), 4),
        "gamma_sweep": sweep,
        "decay_curves": {
            "d_fine": [round(float(x), 4) for x in d_fine],
            "bg_pair_sum_0.34_2ph_norm": [round(float(np.exp(-2.0 * 0.34 * x)), 5) for x in d_fine],
            "bg_pair_sum_0.34_1ph_norm": [round(float(np.exp(-0.5 * 0.34 * x)), 5) for x in d_fine],
            "hs_pair_sum_0.79_2ph_norm": [round(float(np.exp(-2.0 * 0.79 * x)), 5) for x in d_fine],
            "hs_pair_sum_0.79_1ph_norm": [round(float(np.exp(-0.5 * 0.79 * x)), 5) for x in d_fine],
        },
    }


def run_vhom_plateau_readout(rng: np.random.Generator):
    """Finite-depth Kraus-calibrated vs asymptotic V_HOM = eta readout from dephased plateau (Fig. 7d, App. C)."""
    eta_grid = [0.70, 0.75, 0.80, 0.85, 0.90, 0.94, 0.98, 1.00]
    n_trials = 200

    def scrambling_scan(N_loc: int, depths_eval: list[int], sigma_phi: float = 1.5):
        pairs_loc, _, n_ops_loc = sym2_basis(N_loc)
        off_k = _offdiag_indices(pairs_loc)
        D2_loc = len(pairs_loc)
        U2_e = lift_sym2(clements_layer(N_loc, 0, np.pi / 4.0), pairs_loc)
        U2_o = lift_sym2(clements_layer(N_loc, 1, np.pi / 4.0), pairs_loc)
        psi0 = wf_to_fock(state_b1(N_loc, N_loc // 2 - 1, N_loc // 2)[0], pairs_loc)
        P_off_inf = (N_loc - 1.0) / (N_loc + 1.0)
        G_off_inf = 2.0 / (N_loc * (N_loc + 1.0))

        scans = {}
        for mode_key, damp in (
            ("gaussian_sigma1.5", np.exp(-dephasing_matrix(n_ops_loc, np.full(N_loc, sigma_phi**2)))),
            ("uniform_0_2pi", np.eye(D2_loc, dtype=np.float64)),
        ):
            rho = np.outer(psi0, psi0.conj())
            rows = []
            for d in range(1, max(depths_eval) + 1):
                U2 = U2_e if ((d - 1) % 2 == 0) else U2_o
                rho = (U2 @ rho @ U2.conj().T) * damp
                if d in depths_eval:
                    P_off_d = float(np.sum(np.real(np.diag(rho))[off_k]))
                    P_diag_d = 1.0 - P_off_d
                    mean_G_off_d = P_off_d / (0.5 * N_loc * (N_loc - 1.0))
                    err_P = P_off_d - P_off_inf
                    rows.append({
                        "d": d,
                        "P_off_sym_at_d": round(P_off_d, 7),
                        "P_diag_sym_at_d": round(P_diag_d, 7),
                        "P_off_sym_exact_inf": round(P_off_inf, 7),
                        "err_P_off_vs_inf": round(err_P, 7),
                        "abs_err_P_off": float(f"{abs(err_P):.3e}"),
                        "uncal_bias_eta_at_eta1": round(-(N_loc + 1.0) * err_P, 5),
                        "Gamma_off_sym_at_d": round(mean_G_off_d, 7),
                        "Gamma_off_sym_exact_inf": round(G_off_inf, 7),
                        "abs_err_Gamma_off": float(f"{abs(mean_G_off_d - G_off_inf):.3e}"),
                    })
            scans[mode_key] = rows
        return scans

    depths_scan = [6, 10, 11, 12, 16, 19, 20, 24]
    scan_N12 = scrambling_scan(12, depths_scan)
    scan_N6 = scrambling_scan(6, depths_scan)
    get_d = lambda sc, dv: [r for r in sc["gaussian_sigma1.5"] if r["d"] == dv][0]
    conv_N12_d12, conv_N6_d11, conv_N6_d19 = get_d(scan_N12, 12), get_d(scan_N6, 11), get_d(scan_N6, 19)

    def run_mesh(N_loc: int, D_scr: int, P_diag_sym_d: float, M_tot: int):
        P_off_sym_d = 1.0 - P_diag_sym_d
        cols = {k: [] for k in [
            "Gamma_off_exact", "Gamma_diag_exact", "Gamma_off_at_d",
            "Gamma_diag_over_off_exact", "Gamma_diag_over_off_at_d",
            "rho_PNR_exact", "rho_PNR_at_d",
            "eta_flux_mean", "eta_flux_std", "eta_flux_bias", "eta_flux_rmse",
            "eta_ratio_mean", "eta_ratio_std", "eta_ratio_bias", "eta_ratio_rmse",
            "analytic_sigma_flux", "analytic_sigma_ratio",
            "eta_flux_uncal_mean", "eta_flux_uncal_bias", "eta_flux_uncal_rmse",
            "eta_ratio_uncal_mean", "eta_ratio_uncal_bias", "eta_ratio_uncal_rmse",
            "analytic_uncal_bias_flux", "analytic_uncal_bias_ratio",
        ]}
        for eta in eta_grid:
            G_off_inf = 2.0 * (N_loc - eta) / (N_loc * (N_loc**2 - 1.0))
            G_diag_inf = 2.0 * (1.0 + eta) / (N_loc * (N_loc + 1.0))
            rho_pnr_inf = (1.0 + eta) / (2.0 * (N_loc - eta))

            # Finite-depth Kraus distribution: P_off^(-) = 1 by Pauli exclusion, P_diag(d) = ((1+eta)/2) P_diag^(+)(d).
            P_diag_d = 0.5 * (1.0 + eta) * P_diag_sym_d
            P_off_d = 1.0 - P_diag_d
            p_pnr_d = 0.5 * P_diag_d
            rho_pnr_d = p_pnr_d / P_off_d
            G_off_d = P_off_d / (0.5 * N_loc * (N_loc - 1.0))
            G_diag_d = (2.0 * P_diag_d) / N_loc

            sig_flux_an = (2.0 / P_diag_sym_d) * np.sqrt(P_off_d * P_diag_d / float(M_tot))
            sig_ratio_an = (
                (4.0 / (P_diag_sym_d * ((1.0 + 2.0 * rho_pnr_d) ** 2)))
                * rho_pnr_d * np.sqrt((1.0 / (M_tot * p_pnr_d)) + (1.0 / (M_tot * P_off_d)))
            )
            bias_flux_an = (N_loc - (N_loc + 1.0) * P_off_d) - eta
            bias_ratio_an = ((2.0 * N_loc * rho_pnr_d - 1.0) / (1.0 + 2.0 * rho_pnr_d)) - eta

            draws = rng.multinomial(M_tot, [P_off_d, p_pnr_d, p_pnr_d], size=n_trials)
            K_off, K_pnr = draws[:, 0].astype(np.float64), draws[:, 1].astype(np.float64)
            P_off_hat, rho_hat = K_off / float(M_tot), K_pnr / np.maximum(K_off, 1.0)

            est_flux_cal = (2.0 * (1.0 - P_off_hat) / P_diag_sym_d) - 1.0
            est_ratio_cal = (4.0 * rho_hat / (P_diag_sym_d * (1.0 + 2.0 * rho_hat))) - 1.0
            est_flux_unc = N_loc - (N_loc + 1.0) * P_off_hat
            est_ratio_unc = (2.0 * N_loc * rho_hat - 1.0) / (1.0 + 2.0 * rho_hat)

            cols["Gamma_off_exact"].append(round(float(G_off_inf), 7))
            cols["Gamma_diag_exact"].append(round(float(G_diag_inf), 7))
            cols["Gamma_off_at_d"].append(round(float(G_off_d), 7))
            cols["Gamma_diag_over_off_exact"].append(round(float(G_diag_inf / G_off_inf), 5))
            cols["Gamma_diag_over_off_at_d"].append(round(float(G_diag_d / G_off_d), 5))
            cols["rho_PNR_exact"].append(round(float(rho_pnr_inf), 6))
            cols["rho_PNR_at_d"].append(round(float(rho_pnr_d), 6))
            cols["eta_flux_mean"].append(round(float(np.mean(est_flux_cal)), 5))
            cols["eta_flux_std"].append(round(float(np.std(est_flux_cal, ddof=1)), 5))
            cols["eta_flux_bias"].append(round(float(np.mean(est_flux_cal) - eta), 5))
            cols["eta_flux_rmse"].append(round(float(np.sqrt(np.mean((est_flux_cal - eta) ** 2))), 5))
            cols["eta_ratio_mean"].append(round(float(np.mean(est_ratio_cal)), 5))
            cols["eta_ratio_std"].append(round(float(np.std(est_ratio_cal, ddof=1)), 5))
            cols["eta_ratio_bias"].append(round(float(np.mean(est_ratio_cal) - eta), 5))
            cols["eta_ratio_rmse"].append(round(float(np.sqrt(np.mean((est_ratio_cal - eta) ** 2))), 5))
            cols["analytic_sigma_flux"].append(round(float(sig_flux_an), 5))
            cols["analytic_sigma_ratio"].append(round(float(sig_ratio_an), 5))
            cols["eta_flux_uncal_mean"].append(round(float(np.mean(est_flux_unc)), 5))
            cols["eta_flux_uncal_bias"].append(round(float(np.mean(est_flux_unc) - eta), 5))
            cols["eta_flux_uncal_rmse"].append(round(float(np.sqrt(np.mean((est_flux_unc - eta) ** 2))), 5))
            cols["eta_ratio_uncal_mean"].append(round(float(np.mean(est_ratio_unc)), 5))
            cols["eta_ratio_uncal_bias"].append(round(float(np.mean(est_ratio_unc) - eta), 5))
            cols["eta_ratio_uncal_rmse"].append(round(float(np.sqrt(np.mean((est_ratio_unc - eta) ** 2))), 5))
            cols["analytic_uncal_bias_flux"].append(round(float(bias_flux_an), 5))
            cols["analytic_uncal_bias_ratio"].append(round(float(bias_ratio_an), 5))

        return {
            "N_modes": N_loc,
            "D_scrambling_layers": D_scr,
            "P_off_sym_at_d": round(P_off_sym_d, 7),
            "P_diag_sym_at_d": round(P_diag_sym_d, 7),
            "M_tot": M_tot,
            "n_mc_trials": n_trials,
            **cols,
        }

    return {
        "eta_grid": eta_grid,
        "clements_scrambling_convergence_N12": conv_N12_d12,
        "clements_scrambling_convergence_N6_d11": conv_N6_d11,
        "clements_scrambling_convergence_N6_d19": conv_N6_d19,
        "finite_depth_convergence_scan_N12": scan_N12,
        "finite_depth_convergence_scan_N6": scan_N6,
        "N12_M5e4": run_mesh(12, 12, conv_N12_d12["P_diag_sym_at_d"], 50_000),
        "N12_M5e5": run_mesh(12, 12, conv_N12_d12["P_diag_sym_at_d"], 500_000),
        "N6_submesh_M5e4": run_mesh(6, 11, conv_N6_d11["P_diag_sym_at_d"], 50_000),
        "N6_submesh_M5e5": run_mesh(6, 11, conv_N6_d11["P_diag_sym_at_d"], 500_000),
        "N6_submesh_d19_M5e5": run_mesh(6, 19, conv_N6_d19["P_diag_sym_at_d"], 500_000),
    }


def main():
    t0 = time.perf_counter()
    seed = 20261119
    rng = np.random.default_rng(seed)

    # 12-mode dephasing profile (rad^2/layer) with heater hot-spots at ports 4 and 9 (0-based 3, 8).
    gamma_true = np.array(
        [0.16, 0.18, 0.22, 0.55, 0.24, 0.19, 0.17, 0.21, 0.48, 0.20, 0.17, 0.15],
        dtype=np.float64,
    )
    V_hom, M_pairs = 0.94, 25_000

    max_g1_diff, deph_rec = run_dephasing_reconstruction(gamma_true, V_hom, M_pairs, rng)
    coupled_insitu = run_coupled_clements_insitu(gamma_true, V_hom, rng)
    sens_4x = run_sensitivity_vs_1photon(V_hom)
    vhom_ro = run_vhom_plateau_readout(rng)
    elapsed = time.perf_counter() - t0

    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, "diagnostics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {
                "seed": seed,
                "runtime_seconds": round(elapsed, 2),
                "N_modes": 12,
                "V_HOM_default": V_hom,
                "M_pairs_per_state": M_pairs,
                "max_single_realization_g1_diff": max(max_g1_diff, coupled_insitu["single_realization_max_g1_diff"]),
            },
            "dephasing_reconstruction": deph_rec,
            "coupled_clements_insitu_check": coupled_insitu,
            "sensitivity_4x_vs_single_photon": sens_4x,
            "vhom_plateau_readout": vhom_ro,
        }, f, indent=2)
    print(f"diagnostics.json ({os.path.getsize(out_path) / 1024:.1f} KB, {elapsed:.2f} s)")


if __name__ == "__main__":
    main()
