"""Realistic imperfections: nonuniform mode losses and Krein-Rutman survival filtering
(Theorem 4), partial indistinguishability V_HOM in [0, 1], and threshold-detector saturation
plus exact four-photon SPDC contamination on Sym^4(C^8) (Theorem 3) (Sec. V, Fig. 8, Table II).

Writes data/imperfections.json (strictly deterministic, no RNG).
Usage: python -I simulations/imperfections.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import numpy as np
from scipy.sparse import csr_matrix, diags as spdiags, eye as speye, kron as spkron
from scipy.sparse.linalg import eigs

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import (  # noqa: E402
    DATA_DIR,
    antisym_model,
    crossing_stats,
    gamma_ss,
    graph_small_world,
    hbt_antisym,
    hbt_observables,
    k_boson_model,
    rk4,
    state_b1,
    state_b2,
    two_boson_model,
    wf_to_fock,
)


def evolve_unconditional_2to1(
    H2_eff: np.ndarray,
    M2: np.ndarray,
    H1_eff: np.ndarray,
    M1: np.ndarray,
    B_ann: list[np.ndarray],
    kappa: np.ndarray,
    rho2_0: np.ndarray,
    dz: float,
    n_steps: int,
) -> np.ndarray:
    """Trace-preserving unconditional Lindblad evolution of coupled (rho^(2), rho^(1)) sectors
    with loss jump operators sqrt(2 kappa_j) b_j; returns D^(uncond)_{mn}(z) = Tr(b_m^dag b_n rho^(2)) + rho^(1)_{nm}.
    """
    N = H1_eff.shape[0]
    H2_dag, H1_dag = H2_eff.conj().T, H1_eff.conj().T
    O1_h2 = np.array([[B_ann[m].conj().T @ B_ann[n] for n in range(N)] for m in range(N)])

    def rhs(r2, r1):
        dr2 = -1j * (H2_eff @ r2 - r2 @ H2_dag) - M2 * r2
        jump = np.zeros((N, N), dtype=np.complex128)
        for j, kj in enumerate(kappa):
            if kj != 0.0:
                jump += (2.0 * kj) * (B_ann[j] @ r2 @ B_ann[j].conj().T)
        dr1 = -1j * (H1_eff @ r1 - r1 @ H1_dag) - M1 * r1 + jump
        return dr2, dr1

    r2, r1 = rho2_0.copy(), np.zeros((N, N), dtype=np.complex128)
    D_unc = np.empty((n_steps + 1, N, N), dtype=np.complex128)
    D_unc[0] = np.einsum("mnba,ab->mn", O1_h2, r2) + r1.T
    for s in range(1, n_steps + 1):
        k1_2, k1_1 = rhs(r2, r1)
        k2_2, k2_1 = rhs(r2 + 0.5 * dz * k1_2, r1 + 0.5 * dz * k1_1)
        k3_2, k3_1 = rhs(r2 + 0.5 * dz * k2_2, r1 + 0.5 * dz * k2_1)
        k4_2, k4_1 = rhs(r2 + dz * k3_2, r1 + dz * k3_1)
        r2 = r2 + (dz / 6.0) * (k1_2 + 2.0 * k2_2 + 2.0 * k3_2 + k4_2)
        r1 = r1 + (dz / 6.0) * (k1_1 + 2.0 * k2_1 + 2.0 * k3_1 + k4_1)
        D_unc[s] = np.einsum("mnba,ab->mn", O1_h2, r2) + r1.T
    return D_unc


def pf_leading_eigenpair(H_eff: np.ndarray, M_deph: np.ndarray, dk_bound: float = 0.5, need_left: bool = False):
    """Leading Perron-Frobenius eigenpair of L_surv(rho) = -i(H_eff rho - rho H_eff^dag) - M_deph (*) rho.
    Shift sigma = 4 dk_bound + 0.5 > lambda_0 > Re(lambda_k) isolates the unique leading eigenvalue.
    """
    D = H_eff.shape[0]
    H_sp, H_c_sp = csr_matrix(H_eff), csr_matrix(H_eff.conj())
    I_sp = speye(D, format="csr", dtype=np.complex128)
    L_sp = (
        -1j * (spkron(H_sp, I_sp, format="csr") - spkron(I_sp, H_c_sp, format="csr"))
        - spdiags(M_deph.ravel(), offsets=0, shape=(D * D, D * D), format="csr")
    )
    v0 = np.eye(D, dtype=np.complex128).ravel()
    lvec = None
    if D <= 12:
        vals, vecs = np.linalg.eig(L_sp.toarray())
        idx = int(np.argmax(vals.real))
        lam, vec = float(vals[idx].real), vecs[:, idx]
        if need_left:
            vals_l, vecs_l = np.linalg.eig(L_sp.toarray().conj().T)
            lvec = vecs_l[:, int(np.argmax(vals_l.real))]
    else:
        sigma = 4.0 * max(dk_bound, 0.05) + 0.5
        vals, vecs = eigs(L_sp, k=1, sigma=sigma, which="LM", v0=v0, tol=1e-13)
        lam, vec = float(vals[0].real), vecs[:, 0]
        if need_left:
            _, vecs_l = eigs(L_sp.conj().T, k=1, sigma=sigma, which="LM", v0=v0, tol=1e-13)
            lvec = vecs_l[:, 0]

    rho_pf = vec.reshape((D, D))
    if abs(np.trace(rho_pf)) > 1e-15:
        rho_pf = rho_pf / np.trace(rho_pf)
    rho_pf = 0.5 * (rho_pf + rho_pf.conj().T)
    rho_pf = rho_pf / np.real(np.trace(rho_pf))

    L0 = None
    if need_left and lvec is not None:
        L0 = 0.5 * (lvec.reshape((D, D)) + lvec.reshape((D, D)).conj().T)
        ov = np.real(np.trace(L0.conj().T @ rho_pf))
        if abs(ov) > 1e-15:
            L0 = L0 / ov
    return lam, rho_pf, L0


def run_losses_and_vhom():
    """Nonuniform loss scan |delta_kappa|/J in [0, 0.50] and partial indistinguishability V_HOM in [0, 1] (Sec. V.B)."""
    J, gamma_0 = 1.0, 0.45
    dk_grid = [
        0.0, 0.002, 0.003, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05,
        0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
    ]

    # N = 2 dimer Krein-Rutman gap vs perturbative Theorem 4 formula (8 gamma / (3 J^2)) delta_kappa^2.
    adj2 = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    dimer_scan = []
    for dk in dk_grid:
        kap2 = np.array([-dk * J, +dk * J], dtype=np.float64)
        _, _, _, _, H2_eff, M2, _, _, _, _ = two_boson_model(adj2, J, np.full(2, gamma_0), kappa=kap2)
        lam_p, _, _ = pf_leading_eigenpair(H2_eff, M2)
        gap_pert = (8.0 * gamma_0 / (3.0 * J * J)) * ((dk * J) ** 2)
        rel_err = abs(lam_p - gap_pert) / lam_p if lam_p > 1e-14 else 0.0

        # Unbroken PT-symmetric phase at gamma = 0 (|dk| < J => Delta lambda_PF = 0).
        _, _, _, _, H2_pt, M2_pt, _, _, _, _ = two_boson_model(adj2, J, np.zeros(2), kappa=kap2)
        lam_p_pt, _, _ = pf_leading_eigenpair(H2_pt, M2_pt)
        dimer_scan.append({
            "delta_kappa_over_J": dk,
            "Delta_lambda_PF_exact": round(float(lam_p), 8),
            "Delta_lambda_PF_pert_8g_3J2_dk2": round(float(gap_pert), 8),
            "rel_err_pert_vs_exact": round(float(rel_err), 6),
            "Delta_lambda_PF_gamma0_PT_unbroken": round(float(lam_p_pt), 10),
        })

    # N = 8 small-world graph nonuniform loss scan.
    adj8 = graph_small_world(8)
    N = 8
    gamma_vec8 = np.full(N, gamma_0)
    raw_prof = np.array([-1.0, 0.85, -0.45, 0.65, -0.90, 0.95, -0.35, 0.25], dtype=np.float64)
    raw_prof = (raw_prof - np.mean(raw_prof)) / np.max(np.abs(raw_prof - np.mean(raw_prof)))

    G_ideal_sym = gamma_ss(N)
    G_ideal_anti = np.full((N, N), 2.0 / (N * (N - 1)))
    np.fill_diagonal(G_ideal_anti, 0.0)
    off_mask = ~np.eye(N, dtype=bool)

    a2, b2 = 4, 5
    a3, b3, c3 = 7, 0, 6
    Wh_2p, Wnp_2p = state_b1(N, a2, b2, noon_sign=+1.0)
    _, Wnm_2p = state_b1(N, a2, b2, noon_sign=-1.0)
    W1_3p, W2_3p = state_b2(N, a3, b3, c3, sign=1.0)

    dz, n_steps = 0.025, 600
    z_grid = np.linspace(0.0, dz * n_steps, n_steps + 1)
    dense_vs_sparse_err = None
    vhom_cache = {}
    n8_scan = []

    for dk in dk_grid:
        dk_vec = dk * J * raw_prof
        kap_phys = max(dk * J, 0.05) + 0.05 + dk_vec

        _, _, M1_d, _, H2_eff_0, M_s, n_s, pairs, _, B_ann = two_boson_model(adj8, J, gamma_vec8, kappa=dk_vec)
        _, H1_eff_phys, _, _, H2_eff_phys, _, _, _, _, _ = two_boson_model(adj8, J, gamma_vec8, kappa=kap_phys)
        _, Ha_eff_0, M_a, n_a, apairs, idx_a = antisym_model(adj8, J, gamma_vec8, kappa=dk_vec)
        D_s, D_a = len(pairs), len(apairs)

        lam_p, rho_pf_p, _ = pf_leading_eigenpair(H2_eff_0, M_s, dk_bound=dk)
        lam_m, _, _ = pf_leading_eigenpair(Ha_eff_0, M_a, dk_bound=dk)
        G_pf_p = hbt_observables(rho_pf_p[None], n_s, pairs, N)[1][0]

        if abs(dk - 0.03) < 1e-12:
            Is = np.eye(D_s, dtype=np.complex128)
            Ls_dense = -1j * (np.kron(H2_eff_0, Is) - np.kron(Is, H2_eff_0.conj())) - np.diag(M_s.ravel())
            dense_vs_sparse_err = float(abs(lam_p - np.max(np.linalg.eigvals(Ls_dense).real)))

        p_hom, p_np, p_nm = wf_to_fock(Wh_2p, pairs), wf_to_fock(Wnp_2p, pairs), wf_to_fock(Wnm_2p, pairs)
        p_1, p_2 = wf_to_fock(W1_3p, pairs), wf_to_fock(W2_3p, pairs)

        # Antisymmetric states are invariant under passive U(2) preparation.
        p_2p_a = np.zeros(D_a, dtype=np.complex128)
        p_2p_a[idx_a[(a2, b2)]] = 1.0
        p_3p_a = np.zeros(D_a, dtype=np.complex128)
        for p_oth in (b3, c3):
            u, v = (a3, p_oth) if a3 < p_oth else (p_oth, a3)
            p_3p_a[idx_a[(u, v)]] += (1.0 if a3 < p_oth else -1.0) / np.sqrt(2.0)

        tr_hom_s = rk4(H2_eff_0, M_s, np.outer(p_hom, p_hom.conj()), dz, n_steps, hermitian=False)
        tr_np_s = rk4(H2_eff_0, M_s, np.outer(p_np, p_np.conj()), dz, n_steps, hermitian=False)
        tr_nm_s = rk4(H2_eff_0, M_s, np.outer(p_nm, p_nm.conj()), dz, n_steps, hermitian=False)
        tr_1_s = rk4(H2_eff_0, M_s, np.outer(p_1, p_1.conj()), dz, n_steps, hermitian=False)
        tr_2_s = rk4(H2_eff_0, M_s, np.outer(p_2, p_2.conj()), dz, n_steps, hermitian=False)
        tr_2p_a = rk4(Ha_eff_0, M_a, np.outer(p_2p_a, p_2p_a.conj()), dz, n_steps, hermitian=False)
        tr_3p_a = rk4(Ha_eff_0, M_a, np.outer(p_3p_a, p_3p_a.conj()), dz, n_steps, hermitian=False)

        surv_hom_s = np.real(np.trace(tr_hom_s, axis1=1, axis2=2))
        surv_np_s = np.real(np.trace(tr_np_s, axis1=1, axis2=2))
        surv_nm_s = np.real(np.trace(tr_nm_s, axis1=1, axis2=2))
        surv_1_s = np.real(np.trace(tr_1_s, axis1=1, axis2=2))
        surv_2_s = np.real(np.trace(tr_2_s, axis1=1, axis2=2))
        surv_2p_a = np.real(np.trace(tr_2p_a, axis1=1, axis2=2))
        surv_3p_a = np.real(np.trace(tr_3p_a, axis1=1, axis2=2))

        I_hom_s, G_hom_s = hbt_observables(tr_hom_s, n_s, pairs, N)
        I_np_s, G_np_s = hbt_observables(tr_np_s, n_s, pairs, N)
        I_nm_s, G_nm_s = hbt_observables(tr_nm_s, n_s, pairs, N)
        I_1_s, G_1_s = hbt_observables(tr_1_s, n_s, pairs, N)
        I_2_s, G_2_s = hbt_observables(tr_2_s, n_s, pairs, N)
        I_2p_a, G_2p_a = hbt_antisym(tr_2p_a, n_a, apairs, N)
        I_3p_a, G_3p_a = hbt_antisym(tr_3p_a, n_a, apairs, N)

        if dk in (0.0, 0.003, 0.03, 0.08):
            vhom_cache[dk] = {
                "surv_hom_s": surv_hom_s, "surv_np_s": surv_np_s, "surv_nm_s": surv_nm_s,
                "surv_1_s": surv_1_s, "surv_2_s": surv_2_s, "surv_2p_a": surv_2p_a, "surv_3p_a": surv_3p_a,
                "G_hom_s": G_hom_s, "G_np_s": G_np_s, "G_nm_s": G_nm_s,
                "G_1_s": G_1_s, "G_2_s": G_2_s, "G_2p_a": G_2p_a, "G_3p_a": G_3p_a,
            }

        # Unconditional Delta D^(uncond)(z) == 0 check retaining 2 -> 1 loss jumps.
        D_u1 = evolve_unconditional_2to1(H2_eff_phys, M_s, H1_eff_phys, M1_d, B_ann, kap_phys, np.outer(p_1, p_1.conj()), dz, 200)
        D_u2 = evolve_unconditional_2to1(H2_eff_phys, M_s, H1_eff_phys, M1_d, B_ann, kap_phys, np.outer(p_2, p_2.conj()), dz, 200)
        uncond_rdm_diff = float(np.max(np.abs(D_u1 - D_u2)))

        def eval_vhom(eta: float):
            wp, wm = 0.5 * (1.0 + eta), 0.5 * (1.0 - eta)
            n_hom = wp * surv_hom_s + wm * surv_2p_a
            n_np = wp * surv_np_s + wm * surv_2p_a
            n_nm = wp * surv_nm_s + wm * surv_2p_a
            n_1 = wp * surv_1_s + wm * surv_3p_a
            n_2 = wp * surv_2_s + wm * surv_3p_a

            post_I1 = (wp * I_1_s + wm * I_3p_a) / n_1[:, None]
            post_I2 = (wp * I_2_s + wm * I_3p_a) / n_2[:, None]
            post_Ih = (wp * I_hom_s + wm * I_2p_a) / n_hom[:, None]
            post_Inp = (wp * I_np_s + wm * I_2p_a) / n_np[:, None]
            post_Inm = (wp * I_nm_s + wm * I_2p_a) / n_nm[:, None]

            Gt_hom = (wp * G_hom_s + wm * G_2p_a) / n_hom[:, None, None]
            Gt_np = (wp * G_np_s + wm * G_2p_a) / n_np[:, None, None]
            Gt_nm = (wp * G_nm_s + wm * G_2p_a) / n_nm[:, None, None]
            Gt_1 = (wp * G_1_s + wm * G_3p_a) / n_1[:, None, None]
            Gt_2 = (wp * G_2_s + wm * G_3p_a) / n_2[:, None, None]

            G_ideal = wp * G_ideal_sym + wm * G_ideal_anti
            G_emp_np = 0.5 * (Gt_hom[-1] + Gt_np[-1])
            G_emp_B2 = 0.5 * (Gt_1[-1] + Gt_2[-1])

            return {
                "unnormalized_surv_2ph_intensity_diff_B2": round(float(np.max(np.abs(wp * (I_1_s - I_2_s)))), 6),
                "unnormalized_surv_2ph_intensity_diff_HOM_NOONp": round(float(np.max(np.abs(wp * (I_hom_s - I_np_s)))), 6),
                "postselected_intensity_leakage_B2": round(float(np.max(np.abs(post_I1 - post_I2))), 6),
                "postselected_intensity_leakage_HOM_NOONp": round(float(np.max(np.abs(post_Ih - post_Inp))), 6),
                "postselected_intensity_leakage_HOM_NOONm": round(float(np.max(np.abs(post_Ih - post_Inm))), 6),
                "mesoscopic_plateau_shift_Jz15_HOM_NOONp": round(float(np.linalg.norm(Gt_hom[-1] - Gt_np[-1])), 7),
                "mesoscopic_plateau_shift_Jz15_B2": round(float(np.linalg.norm(Gt_1[-1] - Gt_2[-1])), 7),
                "HOM_NOONp_offdiag_uncal": crossing_stats(
                    np.linalg.norm((Gt_hom - G_ideal[None])[:, off_mask], axis=1),
                    np.linalg.norm((Gt_np - G_ideal[None])[:, off_mask], axis=1), z_grid, tol=1e-6,
                ),
                "HOM_NOONp_offdiag_cal": crossing_stats(
                    np.linalg.norm((Gt_hom - G_emp_np[None])[:, off_mask], axis=1),
                    np.linalg.norm((Gt_np - G_emp_np[None])[:, off_mask], axis=1), z_grid, tol=1e-6,
                ),
                "HOM_NOONm_offdiag_uncal": crossing_stats(
                    np.linalg.norm((Gt_hom - G_ideal[None])[:, off_mask], axis=1),
                    np.linalg.norm((Gt_nm - G_ideal[None])[:, off_mask], axis=1), z_grid, tol=1e-6,
                ),
                "B2_fullHBT_uncal": crossing_stats(
                    np.linalg.norm(Gt_1 - G_ideal[None], axis=(1, 2)),
                    np.linalg.norm(Gt_2 - G_ideal[None], axis=(1, 2)), z_grid, tol=1e-6,
                ),
                "B2_fullHBT_cal": crossing_stats(
                    np.linalg.norm(Gt_1 - G_emp_B2[None], axis=(1, 2)),
                    np.linalg.norm(Gt_2 - G_emp_B2[None], axis=(1, 2)), z_grid, tol=1e-6,
                ),
            }

        n8_scan.append({
            "delta_kappa_over_J": dk,
            "lambda0_sym_plus": round(float(lam_p), 7),
            "lambda0_antisym_minus": round(float(lam_m), 7),
            "Delta_lambda_PF": round(float(lam_p - lam_m), 7),
            "PF_steady_shift_from_uniform_Frob": round(float(np.linalg.norm(G_pf_p - G_ideal_sym)), 6),
            "full_unconditional_1body_RDM_diff_B2": uncond_rdm_diff,
            "V_HOM_1.00": eval_vhom(1.00),
            "V_HOM_0.94": eval_vhom(0.94),
        })

    dks_fit = np.array([r["delta_kappa_over_J"] for r in n8_scan if 0.0 < r["delta_kappa_over_J"] <= 0.04])
    slope_B2_100 = float(np.dot(dks_fit, [r["V_HOM_1.00"]["postselected_intensity_leakage_B2"] for r in n8_scan if 0.0 < r["delta_kappa_over_J"] <= 0.04]) / np.dot(dks_fit, dks_fit))
    slope_B2_094 = float(np.dot(dks_fit, [r["V_HOM_0.94"]["postselected_intensity_leakage_B2"] for r in n8_scan if 0.0 < r["delta_kappa_over_J"] <= 0.04]) / np.dot(dks_fit, dks_fit))
    slope_2p_094 = float(np.dot(dks_fit, [r["V_HOM_0.94"]["postselected_intensity_leakage_HOM_NOONp"] for r in n8_scan if 0.0 < r["delta_kappa_over_J"] <= 0.04]) / np.dot(dks_fit, dks_fit))

    # Bosonic purification trajectory p_+(z) -> 1 (dk/J = 0.25, V_HOM = 0.85, Theorem 4).
    dk_pur, eta_pur = 0.25, 0.85
    dk_vec_pur = dk_pur * J * raw_prof
    _, _, _, _, H2_pur, M_s_pur, _, pairs_p, _, _ = two_boson_model(adj8, J, gamma_vec8, kappa=dk_vec_pur)
    _, Ha_pur, M_a_pur, _, apairs_p, idx_a_p = antisym_model(adj8, J, gamma_vec8, kappa=dk_vec_pur)
    lam_p_pur, _, L0_p_pur = pf_leading_eigenpair(H2_pur, M_s_pur, dk_bound=dk_pur, need_left=True)
    lam_m_pur, _, L0_m_pur = pf_leading_eigenpair(Ha_pur, M_a_pur, dk_bound=dk_pur, need_left=True)

    p1_pur = wf_to_fock(W1_3p, pairs_p)
    p3a_pur = np.zeros(len(apairs_p), dtype=np.complex128)
    for p_oth in (b3, c3):
        u, v = (a3, p_oth) if a3 < p_oth else (p_oth, a3)
        p3a_pur[idx_a_p[(u, v)]] += (1.0 if a3 < p_oth else -1.0) / np.sqrt(2.0)

    rho_p0, rho_m0 = np.outer(p1_pur, p1_pur.conj()), np.outer(p3a_pur, p3a_pur.conj())
    w_p = float(np.real(np.trace(L0_p_pur.conj().T @ rho_p0)))
    w_m = float(np.real(np.trace(L0_m_pur.conj().T @ rho_m0)))

    dz_pur, n_pur, stride_pur = 0.04, 2000, 25
    z_pur = np.linspace(0.0, dz_pur * n_pur, n_pur + 1)[::stride_pur]
    sp_pur = np.real(np.trace(rk4(H2_pur, M_s_pur, rho_p0, dz_pur, n_pur, hermitian=False)[::stride_pur], axis1=1, axis2=2))
    sm_pur = np.real(np.trace(rk4(Ha_pur, M_a_pur, rho_m0, dz_pur, n_pur, hermitian=False)[::stride_pur], axis1=1, axis2=2))
    wp_e, wm_e = 0.5 * (1.0 + eta_pur), 0.5 * (1.0 - eta_pur)
    p_plus_exact = (wp_e * sp_pur) / (wp_e * sp_pur + wm_e * sm_pur)
    dlam_pur = lam_p_pur - lam_m_pur
    exp_fac = np.exp(dlam_pur * z_pur)
    p_plus_asymp = (wp_e * w_p * exp_fac) / (wp_e * w_p * exp_fac + wm_e * w_m)

    # Partial indistinguishability scan V_HOM in [0, 1] at dk/J in {0, 0.003, 0.03, 0.08}.
    eta_grid = sorted(set([round(0.05 * k, 4) for k in range(21)] + [0.94, 0.98]))
    vhom_scan = {}
    for dk in (0.0, 0.003, 0.03, 0.08):
        c = vhom_cache[dk]
        rows = []
        for eta in eta_grid:
            wp, wm = 0.5 * (1.0 + eta), 0.5 * (1.0 - eta)
            Gt_hom = (wp * c["G_hom_s"] + wm * c["G_2p_a"]) / (wp * c["surv_hom_s"] + wm * c["surv_2p_a"])[:, None, None]
            Gt_np = (wp * c["G_np_s"] + wm * c["G_2p_a"]) / (wp * c["surv_np_s"] + wm * c["surv_2p_a"])[:, None, None]
            Gt_nm = (wp * c["G_nm_s"] + wm * c["G_2p_a"]) / (wp * c["surv_nm_s"] + wm * c["surv_2p_a"])[:, None, None]
            Gt_1 = (wp * c["G_1_s"] + wm * c["G_3p_a"]) / (wp * c["surv_1_s"] + wm * c["surv_3p_a"])[:, None, None]
            Gt_2 = (wp * c["G_2_s"] + wm * c["G_3p_a"]) / (wp * c["surv_2_s"] + wm * c["surv_3p_a"])[:, None, None]

            G_emp_np = 0.5 * (Gt_hom[-1] + Gt_np[-1])
            G_emp_B2 = 0.5 * (Gt_1[-1] + Gt_2[-1])
            dh = np.linalg.norm((Gt_hom - G_emp_np[None])[:, off_mask], axis=1)
            dnp = np.linalg.norm((Gt_np - G_emp_np[None])[:, off_mask], axis=1)
            dnm = np.linalg.norm((Gt_nm - G_emp_np[None])[:, off_mask], axis=1)
            c_np = crossing_stats(dh, dnp, z_grid, tol=1e-6)
            c_nm = crossing_stats(dh, dnm, z_grid, tol=1e-6)
            c_b2_off = crossing_stats(
                np.linalg.norm((Gt_1 - G_emp_B2[None])[:, off_mask], axis=1),
                np.linalg.norm((Gt_2 - G_emp_B2[None])[:, off_mask], axis=1), z_grid, tol=1e-6,
            )
            c_b2_full = crossing_stats(
                np.linalg.norm(Gt_1 - G_emp_B2[None], axis=(1, 2)),
                np.linalg.norm(Gt_2 - G_emp_B2[None], axis=(1, 2)), z_grid, tol=1e-6,
            )
            rows.append({
                "V_HOM": eta,
                "pi_plus": round(wp, 5),
                "pi_minus": round(wm, 5),
                "HOM_NOONp_offdiag_gap": c_np["max_inversion_gap"] if c_np else 0.0,
                "HOM_NOONp_offdiag_Jz_star": c_np["first_crossing"] if c_np else None,
                "HOM_NOONp_offdiag_rel_pct": c_np["rel_inversion_pct"] if c_np else 0.0,
                "HOM_NOONm_offdiag_gap": c_nm["max_inversion_gap"] if c_nm else 0.0,
                "B2_offdiag_gap": c_b2_off["max_inversion_gap"] if c_b2_off else 0.0,
                "B2_offdiag_order": c_b2_off["order"] if c_b2_off else None,
                "B2_fullHBT_gap": c_b2_full["max_inversion_gap"] if c_b2_full else 0.0,
                "B2_fullHBT_order": c_b2_full["order"] if c_b2_full else None,
                "mesoscopic_plateau_shift_HOM_NOONp": round(float(np.linalg.norm(Gt_hom[-1] - Gt_np[-1])), 7),
                "mesoscopic_plateau_shift_B2": round(float(np.linalg.norm(Gt_1[-1] - Gt_2[-1])), 7),
                "scaling_param_1mEta_dk": round((1.0 - eta) * dk, 7),
            })
        vhom_scan[f"dk_{dk:.3f}"] = rows

    return {
        "dimer_N2_krein_rutman_scan": dimer_scan,
        "network_N8_krein_rutman_and_leakage_scan": n8_scan,
        "sparse_vs_dense_pf_check_err_at_dk0.03": dense_vs_sparse_err,
        "postselected_leakage_linear_slopes": {
            "B2_VHOM_1.00": round(slope_B2_100, 5),
            "B2_VHOM_0.94": round(slope_B2_094, 5),
            "HOM_NOONp_VHOM_0.94": round(slope_2p_094, 5),
        },
        "krein_rutman_bosonic_purification_curve": {
            "N": N,
            "delta_kappa_over_J": dk_pur,
            "V_HOM": eta_pur,
            "Delta_lambda_PF": round(float(dlam_pur), 7),
            "Jz_grid": [round(float(x), 3) for x in z_pur],
            "p_plus_exact": [round(float(x), 6) for x in p_plus_exact],
            "p_plus_theorem4_asymptotic": [round(float(x), 6) for x in p_plus_asymp],
        },
    }, vhom_scan


def run_threshold_and_spdc():
    """Threshold-detector single-click saturation and exact 4-photon SPDC contamination on Sym^4(C^8) (Sec. V.A)."""
    adj8 = graph_small_world(8)
    N, J, gamma_0 = 8, 1.0, 0.45
    gamma_vec = np.full(N, gamma_0)

    _, _, _, H2, _, M2, n2, pairs2, idx2, _ = two_boson_model(adj8, J, gamma_vec)
    H4, M4, n4, tuples4, occ4_to_idx = k_boson_model(adj8, J, gamma_vec, k_photons=4)
    D2, D4 = len(pairs2), len(tuples4)
    occ2_to_idx = {tuple(n2[:, idx].astype(int)): idx for idx in range(D2)}

    def apply_unitary_fock(U1: np.ndarray, init_modes: list[int], tuples_k, occ_map_k) -> np.ndarray:
        state = {tuple([0] * N): 1.0 + 0.0j}
        for m_in in init_modes:
            nxt = {}
            for occ, amp in state.items():
                if abs(amp) < 1e-15:
                    continue
                for q in range(N):
                    u_qm = U1[q, m_in]
                    if abs(u_qm) < 1e-15:
                        continue
                    new_occ = list(occ)
                    new_occ[q] += 1
                    t_new = tuple(new_occ)
                    nxt[t_new] = nxt.get(t_new, 0.0j) + amp * u_qm * np.sqrt(float(new_occ[q]))
            state = nxt
        counts_in = {}
        for m_in in init_modes:
            counts_in[m_in] = counts_in.get(m_in, 0) + 1
        norm_in = 1.0
        for c_val in counts_in.values():
            for f in range(1, c_val + 1):
                norm_in *= np.sqrt(float(f))
        psi_k = np.zeros(len(tuples_k), dtype=np.complex128)
        for t_occ, amp in state.items():
            psi_k[occ_map_k[t_occ]] = amp / norm_in
        return psi_k

    dz, n_steps = 0.025, 240
    z_grid = np.linspace(0.0, dz * n_steps, n_steps + 1)
    off_mask = ~np.eye(N, dtype=bool)

    a2, b2 = 4, 5
    U_id = np.eye(N, dtype=np.complex128)
    U_bs_m = np.eye(N, dtype=np.complex128)
    U_bs_m[np.ix_([a2, b2], [a2, b2])] = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    U_bs_p = np.eye(N, dtype=np.complex128)
    U_bs_p[np.ix_([a2, b2], [a2, b2])] = np.array([[1.0, 1j], [1j, 1.0]]) / np.sqrt(2.0)

    psi2_hom = apply_unitary_fock(U_id, [a2, b2], pairs2, occ2_to_idx)
    psi2_nm = apply_unitary_fock(U_bs_m, [a2, b2], pairs2, occ2_to_idx)
    psi2_np = apply_unitary_fock(U_bs_p, [a2, b2], pairs2, occ2_to_idx)
    psi4_hom = apply_unitary_fock(U_id, [a2, a2, b2, b2], tuples4, occ4_to_idx)
    psi4_nm = apply_unitary_fock(U_bs_m, [a2, a2, b2, b2], tuples4, occ4_to_idx)
    psi4_np = apply_unitary_fock(U_bs_p, [a2, a2, b2, b2], tuples4, occ4_to_idx)

    a3, b3, c3 = 7, 0, 6
    U_bs_ab = np.eye(N, dtype=np.complex128)
    U_bs_ab[np.ix_([a3, b3], [a3, b3])] = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    U_ph_b = np.eye(N, dtype=np.complex128)
    U_ph_b[b3, b3] = 1j
    U_bs_bc = np.eye(N, dtype=np.complex128)
    U_bs_bc[np.ix_([b3, c3], [b3, c3])] = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    U_p1, U_p2 = U_bs_bc @ U_ph_b @ U_bs_ab, U_bs_bc

    psi2_p1 = apply_unitary_fock(U_p1, [a3, b3], pairs2, occ2_to_idx)
    psi2_p2 = apply_unitary_fock(U_p2, [a3, b3], pairs2, occ2_to_idx)
    psi4_p1 = apply_unitary_fock(U_p1, [a3, a3, b3, b3], tuples4, occ4_to_idx)
    psi4_p2 = apply_unitary_fock(U_p2, [a3, a3, b3, b3], tuples4, occ4_to_idx)

    psi2_adj = np.zeros(D2, dtype=np.complex128)
    psi2_adj[idx2[(2, 3)]] = psi2_adj[idx2[(4, 5)]] = 1.0 / np.sqrt(2.0)
    psi2_cz4 = np.zeros(D2, dtype=np.complex128)
    psi2_cz4[idx2[(2, 4)]] = psi2_cz4[idx2[(2, 5)]] = psi2_cz4[idx2[(3, 4)]] = 0.5
    psi2_cz4[idx2[(3, 5)]] = -0.5

    diag_pops = lambda H_m, M_m, psi: np.real(np.diagonal(rk4(H_m, M_m, np.outer(psi, psi.conj()), dz, n_steps), axis1=1, axis2=2))
    pops2_hom, pops2_nm, pops2_np = diag_pops(H2, M2, psi2_hom), diag_pops(H2, M2, psi2_nm), diag_pops(H2, M2, psi2_np)
    pops2_p1, pops2_p2 = diag_pops(H2, M2, psi2_p1), diag_pops(H2, M2, psi2_p2)
    pops2_adj, pops2_cz4 = diag_pops(H2, M2, psi2_adj), diag_pops(H2, M2, psi2_cz4)
    pops4_hom, pops4_nm, pops4_np = diag_pops(H4, M4, psi4_hom), diag_pops(H4, M4, psi4_nm), diag_pops(H4, M4, psi4_np)
    pops4_p1, pops4_p2 = diag_pops(H4, M4, psi4_p1), diag_pops(H4, M4, psi4_p2)

    def click_tensors(n_mat: np.ndarray, eta_det: float):
        # Threshold POVM: P(click_q | n) = 1 - (1 - eta)^n_q.
        p_no = (1.0 - eta_det) ** n_mat
        p_cl = 1.0 - p_no
        inc = np.zeros((n_mat.shape[1], N, N), dtype=np.float64)
        excl = np.zeros((n_mat.shape[1], N, N), dtype=np.float64)
        for q in range(N):
            for r in range(q + 1, N):
                both = p_cl[q] * p_cl[r]
                mask_other = np.ones(N, dtype=bool)
                mask_other[q] = mask_other[r] = False
                inc[:, q, r] = inc[:, r, q] = both
                excl[:, q, r] = excl[:, r, q] = both * np.prod(p_no[mask_other], axis=0)
        return p_cl.T, inc, excl

    sat_scan = []
    for eta_t in [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]:
        cl2, _, _ = click_tensors(n2, eta_t)
        S_hom, S_np = pops2_hom @ cl2, pops2_np @ cl2
        S_p1, S_p2 = pops2_p1 @ cl2, pops2_p2 @ cl2
        S_adj, S_cz4 = pops2_adj @ cl2, pops2_cz4 @ cl2
        sat_scan.append({
            "eta_tot": eta_t,
            "analytic_B1_B2_z0_pct": round(50.0 * eta_t, 4),
            "B1_rel_err_z0_pct": round(100.0 * float(np.max(np.abs(S_hom[0] - S_np[0])) / np.max(S_hom[0])), 4),
            "B1_rel_err_max_z_pct": round(100.0 * float(np.max(np.abs(S_hom - S_np)) / np.max(S_hom)), 4),
            "B2_rel_err_z0_pct": round(100.0 * float(np.max(np.abs(S_p1[0] - S_p2[0])) / np.max(S_p2[0])), 4),
            "B2_rel_err_max_z_pct": round(100.0 * float(np.max(np.abs(S_p1 - S_p2)) / np.max(S_p2)), 4),
            "B2_CZ4_rel_err_z0_pct": round(100.0 * float(np.max(np.abs(S_adj[0] - S_cz4[0])) / np.max(S_adj[0])), 6),
            "B2_CZ4_rel_err_max_z_pct": round(100.0 * float(np.max(np.abs(S_adj - S_cz4)) / np.max(S_adj)), 4),
        })

    ss2, ss4 = np.full((1, D2), 1.0 / D2), np.full((1, D4), 1.0 / D4)
    G_ideal_off = np.full((N, N), 2.0 / (N * (N + 1)))
    np.fill_diagonal(G_ideal_off, 0.0)
    G_ideal_off_norm = float(np.linalg.norm(G_ideal_off[off_mask]))

    spdc_by_eta, curves_eta025 = {}, {}
    stride_curve = 2
    for eta_t in [0.08, 0.25, 0.50, 1.00]:
        _, inc2, excl2 = click_tensors(n2, eta_t)
        _, inc4, excl4 = click_tensors(n4.astype(np.float64), eta_t)

        # Theorem 3 exact inclusive and 4-class exclusive 2-click plateau shift factors.
        thm3_inc = 6.0 - 24.0 * eta_t / (N + 2.0) + 36.0 * (eta_t**2) / ((N + 2.0) * (N + 3.0))
        thm3_exc = (12.0 / ((N + 2.0) * (N + 3.0))) * (
            0.5 * (N - 1.0) * (N - 2.0) * ((1.0 - eta_t) ** 2)
            + 2.0 * (N - 2.0) * (2.0 - eta_t) * (1.0 - eta_t)
            + 2.0 * (3.0 - 3.0 * eta_t + eta_t**2)
            + ((2.0 - eta_t) ** 2)
        )

        contract = lambda pops, ten: np.einsum("td,dqr->tqr", pops, ten)
        i2_h, i2_np, i2_nm = contract(pops2_hom, inc2), contract(pops2_np, inc2), contract(pops2_nm, inc2)
        i2_p1, i2_p2, i2_ss = contract(pops2_p1, inc2), contract(pops2_p2, inc2), contract(ss2, inc2)
        i4_h, i4_np, i4_nm = contract(pops4_hom, inc4), contract(pops4_np, inc4), contract(pops4_nm, inc4)
        i4_p1, i4_p2, i4_ss = contract(pops4_p1, inc4), contract(pops4_p2, inc4), contract(ss4, inc4)

        e2_h, e2_np, e2_nm = contract(pops2_hom, excl2), contract(pops2_np, excl2), contract(pops2_nm, excl2)
        e2_p1, e2_p2, e2_ss = contract(pops2_p1, excl2), contract(pops2_p2, excl2), contract(ss2, excl2)
        e4_h, e4_np, e4_nm = contract(pops4_hom, excl4), contract(pops4_np, excl4), contract(pops4_nm, excl4)
        e4_p1, e4_p2, e4_ss = contract(pops4_p1, excl4), contract(pops4_p2, excl4), contract(ss4, excl4)

        rows = []
        for p4p2 in [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]:
            lin_diff_B2 = float(np.max(np.abs((pops2_p1 @ n2.T + p4p2 * (pops4_p1 @ n4.T)) - (pops2_p2 @ n2.T + p4p2 * (pops4_p2 @ n4.T)))))
            lin_diff_B1 = float(np.max(np.abs((pops2_hom @ n2.T + p4p2 * (pops4_hom @ n4.T)) - (pops2_np @ n2.T + p4p2 * (pops4_np @ n4.T)))))

            C_h, C_np, C_nm = (i2_h + p4p2 * i4_h) / (eta_t**2), (i2_np + p4p2 * i4_np) / (eta_t**2), (i2_nm + p4p2 * i4_nm) / (eta_t**2)
            C_p1, C_p2, C_ss = (i2_p1 + p4p2 * i4_p1) / (eta_t**2), (i2_p2 + p4p2 * i4_p2) / (eta_t**2), (i2_ss + p4p2 * i4_ss) / (eta_t**2)
            shift_num = float(np.linalg.norm((C_ss[0] - G_ideal_off)[off_mask]))

            E_h, E_np, E_nm = e2_h + p4p2 * e4_h, e2_np + p4p2 * e4_np, e2_nm + p4p2 * e4_nm
            E_p1, E_p2, E_ss = e2_p1 + p4p2 * e4_p1, e2_p2 + p4p2 * e4_p2, e2_ss + p4p2 * e4_ss
            excl_shift = float(np.linalg.norm((E_ss[0] / (eta_t**2) - G_ideal_off)[off_mask]))

            dh_u = np.linalg.norm((C_h - G_ideal_off[None])[:, off_mask], axis=1)
            dnp_u = np.linalg.norm((C_np - G_ideal_off[None])[:, off_mask], axis=1)
            dnm_u = np.linalg.norm((C_nm - G_ideal_off[None])[:, off_mask], axis=1)
            dp1_u = np.linalg.norm((C_p1 - G_ideal_off[None])[:, off_mask], axis=1)
            dp2_u = np.linalg.norm((C_p2 - G_ideal_off[None])[:, off_mask], axis=1)

            dh_c = np.linalg.norm((C_h - C_ss)[:, off_mask], axis=1)
            dnp_c = np.linalg.norm((C_np - C_ss)[:, off_mask], axis=1)
            dnm_c = np.linalg.norm((C_nm - C_ss)[:, off_mask], axis=1)
            dp1_c = np.linalg.norm((C_p1 - C_ss)[:, off_mask], axis=1)
            dp2_c = np.linalg.norm((C_p2 - C_ss)[:, off_mask], axis=1)

            P_ss = E_ss / np.sum(E_ss[:, off_mask], axis=1)[:, None, None]
            P_h = E_h[2:] / np.sum(E_h[2:][:, off_mask], axis=1)[:, None, None]
            P_np = E_np[2:] / np.sum(E_np[2:][:, off_mask], axis=1)[:, None, None]
            P_nm = E_nm[2:] / np.sum(E_nm[2:][:, off_mask], axis=1)[:, None, None]
            P_p1 = E_p1 / np.sum(E_p1[:, off_mask], axis=1)[:, None, None]
            P_p2 = E_p2 / np.sum(E_p2[:, off_mask], axis=1)[:, None, None]

            dh_s = np.linalg.norm((P_h - P_ss)[:, off_mask], axis=1)
            dnp_s = np.linalg.norm((P_np - P_ss)[:, off_mask], axis=1)
            dnm_s = np.linalg.norm((P_nm - P_ss)[:, off_mask], axis=1)
            dp1_s = np.linalg.norm((P_p1 - P_ss)[:, off_mask], axis=1)
            dp2_s = np.linalg.norm((P_p2 - P_ss)[:, off_mask], axis=1)

            rows.append({
                "p4_over_p2": p4p2,
                "true_linear_intensity_diff_B1": lin_diff_B1,
                "true_linear_intensity_diff_B2": lin_diff_B2,
                "inclusive_plateau_shift_Frob_exact_D4": round(shift_num, 7),
                "inclusive_plateau_shift_Frob_thm3": round(p4p2 * G_ideal_off_norm * thm3_inc, 7),
                "inclusive_plateau_shift_rel_pct_exact_D4": round(100.0 * shift_num / G_ideal_off_norm, 4),
                "inclusive_plateau_shift_rel_pct_thm3": round(100.0 * p4p2 * thm3_inc, 4),
                "exclusive_vetoed_plateau_shift_Frob": round(excl_shift, 7),
                "exclusive_vetoed_plateau_shift_Frob_thm3": round(p4p2 * G_ideal_off_norm * thm3_exc, 7),
                "exclusive_vetoed_plateau_shift_rel_pct": round(100.0 * excl_shift / G_ideal_off_norm, 4),
                "exclusive_vetoed_plateau_shift_rel_pct_thm3": round(100.0 * p4p2 * thm3_exc, 4),
                "HOM_NOONp_flux_uncal": crossing_stats(dh_u, dnp_u, z_grid, tol=1e-6),
                "HOM_NOONp_flux_cal": crossing_stats(dh_c, dnp_c, z_grid, tol=1e-6),
                "HOM_NOONm_flux_uncal": crossing_stats(dh_u, dnm_u, z_grid, tol=1e-6),
                "HOM_NOONm_flux_cal": crossing_stats(dh_c, dnm_c, z_grid, tol=1e-6),
                "HOM_NOONp_self_norm_from_Jz0.05": crossing_stats(dh_s, dnp_s, z_grid[2:], tol=1e-6),
                "HOM_NOONm_self_norm_from_Jz0.05": crossing_stats(dh_s, dnm_s, z_grid[2:], tol=1e-6),
                "B2_flux_uncal": crossing_stats(dp1_u, dp2_u, z_grid, tol=1e-6),
                "B2_flux_cal": crossing_stats(dp1_c, dp2_c, z_grid, tol=1e-6),
                "B2_self_norm": crossing_stats(dp1_s, dp2_s, z_grid, tol=1e-6),
            })
            if abs(eta_t - 0.25) < 1e-12 and p4p2 in (0.0, 0.02, 0.06, 0.10):
                curves_eta025[f"p4_p2_{p4p2:.2f}"] = {
                    "Jz_grid": [round(float(x), 4) for x in z_grid[::stride_curve]],
                    "Jz_grid_self_B1": [round(float(x), 4) for x in z_grid[2::stride_curve]],
                    "HOM_flux_uncal": [round(float(x), 5) for x in dh_u[::stride_curve]],
                    "NOONp_flux_uncal": [round(float(x), 5) for x in dnp_u[::stride_curve]],
                    "HOM_flux_cal": [round(float(x), 5) for x in dh_c[::stride_curve]],
                    "NOONp_flux_cal": [round(float(x), 5) for x in dnp_c[::stride_curve]],
                    "HOM_self_norm": [round(float(x), 5) for x in dh_s[::stride_curve]],
                    "NOONp_self_norm": [round(float(x), 5) for x in dnp_s[::stride_curve]],
                    "B2_Psi1_flux_cal": [round(float(x), 5) for x in dp1_c[::stride_curve]],
                    "B2_Psi2_flux_cal": [round(float(x), 5) for x in dp2_c[::stride_curve]],
                    "B2_Psi1_self_norm": [round(float(x), 5) for x in dp1_s[::stride_curve]],
                    "B2_Psi2_self_norm": [round(float(x), 5) for x in dp2_s[::stride_curve]],
                }

        spdc_by_eta[f"eta_{eta_t:.2f}"] = {
            "eta_tot": eta_t,
            "theorem3_analytic_rel_factor": round(thm3_inc, 6),
            "theorem3_analytic_frob_slope": round(G_ideal_off_norm * thm3_inc, 6),
            "theorem3_exclusive_4class_rel_factor": round(thm3_exc, 6),
            "theorem3_exclusive_4class_frob_slope": round(G_ideal_off_norm * thm3_exc, 6),
            "p4_scan": rows,
        }

    return {
        "N": N,
        "D2": D2,
        "D4": D4,
        "gamma_over_J": gamma_0,
        "G_ideal_offdiag_Frob_norm": round(G_ideal_off_norm, 7),
        "single_click_saturation_vs_eta_tot": sat_scan,
        "spdc_plateau_and_crossings_by_eta_tot": spdc_by_eta,
        "calibrated_vs_selfnorm_crossing_curves_eta0.25": curves_eta025,
    }


def main():
    t0 = time.perf_counter()
    p1_losses, p2_vhom = run_losses_and_vhom()
    p3_spdc = run_threshold_and_spdc()
    elapsed = time.perf_counter() - t0

    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, "imperfections.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {
                "seed": None,
                "deterministic": True,
                "runtime_seconds": round(elapsed, 2),
                "N_dimer": 2,
                "N_network": 8,
                "D4_SPDC": 330,
                "ports_0based": {"B1": [4, 5], "B2": [7, 0, 6], "B2_CZ4": [2, 3, 4, 5]},
                "ports_1based": {"B1": [5, 6], "B2": [8, 1, 7], "B2_CZ4": [3, 4, 5, 6]},
            },
            "part1_nonuniform_losses_krein_rutman": p1_losses,
            "part2_partial_indistinguishability_VHOM_scan": p2_vhom,
            "part3_threshold_detectors_and_SPDC_D4": p3_spdc,
        }, f, indent=2)
    print(f"imperfections.json ({os.path.getsize(out_path) / 1024:.1f} KB, {elapsed:.2f} s)")


if __name__ == "__main__":
    main()
