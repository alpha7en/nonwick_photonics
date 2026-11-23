"""Exact dimer spectrum, network size threshold N_crit = 5, spectral gap ratios,
modular bottlenecks, Anderson detuning scan, and six-metric audit (Sec. III, Figs. 4-5).

Writes data/core_networks.json.
Usage: python -I simulations/core_networks.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq
from scipy.sparse import csr_matrix, diags as spdiags, eye as speye, kron as spkron

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import (  # noqa: E402
    DATA_DIR,
    crossing_stats,
    gamma_ss,
    graph_chain,
    graph_modular_bridge,
    graph_small_world,
    hbt_observables,
    rk4,
    state_b1,
    state_b2,
    two_boson_model,
    wf_to_fock,
)


def compute_dimer():
    """Exact mirror-even cumulant block (u, c_2, v) and cubic Q_3(lambda) of the N = 2 dimer (Sec. III.A)."""
    # Discriminant of Q_3(lam) = lam^3 + 5 g lam^2 + (4 g^2 + 16) lam + 48 g vanishes at LEP_2.
    g_LEP2 = float(brentq(lambda g: 9.0 * g**6 - 412.0 * g**4 + 64.0 * g**2 - 1024.0, 6.0, 7.5, xtol=1e-14, rtol=1e-14))
    g_EP1 = 4.0

    def dimer_u(g_val: float, z_arr: np.ndarray):
        A = np.array([
            [0.0, 0.0, -4.0 * np.sqrt(2.0)],
            [0.0, -4.0 * g_val, 2.0 * np.sqrt(2.0)],
            [3.0 / np.sqrt(2.0), -np.sqrt(2.0), -g_val],
        ], dtype=np.float64)
        P = expm(A * float(z_arr[1] - z_arr[0]))
        vh = np.array([2.0 / 3.0, 0.0, 0.0])
        vnm = np.array([-1.0 / 3.0, -0.5, 0.0])
        vnp = np.array([-1.0 / 3.0, 0.5, 0.0])
        u_h = np.empty(len(z_arr), dtype=np.float64)
        u_nm = np.empty(len(z_arr), dtype=np.float64)
        u_np = np.empty(len(z_arr), dtype=np.float64)
        for k in range(len(z_arr)):
            u_h[k], u_nm[k], u_np[k] = vh[0], vnm[0], vnp[0]
            vh, vnm, vnp = P @ vh, P @ vnm, P @ vnp
        return u_h, u_nm, u_np

    def crossing_at(g_val: float, state: str = "NOONm", z_max: float = 15.0, n_pts: int = 7501):
        z_fine = np.linspace(0.0, z_max, n_pts)
        u_h, u_nm, u_np = dimer_u(g_val, z_fine)
        u_ref = u_nm if state == "NOONm" else u_np
        # On the dimer, d_HBT(z) = 2 |u(z)|.
        diff = 2.0 * (np.abs(u_h) - np.abs(u_ref))
        min_diff = float(np.min(diff))
        idx_min = int(np.argmin(diff))
        cross_idx = np.where(diff[:-1] * diff[1:] < 0.0)[0]
        if len(cross_idx) == 0 or min_diff >= -1e-12:
            return False, None, 0.0, None
        k0 = int(cross_idx[0])
        z_star = z_fine[k0] - diff[k0] * (z_fine[k0 + 1] - z_fine[k0]) / (diff[k0 + 1] - diff[k0])
        return True, float(z_star), float(-min_diff), float(z_fine[idx_min])

    def max_gap(g_val: float, state: str) -> float:
        return crossing_at(g_val, state, z_max=15.0, n_pts=6001)[2]

    g_vis1e7_nm = float(brentq(lambda g: max_gap(g, "NOONm") - 1e-7, 5.5, 6.70, xtol=1e-8))
    g_vis1e7_np = float(brentq(lambda g: max_gap(g, "NOONp") - 1e-7, 5.5, 6.70, xtol=1e-8))
    g_vis1e4_nm = float(brentq(lambda g: max_gap(g, "NOONm") - 1e-4, 5.5, 6.60, xtol=1e-8))

    ok_at_lep2 = crossing_at(g_LEP2, "NOONm", z_max=30.0, n_pts=6001)[0]
    ok_above_lep2 = crossing_at(g_LEP2 + 1e-3, "NOONm", z_max=30.0, n_pts=6001)[0]
    assert (not ok_at_lep2) and (not ok_above_lep2)

    g_spec = np.unique(np.concatenate([
        np.linspace(0.0, 10.0, 401),
        np.linspace(6.5, 7.0, 51),
        np.array([g_EP1, g_vis1e7_np, g_vis1e7_nm, g_LEP2]),
    ]))
    q3_re, q3_im, q1_re, q1_im = [], [], [], []
    for gv in g_spec:
        r3 = np.roots([1.0, 5.0 * gv, 4.0 * gv**2 + 16.0, 48.0 * gv])
        r3 = r3[np.lexsort((-r3.imag, -r3.real))]
        q3_re.append([round(float(z.real), 7) for z in r3])
        q3_im.append([round(float(z.imag), 7) for z in r3])

        # Single-particle quadratic: lam^2 + g lam + 4 = 0.
        r1 = np.roots([1.0, gv, 4.0])
        r1 = r1[np.lexsort((-r1.imag, -r1.real))]
        q1_re.append([round(float(z.real), 7) for z in r1])
        q1_im.append([round(float(z.imag), 7) for z in r1])

    g_cross = np.unique(np.concatenate([
        np.linspace(0.0, 6.0, 61),
        np.linspace(6.0, 6.60, 31),
        np.array([0.45, 1.5, 4.0, 6.605, g_vis1e7_nm, 6.65, 6.70, g_LEP2, 7.0]),
    ]))
    cross_nm, cross_np = [], []
    for gv in g_cross:
        for state, target in (("NOONm", cross_nm), ("NOONp", cross_np)):
            ok, zs, gap, zmax = crossing_at(float(gv), state)
            target.append({
                "gamma_over_J": round(float(gv), 6),
                "crosses_above_1e-12": ok,
                "crosses_above_1e-7": bool(gap >= 1e-7),
                "Jz_star": round(zs, 6) if zs is not None else None,
                "max_inversion_gap": float(f"{gap:.6e}") if 0 < gap < 1e-5 else round(gap, 6),
                "Jz_at_max_gap": round(zmax, 5) if zmax is not None else None,
            })

    z_curve = np.linspace(0.0, 6.0, 301)
    curves = {}
    for gv in [0.0, 0.45, 1.5, 4.0, round(g_LEP2, 6), 8.0]:
        uh, unm, unp = dimer_u(gv, z_curve)
        key = f"gamma_{gv:.6f}" if 6.0 < gv < 7.0 else f"gamma_{gv:.2f}"
        curves[key] = {
            "gamma_over_J": gv,
            "u_HOM": [round(float(x), 6) for x in uh],
            "u_NOONm": [round(float(x), 6) for x in unm],
            "u_NOONp": [round(float(x), 6) for x in unp],
            "d_HBT_HOM": [round(float(2.0 * abs(x)), 6) for x in uh],
            "d_HBT_NOONm": [round(float(2.0 * abs(x)), 6) for x in unm],
            "d_HBT_NOONp": [round(float(2.0 * abs(x)), 6) for x in unp],
        }

    return {
        "constants": {
            "single_particle_EP1_gamma_over_J": g_EP1,
            "cumulant_LEP2_gamma_over_J": round(g_LEP2, 8),
            "visibility_1e-7_cutoff_HOM_vs_NOONm": round(g_vis1e7_nm, 6),
            "visibility_1e-7_cutoff_HOM_vs_NOONp": round(g_vis1e7_np, 6),
            "visibility_1e-4_cutoff_HOM_vs_NOONm": round(g_vis1e4_nm, 6),
            "zero_crossing_verified_at_LEP2": not ok_at_lep2,
            "Jz_star_at_gamma_0_exact": round(float(np.arccos(1.0 / 3.0) / 4.0), 6),
            "Q3_polynomial": "lambda^3 + 5*gamma*lambda^2 + (4*gamma^2 + 16*J^2)*lambda + 48*gamma*J^2",
            "discriminant_formula": "16*(9*g^6 - 412*g^4 + 64*g^2 - 1024)",
        },
        "spectrum_scan": {
            "gamma_over_J": [round(float(g), 6) for g in g_spec],
            "Q3_real": q3_re,
            "Q3_imag": q3_im,
            "one_body_real": q1_re,
            "one_body_imag": q1_im,
        },
        "crossing_vs_gamma": {"HOM_vs_NOONm": cross_nm, "HOM_vs_NOONp": cross_np},
        "trajectories": {"Jz_grid": [round(float(z), 5) for z in z_curve], "by_gamma": curves},
    }


def compute_n_scaling_and_topologies():
    """Network size threshold N_crit = 5 and dynamics on chains, K_3, small-world, and modular bridge (Sec. III.B-C)."""
    J, dz, n_steps = 1.0, 0.025, 800

    # Initial squared-distance difference d_{HBT,1}^2(0) - d_{HBT,2}^2(0) = (N(N+1) - 24) / (4 N(N+1)) for B2.
    initial_gap_scaling = []
    max_analytic_err = 0.0
    for N in range(2, 21):
        G_ss = gamma_ss(N)
        off_mask = ~np.eye(N, dtype=bool)
        W_hom, W_nm = state_b1(N, 0, 1, noon_sign=-1.0)
        G_h0, G_n0 = 2.0 * np.abs(W_hom) ** 2, 2.0 * np.abs(W_nm) ** 2
        entry = {
            "N": N,
            "D2": N * (N + 1) // 2,
            "B1_HOM_d0_full": round(float(np.linalg.norm(G_h0 - G_ss)), 6),
            "B1_NOON_d0_full": round(float(np.linalg.norm(G_n0 - G_ss)), 6),
            "B1_HOM_d0_offdiag": round(float(np.linalg.norm((G_h0 - G_ss)[off_mask])), 6),
            "B1_NOON_d0_offdiag": round(float(np.linalg.norm((G_n0 - G_ss)[off_mask])), 6),
        }
        if N >= 3:
            W1, W2 = state_b2(N, 0, 1, 2, sign=1.0)
            d1_0 = float(np.linalg.norm(2.0 * np.abs(W1) ** 2 - G_ss))
            d2_0 = float(np.linalg.norm(2.0 * np.abs(W2) ** 2 - G_ss))
            diff_num = d1_0**2 - d2_0**2
            diff_ana = (N * (N + 1) - 24.0) / (4.0 * N * (N + 1))
            max_analytic_err = max(max_analytic_err, abs(diff_num - diff_ana))
            entry.update({
                "B2_d1_0": round(d1_0, 6),
                "B2_d2_0": round(d2_0, 6),
                "B2_d1_sq_minus_d2_sq_analytic": round(diff_ana, 8),
                "B2_d1_sq_minus_d2_sq_numeric": round(diff_num, 8),
                "B2_bunched_starts_farther": bool(diff_ana > 0.0),
            })
        initial_gap_scaling.append(entry)

    def run_graph(adj: np.ndarray, gamma_0: float, ports_2, ports_3, n_st: int = n_steps, store_curves: bool = False):
        N = adj.shape[0]
        zg = np.linspace(0.0, dz * n_st, n_st + 1)
        _, _, _, H2, _, M2, n_ops, pairs, _, _ = two_boson_model(adj, J, np.full(N, gamma_0))
        G_ss = gamma_ss(N)
        off_mask = ~np.eye(N, dtype=bool)

        W_hom, W_nm = state_b1(N, ports_2[0], ports_2[1], noon_sign=-1.0)
        ph, pn = wf_to_fock(W_hom, pairs), wf_to_fock(W_nm, pairs)
        _, Gh = hbt_observables(rk4(H2, M2, np.outer(ph, ph.conj()), dz, n_st), n_ops, pairs, N)
        _, Gn = hbt_observables(rk4(H2, M2, np.outer(pn, pn.conj()), dz, n_st), n_ops, pairs, N)
        dh_full = np.linalg.norm(Gh - G_ss[None], axis=(1, 2))
        dn_full = np.linalg.norm(Gn - G_ss[None], axis=(1, 2))
        dh_off = np.linalg.norm((Gh - G_ss[None])[:, off_mask], axis=1)
        dn_off = np.linalg.norm((Gn - G_ss[None])[:, off_mask], axis=1)

        out = {
            "N": N,
            "gamma_over_J": gamma_0,
            "ports_2_0based": list(ports_2),
            "ports_2_1based": [p + 1 for p in ports_2],
            "B1_HOM_vs_NOONm_full_HBT": crossing_stats(dh_full, dn_full, zg),
            "B1_HOM_vs_NOONm_offdiag": crossing_stats(dh_off, dn_off, zg),
        }
        if ports_3 is not None:
            W1, W2 = state_b2(N, ports_3[0], ports_3[1], ports_3[2], sign=1.0)
            p1, p2 = wf_to_fock(W1, pairs), wf_to_fock(W2, pairs)
            I1, G1 = hbt_observables(rk4(H2, M2, np.outer(p1, p1.conj()), dz, n_st), n_ops, pairs, N)
            I2, G2 = hbt_observables(rk4(H2, M2, np.outer(p2, p2.conj()), dz, n_st), n_ops, pairs, N)
            d1 = np.linalg.norm(G1 - G_ss[None], axis=(1, 2))
            d2 = np.linalg.norm(G2 - G_ss[None], axis=(1, 2))
            d1_neq = np.linalg.norm((G1 - G_ss[None])[:, off_mask], axis=1)
            d2_neq = np.linalg.norm((G2 - G_ss[None])[:, off_mask], axis=1)

            def ctilde_norm(G_t, I_t):
                # Linear-orthogonal cumulant projection onto V_cum (Theorem 1).
                dI = I_t - (2.0 / N)
                G_1b = (dI[:, :, None] + dI[:, None, :]) / float(N + 2)
                idx = np.arange(N)
                G_1b[:, idx, idx] = 4.0 * dI / float(N + 2)
                return np.linalg.norm(G_t - G_ss[None] - G_1b, axis=(1, 2))

            d1_ct, d2_ct = ctilde_norm(G1, I1), ctilde_norm(G2, I2)
            i10, i20 = min(400, n_st), min(800, n_st)
            out.update({
                "ports_3_0based": list(ports_3),
                "ports_3_1based": [p + 1 for p in ports_3],
                "max_intensity_diff": float(np.max(np.abs(I1 - I2))),
                "B2_3port_full_HBT": crossing_stats(d1, d2, zg),
                "B2_3port_offdiag": crossing_stats(d1_neq, d2_neq, zg),
                "B2_3port_Ctilde": crossing_stats(d1_ct, d2_ct, zg),
                "d1_minus_d2_at_Jz10": round(float(d1[i10] - d2[i10]), 6),
                "d1_minus_d2_at_Jz20": round(float(d1[i20] - d2[i20]), 6),
                "d_neq_1_minus_2_at_Jz10": round(float(d1_neq[i10] - d2_neq[i10]), 6),
                "d_neq_1_minus_2_at_Jz20": round(float(d1_neq[i20] - d2_neq[i20]), 6),
                "d_Ctilde_1_minus_2_at_Jz10": round(float(d1_ct[i10] - d2_ct[i10]), 6),
                "d_Ctilde_1_minus_2_at_Jz20": round(float(d1_ct[i20] - d2_ct[i20]), 6),
            })
            if store_curves:
                idx_full = np.arange(0, n_st + 1, 4)
                out["curves_early_Jz0_to_3"] = {
                    "Jz": [round(float(x), 4) for x in zg[:121]],
                    "d1": [round(float(x), 6) for x in d1[:121]],
                    "d2": [round(float(x), 6) for x in d2[:121]],
                    "d_neq_1": [round(float(x), 6) for x in d1_neq[:121]],
                    "d_neq_2": [round(float(x), 6) for x in d2_neq[:121]],
                    "d_Ctilde_1": [round(float(x), 6) for x in d1_ct[:121]],
                    "d_Ctilde_2": [round(float(x), 6) for x in d2_ct[:121]],
                }
                out["curves_full"] = {
                    "Jz": [round(float(x), 4) for x in zg[idx_full]],
                    "d1": [round(float(x), 6) for x in d1[idx_full]],
                    "d2": [round(float(x), 6) for x in d2[idx_full]],
                    "d1_minus_d2": [round(float(x), 6) for x in (d1 - d2)[idx_full]],
                    "d_neq_1": [round(float(x), 6) for x in d1_neq[idx_full]],
                    "d_neq_2": [round(float(x), 6) for x in d2_neq[idx_full]],
                    "d_neq_1_minus_2": [round(float(x), 6) for x in (d1_neq - d2_neq)[idx_full]],
                    "d_Ctilde_1": [round(float(x), 6) for x in d1_ct[idx_full]],
                    "d_Ctilde_2": [round(float(x), 6) for x in d2_ct[idx_full]],
                    "d_Ctilde_1_minus_2": [round(float(x), 6) for x in (d1_ct - d2_ct)[idx_full]],
                }
        return out

    chain_045, chain_250 = {}, {}
    for N_sz in [2, 3, 4, 5, 6, 8, 10, 12, 16, 20]:
        adj_c = graph_chain(N_sz)
        mid = N_sz // 2
        p2 = (mid - 1, mid)
        p3 = (mid, mid - 1, mid + 1) if N_sz >= 3 else None
        keep = N_sz in (3, 4, 5, 8, 16)
        chain_045[f"N_{N_sz}"] = run_graph(adj_c, 0.45, p2, p3, n_steps, store_curves=keep)
        chain_250[f"N_{N_sz}"] = run_graph(adj_c, 2.50, p2, p3, n_steps, store_curves=keep)

    adj_k3 = np.ones((3, 3)) - np.eye(3)
    triangle_trimer = {
        "gamma_0.45": run_graph(adj_k3, 0.45, (0, 1), (0, 1, 2), n_steps, store_curves=True),
        "gamma_2.50": run_graph(adj_k3, 2.50, (0, 1), (0, 1, 2), n_steps, store_curves=True),
    }

    sw_045, sw_250 = {}, {}
    for N_sz in [8, 12, 16, 20]:
        adj_sw = graph_small_world(N_sz)
        p2, p3 = ((4, 5), (7, 0, 6)) if N_sz == 8 else ((0, 1), (0, 1, N_sz // 2 - 1))
        sw_045[f"N_{N_sz}"] = run_graph(adj_sw, 0.45, p2, p3, n_steps, store_curves=(N_sz in (8, 16)))
        sw_250[f"N_{N_sz}"] = run_graph(adj_sw, 2.50, p2, p3, n_steps, store_curves=(N_sz == 8))

    adj_mod16 = graph_modular_bridge(16, J_bridge=0.15)
    modular_bridge = {
        "J_bridge_over_J": 0.15,
        "N": 16,
        "intramodule_gamma0.45": run_graph(adj_mod16, 0.45, (0, 1), (1, 0, 2), 1000, store_curves=True),
        "bridge_straddling_gamma0.45": run_graph(adj_mod16, 0.45, (7, 8), (6, 7, 8), 1000, store_curves=True),
        "bridge_straddling_gamma1.50": run_graph(adj_mod16, 1.50, (7, 8), (6, 7, 8), 1000, store_curves=True),
    }

    return {
        "N_crit_analytic_value": 5,
        "N_crit_formula": "(N*(N+1) - 24) / (4*N*(N+1))",
        "max_analytic_vs_numeric_gap_error": max_analytic_err,
        "initial_gap_N2_to_N20": initial_gap_scaling,
        "triangle_trimer_K3": triangle_trimer,
        "chains_1D_gamma0.45": chain_045,
        "chains_1D_gamma2.50": chain_250,
        "small_world_WS_gamma0.45": sw_045,
        "small_world_WS_gamma2.50": sw_250,
        "modular_bridge_N16": modular_bridge,
    }


def compute_spectral_gaps():
    """Exact single-particle gap Delta_1 and pure-cumulant gap Delta_cum in V_cum = ker Pi_1 (Sec. III.C)."""
    J = 1.0

    def gaps_for(adj: np.ndarray, gamma_0: float):
        N = adj.shape[0]
        H1, _, _, H2, _, M2, _, _, _, B_ann = two_boson_model(adj, J, np.full(N, gamma_0))
        D2 = H2.shape[0]

        I1 = np.eye(N, dtype=np.complex128)
        L1 = -1j * (np.kron(H1, I1) - np.kron(I1, H1.T))
        m1 = np.full((N, N), -gamma_0)
        np.fill_diagonal(m1, 0.0)
        L1 += np.diag(m1.ravel())
        ev1 = np.linalg.eigvals(L1)
        delta_1 = float(-ev1[np.argsort(ev1.real)[::-1]][1].real)

        # Orthonormal bases of V_{1b} = span{b_i^dag b_j} and V_cum = V_{1b}^perp.
        V1_cols = np.column_stack([(B_ann[i].conj().T @ B_ann[j]).ravel() for i in range(N) for j in range(N)])
        Q_full, _ = np.linalg.qr(V1_cols, mode="complete")
        Q_1b, Q_ker = Q_full[:, : N * N], Q_full[:, N * N :]

        H2_sp = csr_matrix(H2)
        I2_sp = speye(D2, format="csr", dtype=np.complex128)
        L2_sp = (
            -1j * (spkron(H2_sp, I2_sp, format="csr") - spkron(I2_sp, H2_sp.T, format="csr"))
            - spdiags(M2.ravel(), offsets=0, shape=(D2 * D2, D2 * D2), format="csr")
        )
        ev_cum = np.linalg.eigvals(Q_ker.conj().T @ (L2_sp @ Q_ker))
        delta_cum = float(-np.max(ev_cum.real))
        inv_err = float(np.linalg.norm(Q_1b.conj().T @ (L2_sp @ Q_ker), "fro"))
        return {
            "N": N,
            "D2": D2,
            "dim_V1b": N * N,
            "dim_Vcum": D2 * D2 - N * N,
            "gamma_over_J": gamma_0,
            "Delta_1": round(delta_1, 6),
            "Delta_cum": round(delta_cum, 6),
            "ratio_Delta_cum_over_Delta_1": round(delta_cum / delta_1, 4),
            "invariance_cross_block_Frob": inv_err,
        }

    cases = [
        ("dimer_N2_gamma0.45", graph_chain(2), 0.45),
        ("dimer_N2_gamma2.50", graph_chain(2), 2.50),
        ("triangle_K3_gamma0.45", np.ones((3, 3)) - np.eye(3), 0.45),
        ("triangle_K3_gamma2.50", np.ones((3, 3)) - np.eye(3), 2.50),
        ("chain_N5_gamma0.45", graph_chain(5), 0.45),
        ("chain_N6_gamma0.45", graph_chain(6), 0.45),
        ("chain_N8_gamma0.45", graph_chain(8), 0.45),
        ("chain_N8_gamma2.50", graph_chain(8), 2.50),
        ("small_world_N8_gamma0.45", graph_small_world(8), 0.45),
        ("small_world_N8_gamma2.50", graph_small_world(8), 2.50),
        ("modular_N8_Jbr0.15_gamma0.20", graph_modular_bridge(8, 0.15), 0.20),
        ("modular_N8_Jbr0.15_gamma0.45", graph_modular_bridge(8, 0.15), 0.45),
        ("modular_N8_Jbr0.15_gamma1.00", graph_modular_bridge(8, 0.15), 1.00),
        ("modular_N8_Jbr0.15_gamma1.50", graph_modular_bridge(8, 0.15), 1.50),
        ("modular_N8_Jbr0.15_gamma2.50", graph_modular_bridge(8, 0.15), 2.50),
    ]
    return {label: gaps_for(adj, g0) for label, adj, g0 in cases}


def compute_anderson_scan(rng: np.random.Generator):
    """Static on-site detuning disorder eps_j ~ U[-W, W] on an N = 16 open chain (Sec. III.C, Fig. 5d)."""
    N, J, gamma_0, dz, n_steps = 16, 1.0, 0.45, 0.025, 600
    z_grid = np.linspace(0.0, dz * n_steps, n_steps + 1)
    adj = graph_chain(N)
    G_ss = gamma_ss(N)
    W1, W2 = state_b2(N, 8, 7, 9, sign=1.0)  # 0-based (8, 7, 9) <-> ports (9, 8, 10)

    scan = []
    tail_W0 = None
    for W_dis in [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        n_real = 1 if W_dis == 0.0 else 20
        gaps, jz_first, rel_pcts, n_cross, tails = [], [], [], [], []
        max_g1_diff = 0.0
        for _ in range(n_real):
            eps = np.zeros(N) if W_dis == 0.0 else rng.uniform(-W_dis, W_dis, size=N)
            _, _, _, H2, _, M2, n_ops, pairs, _, _ = two_boson_model(adj, J, np.full(N, gamma_0), eps=eps)
            p1, p2 = wf_to_fock(W1, pairs), wf_to_fock(W2, pairs)
            I1, G1 = hbt_observables(rk4(H2, M2, np.outer(p1, p1.conj()), dz, n_steps), n_ops, pairs, N)
            I2, G2 = hbt_observables(rk4(H2, M2, np.outer(p2, p2.conj()), dz, n_steps), n_ops, pairs, N)
            max_g1_diff = max(max_g1_diff, float(np.max(np.abs(I1 - I2))))
            d1 = np.linalg.norm(G1 - G_ss[None], axis=(1, 2))
            d2 = np.linalg.norm(G2 - G_ss[None], axis=(1, 2))
            c = crossing_stats(d1, d2, z_grid)
            if c is not None:
                gaps.append(c["max_inversion_gap"])
                jz_first.append(c["first_crossing"])
                rel_pcts.append(c["rel_inversion_pct"])
                n_cross.append(c["n_crossings"])
            tails.append(float(0.5 * (d1[-1] + d2[-1])))

        mean_tail = float(np.mean(tails))
        if W_dis == 0.0:
            tail_W0 = mean_tail
        scan.append({
            "W_over_J": W_dis,
            "n_realizations": n_real,
            "crossing_rate": round(len(gaps) / n_real, 4),
            "max_intensity_diff_across_realizations": max_g1_diff,
            "mean_first_crossing_Jz": round(float(np.mean(jz_first)), 5) if jz_first else None,
            "std_first_crossing_Jz": round(float(np.std(jz_first, ddof=1)), 5) if len(jz_first) > 1 else 0.0,
            "mean_max_inversion_gap": round(float(np.mean(gaps)), 5) if gaps else None,
            "std_max_inversion_gap": round(float(np.std(gaps, ddof=1)), 5) if len(gaps) > 1 else 0.0,
            "mean_rel_inversion_pct": round(float(np.mean(rel_pcts)), 2) if rel_pcts else None,
            "mean_n_crossings_up_to_Jz15": round(float(np.mean(n_cross)), 2) if n_cross else None,
            "mean_residual_distance_at_Jz15": round(mean_tail, 6),
            "std_residual_distance_at_Jz15": round(float(np.std(tails, ddof=1)), 6) if len(tails) > 1 else 0.0,
            "residual_slowdown_factor_vs_W0": round(mean_tail / max(tail_W0, 1e-12), 2),
        })

    return {
        "N": N,
        "topology": "open_1D_chain",
        "ports_B2_0based": [8, 7, 9],
        "ports_B2_1based": [9, 8, 10],
        "gamma_over_J": gamma_0,
        "Jz_max": 15.0,
        "scan_by_W": scan,
    }


def compute_metric_audit():
    """Comparison of six distance metrics (d_HBT, d_neq, d_Ctilde, d_cum, d_HS, d_tr) (Sec. II.B, III.B)."""
    J, dz, n_steps = 1.0, 0.025, 400
    z_grid = np.linspace(0.0, dz * n_steps, n_steps + 1)

    def eval_metrics(adj: np.ndarray, gamma_0: float, W1: np.ndarray, W2: np.ndarray):
        N = adj.shape[0]
        _, _, _, H2, _, M2, n_ops, pairs, _, _ = two_boson_model(adj, J, np.full(N, gamma_0))
        D2 = len(pairs)
        G_ss = gamma_ss(N)
        C_ss = G_ss - (2.0 / N) ** 2
        rho_ss = np.eye(D2, dtype=np.complex128) / float(D2)
        off_mask = ~np.eye(N, dtype=bool)

        p1, p2 = wf_to_fock(W1, pairs), wf_to_fock(W2, pairs)
        tr1 = rk4(H2, M2, np.outer(p1, p1.conj()), dz, n_steps)
        tr2 = rk4(H2, M2, np.outer(p2, p2.conj()), dz, n_steps)
        I1, G1 = hbt_observables(tr1, n_ops, pairs, N)
        I2, G2 = hbt_observables(tr2, n_ops, pairs, N)

        def ctilde(G_t, I_t):
            dI = I_t - (2.0 / N)
            G_1b = (dI[:, :, None] + dI[:, None, :]) / float(N + 2)
            idx = np.arange(N)
            G_1b[:, idx, idx] = 4.0 * dI / float(N + 2)
            return G_t - G_ss[None] - G_1b

        C1, C2 = ctilde(G1, I1), ctilde(G2, I2)
        Cw1 = G1 - I1[:, :, None] * I1[:, None, :]
        Cw2 = G2 - I2[:, :, None] * I2[:, None, :]

        def pack(a1, a2):
            return {
                "d1_0": round(float(a1[0]), 6),
                "d2_0": round(float(a2[0]), 6),
                "d1_at_Jz0.6": round(float(a1[24]), 6),
                "d2_at_Jz0.6": round(float(a2[24]), 6),
                "d1_at_Jz5.0": round(float(a1[200]), 6),
                "d2_at_Jz5.0": round(float(a2[200]), 6),
                "crossing": crossing_stats(a1, a2, z_grid),
            }

        d_tr_1 = np.array([0.5 * np.sum(np.abs(np.linalg.eigvalsh(tr1[k] - rho_ss))) for k in range(n_steps + 1)])
        d_tr_2 = np.array([0.5 * np.sum(np.abs(np.linalg.eigvalsh(tr2[k] - rho_ss))) for k in range(n_steps + 1)])
        return {
            "d_HBT": pack(np.linalg.norm(G1 - G_ss[None], axis=(1, 2)), np.linalg.norm(G2 - G_ss[None], axis=(1, 2))),
            "d_neq": pack(np.linalg.norm((G1 - G_ss[None])[:, off_mask], axis=1), np.linalg.norm((G2 - G_ss[None])[:, off_mask], axis=1)),
            "d_Ctilde_linear_orth": pack(np.linalg.norm(C1, axis=(1, 2)), np.linalg.norm(C2, axis=(1, 2))),
            "d_cum_Wick": pack(np.linalg.norm(Cw1 - C_ss[None], axis=(1, 2)), np.linalg.norm(Cw2 - C_ss[None], axis=(1, 2))),
            "d_HS_density_matrix": pack(np.linalg.norm(tr1 - rho_ss[None], axis=(1, 2)), np.linalg.norm(tr2 - rho_ss[None], axis=(1, 2))),
            "d_trace_density_matrix": pack(d_tr_1, d_tr_2),
        }

    adj_sw, adj_ch = graph_small_world(8), graph_chain(8)
    W1_sw, W2_sw = state_b2(8, 7, 0, 6, sign=1.0)
    W1_ch, W2_ch = state_b2(8, 4, 3, 5, sign=1.0)
    Wh, Wnp = state_b1(8, 4, 5, noon_sign=1.0)
    return {
        "SW8_B2_gamma0.00_unitary_beats": eval_metrics(adj_sw, 0.0, W1_sw, W2_sw),
        "SW8_B2_gamma0.45_underdamped": eval_metrics(adj_sw, 0.45, W1_sw, W2_sw),
        "SW8_B2_gamma2.50_overdamped_aperiodic": eval_metrics(adj_sw, 2.50, W1_sw, W2_sw),
        "SW8_B1_HOM_NOONplus_gamma0.45": eval_metrics(adj_sw, 0.45, Wh, Wnp),
        "SW8_B1_HOM_NOONplus_gamma2.50": eval_metrics(adj_sw, 2.50, Wh, Wnp),
        "Chain8_B2_gamma2.50_overdamped_aperiodic": eval_metrics(adj_ch, 2.50, W1_ch, W2_ch),
    }


def main():
    t0 = time.perf_counter()
    seed = 20261119
    rng = np.random.default_rng(seed)

    dimer = compute_dimer()
    topologies = compute_n_scaling_and_topologies()
    gaps = compute_spectral_gaps()
    anderson = compute_anderson_scan(rng)
    metrics = compute_metric_audit()
    elapsed = time.perf_counter() - t0

    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, "core_networks.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {"seed": seed, "runtime_seconds": round(elapsed, 2)},
            "dimer_N2": dimer,
            "N_scaling_and_topologies": topologies,
            "spectral_gaps_Delta_cum_vs_Delta_1": gaps,
            "anderson_disorder_N16": anderson,
            "metric_audit": metrics,
        }, f, indent=2)
    print(f"core_networks.json ({os.path.getsize(out_path) / 1024:.1f} KB, {elapsed:.2f} s)")


if __name__ == "__main__":
    main()
