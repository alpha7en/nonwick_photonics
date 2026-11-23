"""Symbolic (SymPy) and numerical verification of dimer Q_3(lambda) and LEP_2 (Sec. III.A),
N_crit = 5 and six-metric identities (Sec. III.B), K_N / modular-bridge spectral gap ratios
(Sec. III.C, Appendix B.3), Theorem 3 four-photon SPDC factors (Sec. V.A), Theorem 4
Krein-Rutman Pauli-Bose filtering (Sec. V.B, Appendix B.4), and passive synthesis (Sec. II.C).

Exits 0 iff all algebraic and numerical identities hold.
Usage: python -I proofs/spectral_and_kr.py
"""
from __future__ import annotations

import itertools
import json
import numpy as np
import sympy as sp
from scipy.linalg import expm


def fock_basis(N: int, k_ph: int):
    states = []
    for comb in itertools.combinations_with_replacement(range(N), k_ph):
        occ = [0] * N
        for m in comb:
            occ[m] += 1
        states.append(tuple(occ))
    return states, {s: i for i, s in enumerate(states)}


def annihilation_sympy(N: int, k_ph: int):
    states_k, _ = fock_basis(N, k_ph)
    states_km1, idx_km1 = fock_basis(N, k_ph - 1)
    b_ops = [sp.zeros(len(states_km1), len(states_k)) for _ in range(N)]
    for col, occ in enumerate(states_k):
        for m in range(N):
            if occ[m] > 0:
                new_occ = list(occ)
                new_occ[m] -= 1
                b_ops[m][idx_km1[tuple(new_occ)], col] = sp.sqrt(occ[m])
    return states_k, b_ops


def verify_dimer_and_lep2():
    """Sec. III.A, Appendix B.1: exact 9x9 dimer factorization, LEP_2 = 6.758449, no overdamped crossing."""
    J, gamma, s, x, g = sp.symbols("J gamma s x g", real=True, positive=True)
    _, b2 = annihilation_sympy(2, 2)
    H2 = -J * (b2[0].H * b2[1] + b2[1].H * b2[0])
    M_deph = sp.Matrix([[0, gamma, 4 * gamma], [gamma, 0, gamma], [4 * gamma, gamma, 0]])

    I3 = sp.eye(3)
    L2 = -sp.I * (sp.kronecker_product(H2.subs(J, 1), I3) - sp.kronecker_product(I3, H2.subs(J, 1).T))
    L2 -= sp.diag(*[M_deph.subs(gamma, g)[a, b] for a in range(3) for b in range(3)])
    Q3_x = x**3 + 5 * g * x**2 + (4 * g**2 + 16) * x + 48 * g
    assert sp.expand((x * sp.eye(9) - L2).det() - x * (x + g) * (x**2 + g * x + 4) * (x**2 + 5 * g * x + 4 * g**2 + 4) * Q3_x) == 0

    u, c2, v = sp.symbols("u c2 v", real=True)
    rho_even = sp.Matrix([[sp.Rational(1, 3) - u / 2, sp.I * v, c2], [-sp.I * v, sp.Rational(1, 3) + u, -sp.I * v], [c2, sp.I * v, sp.Rational(1, 3) - u / 2]])
    drho = -sp.I * (H2 * rho_even - rho_even * H2) - sp.matrix_multiply_elementwise(M_deph, rho_even)
    du, dc2, dv = sp.expand(drho[1, 1]), sp.expand(drho[0, 2]), sp.expand(drho[0, 1] / sp.I)
    A_dimer = sp.Matrix([[du.coeff(q) for q in (u, c2, v)], [dc2.coeff(q) for q in (u, c2, v)], [dv.coeff(q) for q in (u, c2, v)]])
    assert sp.expand((x * sp.eye(3) - A_dimer.subs({J: 1, gamma: g})).det() - Q3_x) == 0

    # Laplace transforms and unitality sum rule u_HOM + u_NOON+ + u_NOON- == 0.
    Q3_s = s**3 + 5 * gamma * s**2 + (4 * gamma**2 + 16 * J**2) * s + 48 * gamma * J**2
    res_mat = (s * sp.eye(3) - A_dimer).inv()
    U_HOM = sp.together(res_mat.row(0).dot(sp.Matrix([sp.Rational(2, 3), 0, 0])))
    U_NOONp = sp.together(res_mat.row(0).dot(sp.Matrix([-sp.Rational(1, 3), sp.Rational(1, 2), 0])))
    U_NOONm = sp.together(res_mat.row(0).dot(sp.Matrix([-sp.Rational(1, 3), -sp.Rational(1, 2), 0])))
    assert sp.simplify(U_HOM * Q3_s - sp.Rational(2, 3) * ((s + 4 * gamma) * (s + gamma) + 4 * J**2)) == 0
    assert sp.simplify(U_NOONm * Q3_s + sp.Rational(1, 3) * ((s + 4 * gamma) * (s + gamma) + 16 * J**2)) == 0
    assert sp.simplify(U_NOONp * Q3_s + sp.Rational(1, 3) * ((s + 4 * gamma) * (s + gamma) - 8 * J**2)) == 0
    assert sp.simplify(U_HOM + U_NOONp + U_NOONm) == 0

    # Discriminant, unique positive root LEP_2, and 2x2 Jordan block.
    disc_3 = sp.factor(sp.discriminant(sp.Poly(Q3_x, x), x))
    assert sp.expand(disc_3 - 16 * (9 * g**6 - 412 * g**4 + 64 * g**2 - 1024)) == 0
    pos_roots = [r for r in sp.real_roots(sp.Poly(disc_3, g)) if r > 0]
    assert len(pos_roots) == 1
    g_EP = float(pos_roots[0].evalf(25))
    assert abs(g_EP - 6.758448655569) < 1e-10

    A_ep = np.array([[0.0, 0.0, -4.0 * np.sqrt(2.0)], [0.0, -4.0 * g_EP, 2.0 * np.sqrt(2.0)], [3.0 / np.sqrt(2.0), -np.sqrt(2.0), -g_EP]])
    roots_ep = np.sort(np.roots([1.0, 5.0 * g_EP, 4.0 * g_EP**2 + 16.0, 48.0 * g_EP]).real)
    sv_ep = np.linalg.svd(A_ep - 0.5 * (roots_ep[1] + roots_ep[2]) * np.eye(3), compute_uv=False)
    assert sv_ep[2] < 1e-6 and sv_ep[1] > 1.0

    # Short-time Taylor expansions and zero-dephasing crossing Jz*(0) = (1/4) arccos(1/3).
    w, z = sp.symbols("w z", real=True)
    for expr, exp_tay in (
        (U_HOM, sp.Rational(2, 3) - 4 * J**2 * z**2 + sp.Rational(4, 3) * gamma * J**2 * z**3),
        (U_NOONp, -sp.Rational(1, 3) + 4 * J**2 * z**2 - 4 * gamma * J**2 * z**3),
        (U_NOONm, -sp.Rational(1, 3) + sp.Rational(8, 3) * gamma * J**2 * z**3),
    ):
        ser = sp.series(expr.subs(s, 1 / w), w, 0, 5).removeO()
        tay = sum(sp.simplify(ser.coeff(w, k + 1)) * z**k / sp.factorial(k) for k in range(4))
        assert sp.simplify(tay - exp_tay) == 0

    Jz_star_0 = float(sp.acos(sp.Rational(1, 3)) / 4)
    assert abs((1 / 6 + 0.5 * np.cos(4 * Jz_star_0)) - 1 / 3) < 1e-15

    # Residue signs for all g > g_EP proving zero overdamped crossings.
    assert sp.simplify(Q3_x.subs(x, -4 * g) + 16 * g) == 0
    assert sp.simplify(Q3_x.subs(x, -3 * g) - 6 * g**3) == 0
    assert sp.simplify(Q3_x.subs(x, -g) - 32 * g) == 0
    for gv in (g_EP, g_EP + 1e-4, 7.0, 10.0, 25.0, 100.0):
        A_num = np.array([[0.0, 0.0, -4.0 * np.sqrt(2.0)], [0.0, -4.0 * gv, 2.0 * np.sqrt(2.0)], [3.0 / np.sqrt(2.0), -np.sqrt(2.0), -gv]])
        if gv > g_EP + 1e-7:
            r = np.sort(np.roots([1.0, 5.0 * gv, 4.0 * gv**2 + 16.0, 48.0 * gv]).real)
            assert -4 * gv < r[0] < -3 * gv < -gv < r[1] < r[2] < 0
            res_p = [-(rk**2 + 5 * gv * rk + 4 * gv**2 - 8) / (3 * (3 * rk**2 + 10 * gv * rk + 4 * gv**2 + 16)) for rk in r]
            res_m = [-(rk**2 + 5 * gv * rk + 4 * gv**2 + 16) / (3 * (3 * rk**2 + 10 * gv * rk + 4 * gv**2 + 16)) for rk in r]
            assert res_p[0] > 0 and res_p[1] > 0 and res_p[2] < 0
            assert res_m[0] < 0 and res_m[1] > 0 and res_m[2] < 0 and (res_m[1] + res_m[2] < 0)
        P_dz = expm(A_num * 0.0025)
        yh, yp, ym = np.array([2.0 / 3.0, 0.0, 0.0]), np.array([-1.0 / 3.0, 0.5, 0.0]), np.array([-1.0 / 3.0, -0.5, 0.0])
        for _ in range(2000):
            assert yp[0] < 0.0 and ym[0] < 0.0 and abs(yh[0]) > max(abs(yp[0]), abs(ym[0]))
            yh, yp, ym = P_dz @ yh, P_dz @ yp, P_dz @ ym

    return {"g_EP_2b": round(g_EP, 10), "g_EP_1b": 4.0, "Jz_star_gamma0": round(Jz_star_0, 8)}


def verify_threshold_metrics_and_gaps():
    """Sec. III.B-C, Appendix B.3: N_crit = 5, six-metric formulas, and K_N / modular gap ratio 2 + 2/N."""
    N = sp.Symbol("N", integer=True, positive=True)
    diff_HBT_B2 = sp.factor((sp.Rational(5, 4) - 14 / (N * (N + 1))) - (1 - 8 / (N * (N + 1))))
    assert sp.simplify(diff_HBT_B2 - (N * (N + 1) - 24) / (4 * N * (N + 1))) == 0
    diff_neq_B2 = sp.factor((sp.Rational(1, 8) - 2 / (N * (N + 1))) - (1 - 8 / (N * (N + 1))))
    assert sp.simplify(diff_neq_B2 + (7 * N**2 + 7 * N - 48) / (8 * N * (N + 1))) == 0
    # Nonlinear Wick cumulant C^(k) = Gamma^(k) - I (x) I flips initial ordering for all N >= 3.
    diff_Wick_B2 = sp.factor((1 - sp.Rational(5, 4)) - 6 / (N * (N + 1)))
    assert sp.simplify(diff_Wick_B2 + (N * (N + 1) + 24) / (4 * N * (N + 1))) == 0

    dIa, dIb = 1 - sp.Rational(2, 1) / N, sp.Rational(1, 2) - sp.Rational(2, 1) / N
    in1 = 4 * dIa / (N + 2) + 2 * dIb / (N + 2) + dIb / (N + 2)
    in2 = 2 * (dIa + dIb) / (N + 2)
    diff_Ct_B2 = sp.factor(diff_HBT_B2 - 2 * (in1 - in2))
    assert sp.simplify(diff_Ct_B2 - (N**2 - 17 * N + 6) / (4 * (N + 1) * (N + 2))) == 0

    for Nv in (4, 5, 8):
        G_ss = np.full((Nv, Nv), 2.0 / (Nv * (Nv + 1))); np.fill_diagonal(G_ss, 4.0 / (Nv * (Nv + 1)))
        C_ss = G_ss - (2.0 / Nv) ** 2
        G1 = np.zeros((Nv, Nv)); G1[0, 0] = 1.0; G1[1, 1] = G1[2, 2] = G1[1, 2] = G1[2, 1] = 0.25
        G2 = np.zeros((Nv, Nv)); G2[0, 1] = G2[1, 0] = G2[0, 2] = G2[2, 0] = 0.5
        I_vec = np.zeros(Nv); I_vec[0] = 1.0; I_vec[1] = I_vec[2] = 0.5
        dI = I_vec - 2.0 / Nv
        G_1b = (dI[:, None] + dI[None, :]) / (Nv + 2); np.fill_diagonal(G_1b, 4.0 * dI / (Nv + 2))
        mask = ~np.eye(Nv, dtype=bool)
        assert abs((np.linalg.norm(G1 - G_ss) ** 2 - np.linalg.norm(G2 - G_ss) ** 2) - float(diff_HBT_B2.subs(N, Nv))) < 1e-13
        assert abs((np.sum((G1 - G_ss)[mask] ** 2) - np.sum((G2 - G_ss)[mask] ** 2)) - float(diff_neq_B2.subs(N, Nv))) < 1e-13
        assert abs((np.linalg.norm(G1 - np.outer(I_vec, I_vec) - C_ss) ** 2 - np.linalg.norm(G2 - np.outer(I_vec, I_vec) - C_ss) ** 2) - float(diff_Wick_B2.subs(N, Nv))) < 1e-13
        assert abs((np.linalg.norm(G1 - G_ss - G_1b) ** 2 - np.linalg.norm(G2 - G_ss - G_1b) ** 2) - float(diff_Ct_B2.subs(N, Nv))) < 1e-13

    # Strong-dephasing adiabatic-elimination Markov generator on K_N and modular bridges.
    def markov_gaps(adj: np.ndarray):
        Nv = adj.shape[0]
        states = list(itertools.combinations_with_replacement(range(Nv), 2))
        D2 = len(states)
        W2 = np.zeros((D2, D2))
        for i, (a, b) in enumerate(states):
            o1 = [0] * Nv; o1[a] += 1; o1[b] += 1
            for j, (c, d) in enumerate(states):
                if i == j:
                    continue
                o2 = [0] * Nv; o2[c] += 1; o2[d] += 1
                diff = [o1[k] - o2[k] for k in range(Nv)]
                if sum(abs(v) for v in diff) == 2:
                    m, n = diff.index(1), diff.index(-1)
                    if adj[m, n] != 0:
                        rate = 2.0 * abs(adj[m, n]) ** 2 * (o1[m] * o2[n])
                        W2[i, j] = rate
                        W2[j, j] -= rate
        W1 = 2.0 * np.abs(adj) ** 2; np.fill_diagonal(W1, 0.0); np.fill_diagonal(W1, -W1.sum(axis=1))
        e1, rem = np.sort(np.linalg.eigvalsh(-W1)), list(np.sort(np.linalg.eigvalsh(-W2)))
        for val in e1:
            k = int(np.argmin(np.abs(np.array(rem) - val)))
            assert abs(rem[k] - val) < 1e-10
            rem.pop(k)
        return e1, np.array(rem)

    kn_ratios = {}
    for Nv in (2, 3, 4, 5, 6, 8):
        e1, ec = markov_gaps(np.ones((Nv, Nv)) - np.eye(Nv))
        assert np.allclose(e1[1:], 2.0 * Nv, atol=1e-10) and np.allclose(ec, 4.0 * (Nv + 1), atol=1e-10)
        kn_ratios[f"K_{Nv}"] = round(float(ec[0] / e1[1]), 6)

    # Exact symbolic K_3 adiabatic-elimination spectrum: {0: 1, -6: 2, -16: 3} in units of J^2 / gamma.
    _, b3_k3 = annihilation_sympy(3, 2)
    H_k3 = sum((-(b3_k3[m].H * b3_k3[n]) for m in range(3) for n in range(3) if m != n), sp.zeros(6, 6))
    W_k3 = sp.zeros(6, 6)
    for a in range(6):
        for b in range(6):
            if a != b and H_k3[a, b] != 0:
                rate = 2 * abs(H_k3[a, b]) ** 2
                W_k3[a, b] = rate
                W_k3[b, b] -= rate
    assert W_k3.eigenvals() == {sp.Integer(0): 1, sp.Integer(-6): 2, sp.Integer(-16): 3}

    mod_ratios = {}
    for M in (3, 4, 5, 6):
        Nv = 2 * M
        adj = np.zeros((Nv, Nv))
        adj[:M, :M] = adj[M:, M:] = 1.0 - np.eye(M)
        adj[M - 1, M] = adj[M, M - 1] = 0.05
        e1, ec = markov_gaps(adj)
        ratio = float(ec[0] / e1[1])
        assert abs(ratio - (2.0 + 2.0 / Nv)) < 3e-3
        mod_ratios[f"Mod_N={Nv}"] = round(ratio, 6)

    return {"N_crit_B2": 5, "KN_exact_ratios": kn_ratios, "modular_bottleneck_ratios": mod_ratios}


def verify_theorem4_krein_rutman():
    """Theorem 4 (Sec. V.B, Appendix B.4): unconditional Delta D = 0, dimer P_6(0) < 0, general-N gap."""
    J, g, dk, x = sp.symbols("J g dk x", real=True, positive=True)

    # Full Fock space H_0 (+) H_1 (+) H_2 (N = 2): exact single-particle closure under inhomogeneous loss.
    full_basis = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]
    f_idx = {s: i for i, s in enumerate(full_basis)}
    b_full = [sp.zeros(6, 6) for _ in range(2)]
    for col, occ in enumerate(full_basis):
        for m in range(2):
            if occ[m] > 0:
                no = list(occ); no[m] -= 1
                b_full[m][f_idx[tuple(no)], col] = sp.sqrt(occ[m])
    n_full = [b.H * b for b in b_full]
    H_full = -J * (b_full[0].H * b_full[1] + b_full[1].H * b_full[0])
    k1, k2, g1, g2 = sp.symbols("k1 k2 g1 g2", real=True, positive=True)
    ks, gs = [k1, k2], [g1, g2]

    def L_unc_dag(O):
        res = sp.I * (H_full * O - O * H_full)
        for j in range(2):
            c1 = n_full[j] * O - O * n_full[j]
            res -= sp.Rational(1, 2) * gs[j] * (n_full[j] * c1 - c1 * n_full[j])
            res += 2 * ks[j] * (b_full[j].H * O * b_full[j] - sp.Rational(1, 2) * (n_full[j] * O + O * n_full[j]))
        return sp.simplify(res)

    E_full = {(m, n): b_full[m].H * b_full[n] for m in range(2) for n in range(2)}
    h_mat = sp.Matrix([[0, -J], [-J, 0]])
    for m in range(2):
        for n in range(2):
            rhs = sum((sp.I * (h_mat[p, m] * E_full[(p, n)] - h_mat[n, p] * E_full[(m, p)]) for p in range(2)), sp.zeros(6, 6))
            if m != n:
                rhs -= sp.Rational(1, 2) * (gs[m] + gs[n]) * E_full[(m, n)]
            rhs -= (ks[m] + ks[n]) * E_full[(m, n)]
            assert sp.simplify(L_unc_dag(E_full[(m, n)]) - rhs) == sp.zeros(6, 6)

    # Non-perturbative dimer characteristic polynomial: det(x I_9 - (L_surv^(+) + 4 k_0 I_9)) = P_3(x) P_6(x).
    _, b2 = annihilation_sympy(2, 2)
    H2 = -J * (b2[0].H * b2[1] + b2[1].H * b2[0])
    D_op = sp.diag(2, 0, -2)
    H_eff = H2 + sp.I * dk * D_op
    I3 = sp.eye(3)
    M_d = sp.Matrix([[0, g, 4 * g], [g, 0, g], [4 * g, g, 0]])
    L_shift = -sp.I * sp.kronecker_product(H_eff, I3) + sp.I * sp.kronecker_product(I3, H_eff.conjugate())
    L_shift -= sp.diag(*[M_d[a, b] for a in range(3) for b in range(3)])
    P3 = x**3 + 6 * g * x**2 + (9 * g**2 + 4 * J**2 - 4 * dk**2) * x + 4 * g**3 + 4 * J**2 * g - 16 * dk**2 * g
    P6 = (
        x**6 + 6 * g * x**5 + (9 * g**2 + 20 * J**2 - 20 * dk**2) * x**4
        + (4 * g**3 + 84 * J**2 * g - 112 * dk**2 * g) * x**3
        + (64 * J**4 - 128 * J**2 * dk**2 + 64 * J**2 * g**2 + 64 * dk**4 - 144 * dk**2 * g**2) * x**2
        + (192 * J**4 * g - 448 * J**2 * dk**2 * g + 256 * dk**4 * g - 64 * dk**2 * g**3) * x
        - 512 * J**2 * dk**2 * g**2
    )
    assert sp.expand((x * sp.eye(9) - L_shift).det(method="berkowitz") - P3 * P6) == 0
    assert sp.simplify(P6.subs(x, 0) + 512 * J**2 * dk**2 * g**2) == 0
    assert sp.simplify(-P6.subs(x, 0) / sp.diff(P6, x).subs({x: 0, dk: 0}) - 8 * g * dk**2 / (3 * J**2)) == 0
    assert sp.expand(P6.subs(g, 0) - x**2 * (x**2 + 4 * (J**2 - dk**2)) * (x**2 + 16 * (J**2 - dk**2))) == 0

    # Left-eigenoperator perturbation: O(dk) vanishes on V_cum, O(dk^2) mismatch = (4 J^2 - g^2) / (3 J^4).
    L0_dag = lambda X: sp.I * (H2 * X - X * H2) - sp.matrix_multiply_elementwise(M_d, X)
    V_d = lambda X: D_op * X + X * D_op
    Z_op, Y_op = sp.diag(1, 0, -1), sp.I * (b2[0].H * b2[1] - b2[1].H * b2[0]) / 2
    a_s, b_s = sp.symbols("a_s b_s")
    l1_op = sp.simplify((a_s * Z_op + b_s * Y_op).subs(sp.solve(list(L0_dag(a_s * Z_op + b_s * Y_op) + V_d(I3)), [a_s, b_s])))
    rho_hom = sp.diag(0, 1, 0)
    rho_nm = sp.Matrix([[sp.Rational(1, 2), 0, -sp.Rational(1, 2)], [0, 0, 0], [-sp.Rational(1, 2), 0, sp.Rational(1, 2)]])
    assert sp.simplify(sp.trace(l1_op * (rho_hom - rho_nm))) == 0
    x0, x1, cs, vs = sp.symbols("x0 x1 cs vs")
    l2_ans = sp.Matrix([[x0, sp.I * vs, cs], [-sp.I * vs, x1, -sp.I * vs], [cs, sp.I * vs, x0]])
    l2_op = l2_ans.subs(sp.solve(list(sp.simplify(L0_dag(l2_ans) + V_d(l1_op) - 8 * g / (3 * J**2) * I3)) + [2 * x0 + x1], [x0, x1, cs, vs]))
    assert sp.simplify(sp.trace(l2_op * (rho_hom - rho_nm)) - (4 * J**2 - g**2) / (3 * J**4)) == 0

    # General-N Pauli-Bose survival gap: Delta lambda_PF = (16 / (N^2 - 1)) Tr(dk (-L_0^(1))^{-1}(dk)) + O(dk^4).
    general_N = {}
    cases = [
        ("N2_dimer", np.array([[0, -1], [-1, 0]], dtype=complex), [0.45, 0.45], [-0.005, 0.005]),
        ("N3_K3", np.array([[0, -1, -1], [-1, 0, -1], [-1, -1, 0]], dtype=complex), [0.45, 0.45, 0.45], [-0.005, 0.0, 0.005]),
        ("N4_chain_inhom", -np.diag([1.0, 1.3, 0.8], 1) - np.diag([1.0, 1.3, 0.8], -1), [0.3, 0.5, 0.4, 0.6], [-4e-4, 2e-4, -1e-4, 3e-4]),
        ("N5_ring_chord", -(np.diag(np.ones(4), 1) + np.diag(np.ones(4), -1) + np.array([[0, 0, 0, 0, 1], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0]])), [0.4, 0.35, 0.5, 0.45, 0.3], [-3e-4, 1e-4, -2e-4, 4e-4, 0.0]),
    ]
    for label, H1, gvec, dkvec in cases:
        H1 = H1.astype(np.complex128)
        Nv = len(gvec)
        M1 = np.array([[0.5 * (gvec[a] + gvec[b]) * (1 if a != b else 0) for b in range(Nv)] for a in range(Nv)])
        L1 = -1j * (np.kron(H1, np.eye(Nv)) - np.kron(np.eye(Nv), H1.T)) - np.diag(M1.ravel())
        dk_mat = np.diag(dkvec).astype(np.complex128)
        X1 = (np.linalg.pinv(-L1) @ dk_mat.ravel()).reshape(Nv, Nv)
        Q_val = float(np.real(np.trace(dk_mat @ X1)))
        assert abs(Q_val - sum((gvec[m] + gvec[n]) * abs(X1[m, n]) ** 2 for m in range(Nv) for n in range(m + 1, Nv))) < 1e-12
        pred_gap = (16.0 / (Nv**2 - 1)) * Q_val

        H1_eff = H1 - 1j * np.diag(dkvec)
        sp_st = list(itertools.combinations_with_replacement(range(Nv), 2))
        Dp = len(sp_st)
        occ_p = [[(1 if a == k else 0) + (1 if b == k else 0) for k in range(Nv)] for a, b in sp_st]
        Hp = np.zeros((Dp, Dp), dtype=np.complex128)
        for i, (a, b) in enumerate(sp_st):
            Hp[i, i] = H1_eff[a, a] + H1_eff[b, b]
            for j in range(Dp):
                if i != j:
                    diff = [occ_p[i][k] - occ_p[j][k] for k in range(Nv)]
                    if sum(abs(v) for v in diff) == 2:
                        m, n = diff.index(1), diff.index(-1)
                        Hp[i, j] = H1_eff[m, n] * np.sqrt(occ_p[i][m] * occ_p[j][n])
        Mp = np.array([[0.5 * sum(gvec[k] * (occ_p[a][k] - occ_p[b][k]) ** 2 for k in range(Nv)) for b in range(Dp)] for a in range(Dp)])
        Lp = -1j * np.kron(Hp, np.eye(Dp)) + 1j * np.kron(np.eye(Dp), Hp.conj()) - np.diag(Mp.ravel())

        sm_st = list(itertools.combinations(range(Nv), 2))
        Dm = len(sm_st)
        occ_m = [[1 if k in p else 0 for k in range(Nv)] for p in sm_st]
        Hm = np.zeros((Dm, Dm), dtype=np.complex128)
        for i, (i1, i2) in enumerate(sm_st):
            Hm[i, i] = H1_eff[i1, i1] + H1_eff[i2, i2]
            for j, (j1, j2) in enumerate(sm_st):
                if i != j:
                    Hm[i, j] = (
                        (H1_eff[i2, j2] if i1 == j1 else 0.0)
                        + (H1_eff[i1, j1] if i2 == j2 else 0.0)
                        - (H1_eff[i2, j1] if i1 == j2 else 0.0)
                        - (H1_eff[i1, j2] if i2 == j1 else 0.0)
                    )
        Mm = np.array([[0.5 * sum(gvec[k] * (occ_m[a][k] - occ_m[b][k]) ** 2 for k in range(Nv)) for b in range(Dm)] for a in range(Dm)])
        Lm = -1j * np.kron(Hm, np.eye(Dm)) + 1j * np.kron(np.eye(Dm), Hm.conj()) - np.diag(Mm.ravel())

        exact_gap = float(np.max(np.linalg.eigvals(Lp).real) - np.max(np.linalg.eigvals(Lm).real))
        rel_err = abs(exact_gap - pred_gap) / exact_gap
        assert exact_gap > 0 and rel_err < 2e-3
        general_N[label] = {"exact_gap": round(exact_gap, 8), "pred_gap": round(pred_gap, 8), "rel_err": round(rel_err, 6)}

    return {"dimer_P6_0": "-512*J^2*dk^2*g^2", "general_N_Pauli_Bose": general_N}


def verify_passive_synthesis_and_spdc():
    """Sec. II.C, V.A, VI.A: passive optical synthesis (B1, B2, B2-CZ4) and exact Theorem 3 SPDC factors."""
    # B1, B2, and B2-CZ4 passive synthesis + Autonne-Takagi rank obstruction.
    U_DC = sp.Matrix([[1, sp.I], [sp.I, 1]]) / sp.sqrt(2)
    U_H = sp.Matrix([[1, 1], [1, -1]]) / sp.sqrt(2)
    S_11 = sp.Matrix([[0, 1], [1, 0]]) / sp.sqrt(2)
    assert sp.simplify(U_DC * S_11 * U_DC.T - sp.I * sp.eye(2) / sp.sqrt(2)) == sp.zeros(2, 2)
    assert sp.simplify(U_H * S_11 * U_H.T - sp.diag(1, -1) / sp.sqrt(2)) == sp.zeros(2, 2)

    phi = sp.Symbol("phi", real=True)
    U_DC_ab = sp.Matrix([[1 / sp.sqrt(2), sp.I / sp.sqrt(2), 0], [sp.I / sp.sqrt(2), 1 / sp.sqrt(2), 0], [0, 0, 1]])
    U_H_bc = sp.Matrix([[1, 0, 0], [0, 1 / sp.sqrt(2), 1 / sp.sqrt(2)], [0, 1 / sp.sqrt(2), -1 / sp.sqrt(2)]])
    U_b2 = U_H_bc * sp.diag(1, sp.exp(sp.I * phi / 2), 1) * U_DC_ab
    assert sp.simplify(U_b2.H * U_b2 - sp.eye(3)) == sp.zeros(3, 3)
    S_110 = sp.Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 0]]) / sp.sqrt(2)
    S_b2_1 = sp.simplify(U_b2 * S_110 * U_b2.T)
    S_b2_2 = sp.simplify(U_H_bc * S_110 * U_H_bc.T)
    S_b2_1_tgt = sp.Matrix([
        [1 / sp.sqrt(2), 0, 0],
        [0, sp.exp(sp.I * phi) / (2 * sp.sqrt(2)), sp.exp(sp.I * phi) / (2 * sp.sqrt(2))],
        [0, sp.exp(sp.I * phi) / (2 * sp.sqrt(2)), sp.exp(sp.I * phi) / (2 * sp.sqrt(2))],
    ])
    S_b2_2_tgt = sp.Matrix([[0, sp.Rational(1, 2), sp.Rational(1, 2)], [sp.Rational(1, 2), 0, 0], [sp.Rational(1, 2), 0, 0]])
    assert sp.simplify(S_b2_1 - sp.I * S_b2_1_tgt) == sp.zeros(3, 3)
    assert sp.simplify(S_b2_2 - S_b2_2_tgt) == sp.zeros(3, 3)
    assert sp.simplify(2 * (S_b2_1.H * S_b2_1).T - 2 * (S_b2_2.H * S_b2_2).T) == sp.zeros(3, 3)

    S_fock4 = sp.Matrix([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]) / sp.sqrt(2)
    S_adj = sp.Matrix([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]) / 2
    S_cz4 = sp.Matrix([[0, 0, 1, 1], [0, 0, 1, -1], [1, 1, 0, 0], [1, -1, 0, 0]]) / (2 * sp.sqrt(2))
    assert S_fock4.rank() == 2 and S_adj.rank() == 4 and S_cz4.rank() == 4
    assert (S_fock4.H * S_fock4).eigenvals() == {sp.Rational(1, 2): 2, sp.Integer(0): 2}
    assert (S_adj.H * S_adj).eigenvals() == {sp.Rational(1, 4): 4}
    assert (S_cz4.H * S_cz4).eigenvals() == {sp.Rational(1, 4): 4}
    SWAP_bc = sp.Matrix([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    BS_cd = sp.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1 / sp.sqrt(2), 1 / sp.sqrt(2)], [0, 0, 1 / sp.sqrt(2), -1 / sp.sqrt(2)]])
    U_sw = BS_cd * SWAP_bc
    assert sp.simplify(U_sw.H * U_sw - sp.eye(4)) == sp.zeros(4, 4)
    assert sp.simplify(U_sw * S_adj * U_sw.T - S_cz4) == sp.zeros(4, 4)

    # Dual-pumped TMSV 4-photon component has D^(4) = I_4 (invariant under any U_sw in U(4)).
    states4, b4 = annihilation_sympy(4, 4)
    _, idx4 = fock_basis(4, 4)
    psi4 = sp.zeros(len(states4), 1)
    psi4[idx4[(2, 2, 0, 0)]] = psi4[idx4[(1, 1, 1, 1)]] = psi4[idx4[(0, 0, 2, 2)]] = 1 / sp.sqrt(3)
    for m in range(4):
        for n in range(4):
            assert (psi4.H * (b4[m].H * b4[n]) * psi4)[0, 0] == (1 if m == n else 0)

    # Theorem 3: exact 4-class inclusive and exclusive 4-photon SPDC factors vs brute-force Sym^4(C^N).
    N, eta_t = sp.symbols("N eta_tot", positive=True)
    pref = sp.Rational(12, 1) / ((N + 2) * (N + 3))
    c_11 = sp.Rational(1, 2) * (N - 1) * (N - 2)
    c_21 = 2 * (N - 2) * (2 - eta_t)
    c_31 = 2 * (3 - 3 * eta_t + eta_t**2)
    c_22 = (2 - eta_t) ** 2
    R_incl = sp.simplify(pref * (c_11 + c_21 + c_31 + c_22))
    R_excl = sp.simplify(pref * (c_11 * (1 - eta_t) ** 2 + c_21 * (1 - eta_t) + c_31 + c_22))
    assert sp.simplify(R_incl - (6 - 24 * eta_t / (N + 2) + 36 * eta_t**2 / ((N + 2) * (N + 3)))) == 0
    assert sp.simplify(R_excl - (6 - 12 * N * eta_t / (N + 2) + 6 * N * (N + 1) * eta_t**2 / ((N + 2) * (N + 3)))) == 0

    for Nv in (4, 6, 8, 12):
        st4, _ = fock_basis(Nv, 4)
        v_inc = v_exc = sp.Integer(0)
        for occ in st4:
            if occ[0] >= 1 and occ[1] >= 1:
                p_qr = (1 - (1 - eta_t) ** occ[0]) * (1 - (1 - eta_t) ** occ[1])
                v_inc += p_qr
                v_exc += p_qr * (1 - eta_t) ** (4 - occ[0] - occ[1])
        v2 = sp.Rational(2, Nv * (Nv + 1)) * eta_t**2
        assert sp.simplify(v_inc / (len(st4) * v2) - R_incl.subs(N, Nv)) == 0
        assert sp.simplify(v_exc / (len(st4) * v2) - R_excl.subs(N, Nv)) == 0

    shift_inc = float(sp.Rational(1, 50) * R_incl.subs({N: 8, eta_t: sp.Rational(1, 4)}) * 100)
    shift_exc = float(sp.Rational(1, 50) * R_excl.subs({N: 8, eta_t: sp.Rational(1, 4)}) * 100)
    assert abs(shift_inc - 10.8409090909) < 1e-8 and abs(shift_exc - 7.6909090909) < 1e-8

    # Three-photon NOON passive synthesis no-go via single-particle D-spectrum invariance.
    states3, b3 = annihilation_sympy(3, 3)
    _, idx3 = fock_basis(3, 3)
    for sgn in (+1, -1):
        psi_noon3 = sp.zeros(len(states3), 1)
        psi_noon3[idx3[(3, 0, 0)]] = 1 / sp.sqrt(2)
        psi_noon3[idx3[(0, 3, 0)]] = sgn / sp.sqrt(2)
        D_n3 = sp.Matrix([[sp.simplify((psi_noon3.H * (b3[m].H * b3[n]) * psi_noon3)[0, 0]) for n in range(3)] for m in range(3)])
        spec_n3 = D_n3.eigenvals()
        assert spec_n3 == {sp.Rational(3, 2): 2, sp.Integer(0): 1}
        for prod_spec in ({sp.Integer(3): 1, sp.Integer(0): 2}, {sp.Integer(2): 1, sp.Integer(1): 1, sp.Integer(0): 1}, {sp.Integer(1): 3}):
            assert spec_n3 != prod_spec

    return {
        "passive_synthesis_B1_B2_CZ4_verified": True,
        "SPDC_N8_eta025_p002_incl_pct": round(shift_inc, 6),
        "SPDC_N8_eta025_p002_excl_pct": round(shift_exc, 6),
    }


def main():
    print(json.dumps({
        "dimer_and_LEP2": verify_dimer_and_lep2(),
        "threshold_metrics_and_gaps": verify_threshold_metrics_and_gaps(),
        "Theorem_4_Krein_Rutman": verify_theorem4_krein_rutman(),
        "passive_synthesis_and_Theorem_3_SPDC": verify_passive_synthesis_and_spdc(),
    }, indent=2))


if __name__ == "__main__":
    main()
