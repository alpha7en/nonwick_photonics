"""Symbolic (SymPy) and numerical verification of Theorem 1 (orthogonal splitting),
Theorem 2 (shot-by-shot g^(1)-invisibility and finite-correlation-length Redfield / OU HEOM),
three-photon filtration and (|Xi_1>, |Xi_2>) (Sec. VI.A), and the Bose-Hubbard BBGKY
linearity witness [Eq. (16)] (Sec. II.B, VI.A, Appendices A-B).

Exits 0 iff all algebraic and numerical identities hold.
Usage: python -I proofs/algebra_and_witness.py
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


def annihilation_numpy(N: int, k_ph: int):
    states_k, _ = fock_basis(N, k_ph)
    states_km1, idx_km1 = fock_basis(N, k_ph - 1)
    b_ops = [np.zeros((len(states_km1), len(states_k)), dtype=np.complex128) for _ in range(N)]
    for col, occ in enumerate(states_k):
        for m in range(N):
            if occ[m] > 0:
                new_occ = list(occ)
                new_occ[m] -= 1
                b_ops[m][idx_km1[tuple(new_occ)], col] = np.sqrt(occ[m])
    return states_k, b_ops


def lift_sym_k_sympy(U_mat: sp.Matrix, k_ph: int) -> sp.Matrix:
    N = U_mat.rows
    states_k, idx_k = fock_basis(N, k_ph)
    U_sk = sp.zeros(len(states_k), len(states_k))
    for col, occ_in in enumerate(states_k):
        modes_in = [m for m, c in enumerate(occ_in) for _ in range(c)]
        norm_in = sp.sqrt(sp.prod(sp.factorial(c) for c in occ_in))
        for out_modes in itertools.product(range(N), repeat=k_ph):
            amp = sp.prod(U_mat[p, modes_in[j]] for j, p in enumerate(out_modes))
            if amp == 0:
                continue
            occ_out = [0] * N
            for p in out_modes:
                occ_out[p] += 1
            norm_out = sp.sqrt(sp.prod(sp.factorial(c) for c in occ_out))
            U_sk[idx_k[tuple(occ_out)], col] += amp * norm_out / norm_in
    return sp.simplify(U_sk)


def lift_sym2_numpy(U_mat: np.ndarray) -> np.ndarray:
    N = U_mat.shape[0]
    states2, _ = fock_basis(N, 2)
    D2 = len(states2)
    U_s2 = np.zeros((D2, D2), dtype=np.complex128)
    for col, occ_in in enumerate(states2):
        m1, m2 = [m for m, c in enumerate(occ_in) for _ in range(c)]
        norm_in = np.sqrt(2.0) if m1 == m2 else 1.0
        for row, occ_out in enumerate(states2):
            p1, p2 = [p for p, c in enumerate(occ_out) for _ in range(c)]
            norm_out = np.sqrt(2.0) if p1 == p2 else 1.0
            U_s2[row, col] = (U_mat[p1, m1] * U_mat[p2, m2] + U_mat[p1, m2] * U_mat[p2, m1]) / (norm_in * norm_out)
    return U_s2


def comm(A, B):
    return A * B - B * A


def verify_theorem1_splitting():
    """Theorem 1 (Sec. II.B, Appendix A): V_{1b} (+) V_cum splitting, closed-form P_{1b}, and C_tilde."""
    out = {}
    for N in (2, 3, 4):
        states2, b2 = annihilation_sympy(N, 2)
        _, b1 = annihilation_sympy(N, 1)
        D2 = len(states2)
        E2 = {(m, n): b2[m].H * b2[n] for m in range(N) for n in range(N)}

        # Gram matrix G_{(m,n),(p,q)} = (N + 2) delta_{mp} delta_{nq} + delta_{mn} delta_{pq}.
        for m in range(N):
            for n in range(N):
                assert sp.simplify(sp.trace(E2[(n, m)]) - (N + 1) * (1 if m == n else 0)) == 0
                for p in range(N):
                    for q in range(N):
                        tr_4 = sp.trace((b1[p] * b2[n]).H * (b1[m] * b2[q]))
                        assert sp.simplify(tr_4 - ((1 if (m == p and n == q) else 0) + (1 if (m == n and p == q) else 0))) == 0
                        lhs_g = sp.trace(E2[(m, n)].H * E2[(p, q)])
                        rhs_g = (N + 2) * (1 if (m == p and n == q) else 0) + (1 if (m == n and p == q) else 0)
                        assert sp.simplify(lhs_g - rhs_g) == 0

        S_mat = sp.Matrix.hstack(*[sp.Matrix(list(E2[(m, n)])) for m in range(N) for n in range(N)])
        G_mat = S_mat.H * S_mat
        vec_I = sp.Matrix([1 if m == n else 0 for m in range(N) for n in range(N)])
        assert G_mat == (N + 2) * sp.eye(N * N) + vec_I * vec_I.T

        G_inv = (sp.eye(N * N) - vec_I * vec_I.T / (2 * (N + 1))) / (N + 2)
        assert sp.simplify(G_mat * G_inv - sp.eye(N * N)) == sp.zeros(N * N)
        assert G_mat.eigenvals() == {sp.Integer(N + 2): N * N - 1, sp.Integer(2 * N + 2): 1}

        P_1b = S_mat * G_inv * S_mat.H
        Q_cum = sp.eye(D2 * D2) - P_1b
        assert P_1b == P_1b.H and sp.simplify(P_1b * P_1b - P_1b) == sp.zeros(D2 * D2)
        dim_cum = N * N * (N - 1) * (N + 3) // 4
        assert sp.trace(P_1b) == N * N and sp.trace(Q_cum) == dim_cum

        if N <= 3:
            A_sym = sp.Matrix(D2, D2, lambda a, b: sp.Symbol(f"A_{a}_{b}"))
            D_A = sp.Matrix(N, N, lambda m, n: sp.trace(E2[(m, n)] * A_sym))
            tr_D = sp.trace(D_A)
            assert sp.simplify(tr_D - 2 * sp.trace(A_sym)) == 0
            X_A = (D_A.T - (tr_D / (2 * (N + 1))) * sp.eye(N)) / (N + 2)
            P1b_A = sum((X_A[m, n] * E2[(m, n)] for m in range(N) for n in range(N)), sp.zeros(D2, D2))
            assert sp.simplify(sp.Matrix(list(P1b_A)) - P_1b * sp.Matrix(list(A_sym))) == sp.zeros(D2 * D2, 1)

        rho_ss = sp.Matrix(list(sp.eye(D2) / D2))
        assert sp.simplify(P_1b * rho_ss - rho_ss) == sp.zeros(D2 * D2, 1)

        if N <= 3:
            # Arbitrary complex Hermitian h and inhomogeneous local dephasing {gamma_j}.
            H1 = sp.zeros(N, N)
            for m in range(N):
                H1[m, m] = sp.Symbol(f"eps_{m}", real=True)
                for n in range(m + 1, N):
                    jr, ji = sp.Symbol(f"Jr_{m}{n}", real=True), sp.Symbol(f"Ji_{m}{n}", real=True)
                    H1[m, n], H1[n, m] = jr + sp.I * ji, jr - sp.I * ji
            gammas = [sp.Symbol(f"g_{m}", real=True) for m in range(N)]

            H2 = sum((H1[m, n] * E2[(m, n)] for m in range(N) for n in range(N)), sp.zeros(D2, D2))
            I_D2, I_N = sp.eye(D2), sp.eye(N)
            L2 = -sp.I * (sp.kronecker_product(H2, I_D2) - sp.kronecker_product(I_D2, H2.T))
            L1 = -sp.I * (sp.kronecker_product(H1, I_N) - sp.kronecker_product(I_N, H1.T))
            for j in range(N):
                nj = [states2[a][j] for a in range(D2)]
                L2 += sp.diag(*[-sp.Rational(1, 2) * gammas[j] * (nj[a] - nj[b]) ** 2 for a in range(D2) for b in range(D2)])
                L1 += sp.diag(*[-sp.Rational(1, 2) * gammas[j] * ((1 if a == j else 0) - (1 if b == j else 0)) ** 2 for a in range(N) for b in range(N)])

            assert sp.simplify(P_1b * L2 * Q_cum) == sp.zeros(D2 * D2)
            assert sp.simplify(Q_cum * L2 * P_1b) == sp.zeros(D2 * D2)
            assert sp.simplify(L2 * S_mat - S_mat * L1) == sp.zeros(D2 * D2, N * N)
            assert sp.simplify(S_mat.H * L2 - L1 * S_mat.H) == sp.zeros(N * N, D2 * D2)

        # HBT projection on V_{1b} vs Wick cumulant contamination.
        dI = [sp.Symbol(f"dI_{m}", real=True) for m in range(N)]
        dI_sub = dI[:-1] + [-sum(dI[:-1])]
        drho_1b = sum((dI_sub[m] * E2[(m, m)] for m in range(N)), sp.zeros(D2, D2)) / (N + 2)
        for q in range(N):
            for r in range(N):
                Gamma_qr = E2[(q, q)] * (E2[(q, q)] - sp.eye(D2)) if q == r else E2[(q, q)] * E2[(r, r)]
                val_1b = sp.simplify(sp.trace(Gamma_qr * drho_1b))
                assert sp.simplify(val_1b - (1 + (1 if q == r else 0)) * (dI_sub[q] + dI_sub[r]) / (N + 2)) == 0
                G_ss = sp.Rational(2 * (1 + (1 if q == r else 0)), N * (N + 1))
                Iq, Ir = sp.Rational(2, N) + dI_sub[q], sp.Rational(2, N) + dI_sub[r]
                C_wick = sp.expand((G_ss + val_1b - Iq * Ir) - (G_ss - sp.Rational(4, N * N)))
                exp_wick = (-sp.Rational(N + 4, N * (N + 2)) * (dI_sub[q] + dI_sub[r]) - dI_sub[q] * dI_sub[r]) if q != r else (-sp.Rational(8, N * (N + 2)) * dI_sub[q] - dI_sub[q] ** 2)
                assert sp.simplify(C_wick - exp_wick) == 0
                assert sp.simplify(S_mat.H * (Q_cum * sp.Matrix(list(Gamma_qr)))) == sp.zeros(N * N, 1)

        out[f"N={N}_symbolic"] = {"D2": D2, "dim_V1b": N * N, "dim_Vcum": dim_cum}

    # Spectral union Spec(L^(2)) = Spec(L^(1)) |_| Spec(L^(2)|_{V_cum}) for N = 4, 5, 6.
    rng = np.random.default_rng(20261119)
    for N in (4, 5, 6):
        states2, b2 = annihilation_numpy(N, 2)
        D2 = len(states2)
        E2 = {(m, n): b2[m].conj().T @ b2[n] for m in range(N) for n in range(N)}
        S_mat = np.column_stack([E2[(m, n)].ravel() for m in range(N) for n in range(N)])
        vec_I = np.eye(N, dtype=np.complex128).ravel()
        P_1b = S_mat @ ((np.eye(N * N) - np.outer(vec_I, vec_I) / (2 * (N + 1))) / (N + 2)) @ S_mat.conj().T
        Q_cum = np.eye(D2 * D2) - P_1b

        M_rand = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
        H1 = 0.5 * (M_rand + M_rand.conj().T)
        gammas = rng.uniform(0.2, 2.5, size=N)
        H2 = sum(H1[m, n] * E2[(m, n)] for m in range(N) for n in range(N))

        L2 = -1j * (np.kron(H2, np.eye(D2)) - np.kron(np.eye(D2), H2.T))
        M_d2 = sum(0.5 * gammas[j] * (np.array([s[j] for s in states2], float)[:, None] - np.array([s[j] for s in states2], float)[None, :]) ** 2 for j in range(N))
        L2 -= np.diag(M_d2.ravel())

        L1 = -1j * (np.kron(H1, np.eye(N)) - np.kron(np.eye(N), H1.T))
        M_d1 = 0.5 * (gammas[:, None] + gammas[None, :])
        np.fill_diagonal(M_d1, 0.0)
        L1 -= np.diag(M_d1.ravel())

        err_inv = max(float(np.linalg.norm(P_1b @ L2 @ Q_cum)), float(np.linalg.norm(Q_cum @ L2 @ P_1b)))
        assert err_inv < 1e-13

        u_q, _, _ = np.linalg.svd(Q_cum)
        B_cum = u_q[:, : D2 * D2 - N * N]
        ev_L2 = np.linalg.eigvals(L2)
        ev_union = np.concatenate([np.linalg.eigvals(L1), np.linalg.eigvals(B_cum.conj().T @ L2 @ B_cum)])
        used = np.zeros(len(ev_union), dtype=bool)
        max_spec_err = 0.0
        for e in ev_L2:
            dists = np.abs(ev_union - e)
            dists[used] = np.inf
            k_best = int(np.argmin(dists))
            used[k_best] = True
            max_spec_err = max(max_spec_err, float(dists[k_best]))
        assert max_spec_err < 1e-11
        out[f"N={N}_numeric"] = {"off_diag_invariance_norm": err_inv, "max_spectral_union_err": max_spec_err}
    return out


def verify_theorem2_invisibility_and_redfield():
    """Theorem 2 (Sec. II.B, Appendix B.2): shot-by-shot intertwining, discrete noise, Redfield vs OU HEOM."""
    K_herm = sp.Matrix([
        [sp.Rational(1, 2), sp.Rational(1, 3) + sp.I * sp.Rational(1, 5), sp.Rational(-1, 4)],
        [sp.Rational(1, 3) - sp.I * sp.Rational(1, 5), sp.Rational(-1, 3), sp.Rational(2, 5) - sp.I * sp.Rational(1, 2)],
        [sp.Rational(-1, 4), sp.Rational(2, 5) + sp.I * sp.Rational(1, 2), sp.Rational(1, 5)],
    ])
    U3 = sp.simplify((sp.eye(3) - sp.I * K_herm) * (sp.eye(3) + sp.I * K_herm).inv())
    U3_s2 = lift_sym_k_sympy(U3, 2)
    _, b2_3 = annihilation_sympy(3, 2)
    E2_3 = {(m, n): b2_3[m].H * b2_3[n] for m in range(3) for n in range(3)}

    for m in range(3):
        for n in range(3):
            lhs = sp.simplify(U3_s2.H * E2_3[(m, n)] * U3_s2)
            rhs = sum((sp.conjugate(U3[m, p]) * U3[n, q] * E2_3[(p, q)] for p in range(3) for q in range(3)), sp.zeros(6, 6))
            assert sp.simplify(lhs - rhs) == sp.zeros(6, 6)
            lhs2 = sp.simplify(U3_s2 * E2_3[(m, n)] * U3_s2.H)
            X_rot = U3 * sp.Matrix(3, 3, lambda a, b: 1 if (a == m and b == n) else 0) * U3.H
            rhs2 = sum((X_rot[p, q] * E2_3[(p, q)] for p in range(3) for q in range(3)), sp.zeros(6, 6))
            assert sp.simplify(lhs2 - rhs2) == sp.zeros(6, 6)

    # Single-realization check on N = 6 Clements mesh and disordered segmented waveguides.
    rng = np.random.default_rng(424242)
    N6 = 6
    states2_6, b2_6 = annihilation_numpy(N6, 2)
    _, idx2_6 = fock_basis(N6, 2)
    D2_6 = len(states2_6)
    E2_6 = {(m, n): b2_6[m].conj().T @ b2_6[n] for m in range(N6) for n in range(N6)}
    extract_D = lambda psi: np.array([[np.vdot(psi, E2_6[(m, n)] @ psi) for n in range(N6)] for m in range(N6)])

    p_hom = np.zeros(D2_6, dtype=np.complex128); p_hom[idx2_6[(0, 0, 1, 1, 0, 0)]] = 1.0
    p_np = np.zeros(D2_6, dtype=np.complex128); p_np[idx2_6[(0, 0, 2, 0, 0, 0)]] = p_np[idx2_6[(0, 0, 0, 2, 0, 0)]] = 1 / np.sqrt(2)
    p_b2_1 = np.zeros(D2_6, dtype=np.complex128)
    p_b2_1[idx2_6[(0, 2, 0, 0, 0, 0)]] = 1 / np.sqrt(2)
    p_b2_1[idx2_6[(0, 0, 2, 0, 0, 0)]] = p_b2_1[idx2_6[(0, 0, 0, 2, 0, 0)]] = 0.5 / np.sqrt(2)
    p_b2_1[idx2_6[(0, 0, 1, 1, 0, 0)]] = 0.5
    p_b2_2 = np.zeros(D2_6, dtype=np.complex128); p_b2_2[idx2_6[(0, 1, 1, 0, 0, 0)]] = p_b2_2[idx2_6[(0, 1, 0, 1, 0, 0)]] = 1 / np.sqrt(2)
    p_adj = np.zeros(D2_6, dtype=np.complex128); p_adj[idx2_6[(0, 1, 1, 0, 0, 0)]] = p_adj[idx2_6[(0, 0, 0, 1, 1, 0)]] = 1 / np.sqrt(2)
    p_cz4 = np.zeros(D2_6, dtype=np.complex128)
    p_cz4[idx2_6[(0, 1, 0, 1, 0, 0)]] = p_cz4[idx2_6[(0, 1, 0, 0, 1, 0)]] = p_cz4[idx2_6[(0, 0, 1, 1, 0, 0)]] = 0.5
    p_cz4[idx2_6[(0, 0, 1, 0, 1, 0)]] = -0.5

    U_clem = np.eye(N6, dtype=np.complex128)
    for layer in range(5):
        U_mzi = np.eye(N6, dtype=np.complex128)
        for k in range(layer % 2, N6 - 1, 2):
            U_mzi[k : k + 2, k : k + 2] = np.array([[np.cos(0.44), 1j * np.sin(0.44)], [1j * np.sin(0.44), np.cos(0.44)]])
        U_clem = np.diag(np.exp(-1j * rng.normal(0.0, 0.6, size=N6))) @ U_mzi @ U_clem
    U_clem_s2 = lift_sym2_numpy(U_clem)

    U_fsw = np.eye(N6, dtype=np.complex128)
    H_hop = -(np.diag(np.ones(N6 - 1), 1) + np.diag(np.ones(N6 - 1), -1)).astype(np.complex128)
    for _ in range(10):
        U_fsw = expm(-1j * (H_hop + np.diag(rng.uniform(-1.5, 1.5, size=N6))) * 0.2) @ U_fsw
    U_fsw_s2 = lift_sym2_numpy(U_fsw)

    max_shot_diff = 0.0
    for p1, p2 in ((p_hom, p_np), (p_b2_1, p_b2_2), (p_adj, p_cz4)):
        assert np.linalg.norm(extract_D(p1) - extract_D(p2)) < 1e-15
        for U_s2, U_1 in ((U_clem_s2, U_clem), (U_fsw_s2, U_fsw)):
            D1_o, D2_o = extract_D(U_s2 @ p1), extract_D(U_s2 @ p2)
            diff = max(np.linalg.norm(D1_o - D2_o), np.linalg.norm(D1_o - U_1.conj() @ extract_D(p1) @ U_1.T))
            assert diff < 1e-14
            max_shot_diff = max(max_shot_diff, float(diff))

    # Discrete Gaussian vs uniform phase-noise cumulant expansion.
    gamma, dz, W_phi, k_sym = sp.symbols("gamma dz W_phi k", positive=True)
    rate_g = -sp.log(sp.exp(-sp.Rational(1, 2) * gamma * dz * k_sym**2)) / dz
    assert sp.simplify(rate_g / rate_g.subs(k_sym, 1) - k_sym**2) == 0
    ratio_unif = sp.series(
        ((-sp.log(sp.sin(2 * W_phi) / (2 * W_phi))) / (-sp.log(sp.sin(W_phi) / W_phi))).subs(W_phi, sp.sqrt(3 * gamma * dz)),
        dz, 0, 3,
    ).removeO()
    assert sp.simplify(ratio_unif - (4 + sp.Rational(6, 5) * gamma * dz + sp.Rational(179, 175) * (gamma * dz) ** 2)) == 0

    # Exact dimer Redfield / TCL2 generator vs Ornstein-Uhlenbeck HEOM (Appendix B.2).
    J, lc = sp.symbols("J lc", real=True, positive=True)
    u, c2, v = sp.symbols("u c2 v", real=True)
    p = u + sp.Rational(1, 3)
    rho_d = sp.Matrix([[(1 - p) / 2, sp.I * v, c2], [-sp.I * v, p, -sp.I * v], [c2, sp.I * v, (1 - p) / 2]])
    H2_d = -J * sp.Matrix([[0, sp.sqrt(2), 0], [sp.sqrt(2), 0, sp.sqrt(2)], [0, sp.sqrt(2), 0]])
    n0_d, n1_d = sp.diag(2, 1, 0), sp.diag(0, 1, 2)
    Jz_op = n0_d - n1_d
    Jy_op = sp.I * sp.Matrix([[0, -sp.sqrt(2), 0], [sp.sqrt(2), 0, -sp.sqrt(2)], [0, sp.sqrt(2), 0]])

    Lam = 1 / (1 + 4 * J**2 * lc**2)
    N_diff = Lam * (Jz_op + 2 * J * lc * Jy_op)
    assert sp.simplify(N_diff + sp.I * lc * comm(H2_d, N_diff) - Jz_op) == sp.zeros(3, 3)

    rhs_red = -sp.I * comm(H2_d, rho_d) - sp.Rational(1, 4) * gamma * comm(Jz_op, comm(N_diff, rho_d))
    du, dc2, dv = sp.expand(sp.simplify(rhs_red[1, 1])), sp.expand(sp.simplify(rhs_red[0, 2])), sp.expand(sp.simplify(rhs_red[0, 1] / sp.I))
    A_red = sp.simplify(sp.Matrix([[du.coeff(q) for q in (u, c2, v)], [dc2.coeff(q) for q in (u, c2, v)], [dv.coeff(q) for q in (u, c2, v)]]))
    A_exp = sp.Matrix([
        [0, 0, -4 * sp.sqrt(2) * J],
        [0, -4 * gamma * Lam, 2 * sp.sqrt(2) * J * (1 + 2 * gamma * lc * Lam)],
        [3 * J / sp.sqrt(2) * (1 + gamma * lc * Lam), -sp.sqrt(2) * J * (1 - gamma * lc * Lam), -gamma * Lam],
    ])
    assert sp.simplify(A_red - A_exp) == sp.zeros(3, 3)
    assert sp.simplify(A_red[1, 1] / A_red[2, 2] - 4) == 0

    rhs_trunc = -sp.I * comm(H2_d, rho_d) - sp.Rational(1, 2) * gamma * sum(
        (
            comm(nj, comm(nj, rho_d))
            - sp.I * lc * comm(nj, comm(comm(H2_d, nj), rho_d))
            - lc**2 * comm(nj, comm(comm(H2_d, comm(H2_d, nj)), rho_d))
            for nj in (n0_d, n1_d)
        ),
        sp.zeros(3, 3),
    )
    A_trunc = sp.Matrix([
        [sp.expand(sp.simplify(rhs_trunc[1, 1])).coeff(q) for q in (u, c2, v)],
        [sp.expand(sp.simplify(rhs_trunc[0, 2])).coeff(q) for q in (u, c2, v)],
        [sp.expand(sp.simplify(rhs_trunc[0, 1] / sp.I)).coeff(q) for q in (u, c2, v)],
    ])
    assert sp.simplify(A_trunc - A_red.applyfunc(lambda e: sp.series(e, lc, 0, 3).removeO())) == sp.zeros(3, 3)

    # Ornstein-Uhlenbeck HEOM cross-check confirming O(gamma^2) convergence at finite J l_c = 0.2.
    def heom_slow_eigs(J_v: float, g_v: float, lc_v: float, K: int = 8):
        H_n = np.array([[0, -np.sqrt(2) * J_v, 0], [-np.sqrt(2) * J_v, 0, -np.sqrt(2) * J_v], [0, -np.sqrt(2) * J_v, 0]], dtype=complex)
        D_n = np.diag([2.0, 0.0, -2.0])
        I3 = np.eye(3)
        L_H = -1j * (np.kron(H_n, I3) - np.kron(I3, H_n.T))
        L_D = -1j * (np.kron(D_n, I3) - np.kron(I3, D_n.T))
        M = np.zeros((9 * (K + 1), 9 * (K + 1)), dtype=complex)
        for k in range(K + 1):
            sl = slice(9 * k, 9 * (k + 1))
            M[sl, sl] = L_H - (k / lc_v) * np.eye(9)
            if k + 1 <= K:
                M[sl, slice(9 * (k + 1), 9 * (k + 2))] = L_D
            if k - 1 >= 0:
                M[sl, slice(9 * (k - 1), 9 * k)] = (k * g_v / (4.0 * lc_v)) * L_D
        slow = [e for e in np.linalg.eigvals(M) if e.real > -2.5]
        return (
            min(slow, key=lambda e: abs(e - (-3 * g_v / (1 + 4 * J_v**2 * lc_v**2)))),
            min(slow, key=lambda e: abs(e - (-g_v / (1 + 4 * J_v**2 * lc_v**2) + 4.0j * J_v))),
        )

    heom_ratios = []
    for g_v in (0.02, 0.005):
        r_h, o_h = heom_slow_eigs(1.0, g_v, 0.2)
        ev_t = np.linalg.eigvals(np.array(A_red.subs({J: 1.0, gamma: g_v, lc: 0.2}).tolist(), dtype=float))
        err_r = abs(r_h - min(ev_t, key=lambda e: abs(e.imag))) / (g_v**2)
        err_o = abs(o_h - max(ev_t, key=lambda e: e.imag)) / (g_v**2)
        assert err_r < 0.2 and err_o < 0.2
        heom_ratios.append((g_v, round(err_r, 5), round(err_o, 5)))
    assert abs(heom_ratios[0][1] - heom_ratios[1][1]) / heom_ratios[1][1] < 0.05

    # General N = 3 Redfield: V_{1b} (+) V_cum invariance and exact 4:1 diagonal rate ratio.
    M_r3 = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
    H1_3 = 0.5 * (M_r3 + M_r3.conj().T)
    gam_3, lc_n = np.array([0.4, 0.9, 1.3]), 0.25
    I3_n, I6 = np.eye(3), np.eye(6)
    ad_h1 = np.kron(H1_3, I3_n) - np.kron(I3_n, H1_3.T)
    res_op1 = np.linalg.inv(np.eye(9) + 1j * lc_n * ad_h1)
    _, b2_3n = annihilation_numpy(3, 2)
    _, idx2_3n = fock_basis(3, 2)
    E2_3n = {(m, n): b2_3n[m].conj().T @ b2_3n[n] for m in range(3) for n in range(3)}
    H2_3n = sum(H1_3[m, n] * E2_3n[(m, n)] for m in range(3) for n in range(3))
    L_red_3 = -1j * (np.kron(H2_3n, I6) - np.kron(I6, H2_3n.T))
    L_red_1 = -1j * ad_h1
    for j in range(3):
        Pj = np.zeros((3, 3), dtype=np.complex128); Pj[j, j] = 1.0
        Nj_1 = (res_op1 @ Pj.ravel()).reshape(3, 3)
        Nj_2 = sum(Nj_1[m, n] * E2_3n[(m, n)] for m in range(3) for n in range(3))
        nj_2 = E2_3n[(j, j)]
        L_red_3 -= 0.5 * gam_3[j] * ((np.kron(nj_2, I6) - np.kron(I6, nj_2.T)) @ (np.kron(Nj_2, I6) - np.kron(I6, Nj_2.T)))
        L_red_1 -= 0.5 * gam_3[j] * ((np.kron(Pj, I3_n) - np.kron(I3_n, Pj.T)) @ (np.kron(Nj_1, I3_n) - np.kron(I3_n, Nj_1.T)))

    S3 = np.column_stack([E2_3n[(m, n)].ravel() for m in range(3) for n in range(3)])
    vI3 = np.eye(3).ravel()
    P1b_3 = S3 @ ((np.eye(9) - np.outer(vI3, vI3) / 8.0) / 5.0) @ S3.conj().T
    Qcum_3 = np.eye(36) - P1b_3
    assert np.linalg.norm(P1b_3 @ L_red_3 @ Qcum_3) < 1e-13 and np.linalg.norm(Qcum_3 @ L_red_3 @ P1b_3) < 1e-13
    for a in range(3):
        for b in range(a + 1, 3):
            oa, ob = [0, 0, 0], [0, 0, 0]; oa[a] = 2; ob[b] = 2
            ia, ib = idx2_3n[tuple(oa)], idx2_3n[tuple(ob)]
            assert abs(np.real(L_red_3[ia * 6 + ib, ia * 6 + ib]) / np.real(L_red_1[a * 3 + b, a * 3 + b]) - 4.0) < 1e-13

    return {
        "max_shot_by_shot_D_diff_N6": max_shot_diff,
        "uniform_phase_ratio_series": str(ratio_unif),
        "heom_vs_Redfield_O_g2_residuals": heom_ratios,
        "general_N3_Redfield_invariance_and_4to1": True,
    }


def verify_three_photon_filtration():
    """Sec. VI.A: 3-photon dephasing ladder {1..7, 9}*gamma, 3-tier filtration, and (|Xi_1>, |Xi_2>)."""
    states3_6, _ = fock_basis(6, 3)
    rates = {sum((states3_6[a][j] - states3_6[b][j]) ** 2 for j in range(6)) // 2 for a in range(len(states3_6)) for b in range(a + 1, len(states3_6))}
    assert sorted(rates) == [1, 2, 3, 4, 5, 6, 7, 9]

    N = 3
    states3, b3 = annihilation_sympy(N, 3)
    states2, b2 = annihilation_sympy(N, 2)
    D3, D2 = len(states3), len(states2)
    S1 = sp.Matrix.hstack(*[sp.Matrix(list(b3[m].H * b3[n])) for m in range(N) for n in range(N)])
    B_ops = [
        (b2[modes[0]] * b3[modes[1]]) / (sp.sqrt(2) if modes[0] == modes[1] else sp.Integer(1))
        for occ2 in states2
        for modes in [[m for m, c in enumerate(occ2) for _ in range(c)]]
    ]
    S2 = sp.Matrix.hstack(*[sp.Matrix(list(B_ops[a].H * B_ops[b])) for a in range(D2) for b in range(D2)])
    assert S1.rank() == 9 and S2.rank() == 36 and sp.Matrix.hstack(S2, S1).rank() == 36

    S1_n, S2_n = np.array(S1.tolist(), dtype=np.complex128), np.array(S2.tolist(), dtype=np.complex128)
    P1 = S1_n @ np.linalg.inv(S1_n.conj().T @ S1_n) @ S1_n.conj().T
    P_W2 = S2_n @ np.linalg.inv(S2_n.conj().T @ S2_n) @ S2_n.conj().T
    P2_cum, P3_cum = P_W2 - P1, np.eye(D3 * D3) - P_W2

    rng = np.random.default_rng(333333)
    M3 = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    H1_n = 0.5 * (M3 + M3.conj().T)
    gam_n = np.array([0.35, 1.15, 0.72])
    b3_n = [np.array(m.tolist(), dtype=np.complex128) for m in b3]
    b2_n = [np.array(m.tolist(), dtype=np.complex128) for m in b2]
    H3_n = sum(H1_n[m, n] * (b3_n[m].conj().T @ b3_n[n]) for m in range(N) for n in range(N))
    H2_n = sum(H1_n[m, n] * (b2_n[m].conj().T @ b2_n[n]) for m in range(N) for n in range(N))
    L3_n = -1j * (np.kron(H3_n, np.eye(10)) - np.kron(np.eye(10), H3_n.T)) - np.diag(
        sum(0.5 * gam_n[j] * (np.array([s[j] for s in states3], float)[:, None] - np.array([s[j] for s in states3], float)[None, :]) ** 2 for j in range(N)).ravel()
    )
    L2_n = -1j * (np.kron(H2_n, np.eye(6)) - np.kron(np.eye(6), H2_n.T)) - np.diag(
        sum(0.5 * gam_n[j] * (np.array([s[j] for s in states2], float)[:, None] - np.array([s[j] for s in states2], float)[None, :]) ** 2 for j in range(N)).ravel()
    )

    intertwine_err = float(np.linalg.norm(L3_n @ S2_n - S2_n @ L2_n))
    assert intertwine_err < 1e-12
    projs = (P1, P2_cum, P3_cum)
    max_off = max(float(np.linalg.norm(projs[a] @ L3_n @ projs[b])) for a in range(3) for b in range(3) if a != b)
    assert max_off < 1e-12

    # Simultaneously g^(1)- and g^(2)-blind pair (|Xi_1(theta)>, |Xi_2(theta)>) for arbitrary theta.
    th = sp.Symbol("theta", real=True)
    _, idx3 = fock_basis(N, 3)
    c, s = sp.cos(th), sp.sin(th)

    def make_xi(sign: int) -> sp.Matrix:
        vec = sp.zeros(D3, 1)
        for k in range(4):
            vec[idx3[(3 - k, k, 0)], 0] = sp.expand(sp.sqrt(sp.binomial(3, k)) * (c ** (3 - k) * (-s) ** k + sign * s ** (3 - k) * c ** k) / sp.sqrt(2))
        return vec

    xi1, xi2 = make_xi(+1), make_xi(-1)
    for entry in S2.H * sp.Matrix(list(xi1 * xi1.H - xi2 * xi2.H)):
        assert sp.trigsimp(sp.expand(entry)) == 0
    for k in range(4):
        dp = sp.expand(xi1[idx3[(3 - k, k, 0)], 0] ** 2 - xi2[idx3[(3 - k, k, 0)], 0] ** 2)
        assert sp.trigsimp(dp - sp.Rational(1, 4) * sp.binomial(3, k) * ((-1) ** k) * sp.sin(2 * th) ** 3) == 0

    U_bc_s3 = lift_sym_k_sympy(sp.Matrix([[1, 0, 0], [0, 1 / sp.sqrt(2), 1 / sp.sqrt(2)], [0, 1 / sp.sqrt(2), -1 / sp.sqrt(2)]]), 3)
    xi1_3 = sp.simplify(U_bc_s3 * xi1.subs({sp.cos(th): sp.Rational(3, 5), sp.sin(th): sp.Rational(4, 5)}))
    xi2_3 = sp.simplify(U_bc_s3 * xi2.subs({sp.cos(th): sp.Rational(3, 5), sp.sin(th): sp.Rational(4, 5)}))
    assert S2.H * sp.Matrix(list(xi1_3 * xi1_3.H - xi2_3 * xi2_3.H)) == sp.zeros(36, 1)
    assert sp.simplify(sum(xi1_3[a, 0] ** 4 - xi2_3[a, 0] ** 4 for a in range(D3))) != 0

    return {
        "Nph3_dephasing_rates": [1, 2, 3, 4, 5, 6, 7, 9],
        "N3_filtration_dims": [9, 27, 64],
        "L3_intertwining_err": intertwine_err,
        "L3_max_off_block_err": max_off,
        "xi_g1_g2_blind_arbitrary_theta": True,
    }


def verify_bose_hubbard_witness():
    """Sec. VI.A, Eq. (16): [H_int, b_m^dag b_n] = U K_{mn} in V_cum and short-time leakage."""
    J, U, gamma, th = sp.symbols("J U gamma theta", real=True)
    for N in (2, 3):
        _, b2 = annihilation_sympy(N, 2)
        _, b1 = annihilation_sympy(N, 1)
        D2 = N * (N + 1) // 2
        E2 = {(m, n): b2[m].H * b2[n] for m in range(N) for n in range(N)}
        H_int = sum((sp.Rational(1, 2) * U * E2[(j, j)] * (E2[(j, j)] - sp.eye(D2)) for j in range(N)), sp.zeros(D2, D2))

        S_mat = sp.Matrix.hstack(*[sp.Matrix(list(E2[(m, n)])) for m in range(N) for n in range(N)])
        vec_I = sp.Matrix([1 if m == n else 0 for m in range(N) for n in range(N)])
        P_1b = S_mat * ((sp.eye(N * N) - vec_I * vec_I.T / (2 * (N + 1))) / (N + 2)) * S_mat.H
        Q_cum = sp.eye(D2 * D2) - P_1b

        for m in range(N):
            for n in range(N):
                K_mn = (b1[m] * b2[m]).H * (b1[m] * b2[n]) - (b1[m] * b2[n]).H * (b1[n] * b2[n])
                assert sp.simplify(comm(H_int, E2[(m, n)]) - U * K_mn) == sp.zeros(D2, D2)
                assert sp.simplify(P_1b * sp.Matrix(list(K_mn))) == sp.zeros(D2 * D2, 1)
                if m != n:
                    assert sp.simplify(sp.trace(K_mn.H * K_mn) - 4) == 0

        L_int = -sp.I * (sp.kronecker_product(H_int, sp.eye(D2)) - sp.kronecker_product(sp.eye(D2), H_int.T))
        assert sp.simplify(P_1b * L_int * P_1b) == sp.zeros(D2 * D2)
        assert sp.simplify(P_1b * L_int * Q_cum) != sp.zeros(D2 * D2)
        K_01 = (b1[0] * b2[0]).H * (b1[0] * b2[1]) - (b1[0] * b2[1]).H * (b1[1] * b2[1])
        assert sp.simplify(sp.trace(E2[(0, 1)] * (-sp.I * comm(H_int, K_01.H))) - 4 * sp.I * U) == 0

    # Short-time witness for weighted 3-port B2(theta) on K_3, dimer B1, and 4-site B2-CZ4.
    states2_3, b2_3 = annihilation_sympy(3, 2)
    _, idx2_3 = fock_basis(3, 2)
    E2_3 = {(m, n): b2_3[m].H * b2_3[n] for m in range(3) for n in range(3)}
    H_tot_3 = (
        -J * sum((E2_3[(m, n)] for m in range(3) for n in range(3) if m != n), sp.zeros(6, 6))
        + sum((sp.Rational(1, 2) * U * E2_3[(j, j)] * (E2_3[(j, j)] - sp.eye(6)) for j in range(3)), sp.zeros(6, 6))
    )
    M_d3 = sp.Matrix([[sp.Rational(1, 2) * gamma * sum((states2_3[a][j] - states2_3[b][j]) ** 2 for j in range(3)) for b in range(6)] for a in range(6)])
    L_U3 = lambda X: -sp.I * comm(H_tot_3, X) - sp.matrix_multiply_elementwise(M_d3, X)

    c, s = sp.cos(th), sp.sin(th)
    psi1 = sp.zeros(6, 1)
    psi1[idx2_3[(2, 0, 0)]] = 1 / sp.sqrt(2)
    psi1[idx2_3[(0, 2, 0)]] = c**2 / sp.sqrt(2)
    psi1[idx2_3[(0, 1, 1)]] = c * s
    psi1[idx2_3[(0, 0, 2)]] = s**2 / sp.sqrt(2)
    psi2 = sp.zeros(6, 1); psi2[idx2_3[(1, 1, 0)]] = c; psi2[idx2_3[(1, 0, 1)]] = s
    d1_th = L_U3(psi1 * psi1.H - psi2 * psi2.H)
    d2_th = L_U3(d1_th)
    assert sp.trigsimp(sp.trace(E2_3[(1, 2)] * d1_th) - sp.I * U * sp.sin(4 * th) / 4) == 0
    assert sp.trigsimp(sp.trace(E2_3[(1, 1)] * d2_th) + sp.Rational(1, 2) * U * J * sp.sin(4 * th)) == 0
    assert sp.simplify(sp.trace(E2_3[(0, 1)] * d2_th).subs(th, sp.pi / 4) + 6 * U * J) == 0

    # Doublon-free B2-CZ4 on 4-site chain: O(t) vanishes, O(U J t^2) turns on via virtual doublons.
    _, b2_4 = annihilation_sympy(4, 2)
    _, idx2_4 = fock_basis(4, 2)
    E2_4 = {(m, n): b2_4[m].H * b2_4[n] for m in range(4) for n in range(4)}
    Htot_4 = (
        -J * sum((E2_4[(k, k + 1)] + E2_4[(k + 1, k)] for k in range(3)), sp.zeros(10, 10))
        + sum((sp.Rational(1, 2) * U * E2_4[(j, j)] * (E2_4[(j, j)] - sp.eye(10)) for j in range(4)), sp.zeros(10, 10))
    )
    phi_adj = sp.zeros(10, 1); phi_adj[idx2_4[(1, 1, 0, 0)]] = phi_adj[idx2_4[(0, 0, 1, 1)]] = 1 / sp.sqrt(2)
    phi_cz4 = sp.zeros(10, 1)
    phi_cz4[idx2_4[(1, 0, 1, 0)]] = phi_cz4[idx2_4[(1, 0, 0, 1)]] = phi_cz4[idx2_4[(0, 1, 1, 0)]] = sp.Rational(1, 2)
    phi_cz4[idx2_4[(0, 1, 0, 1)]] = -sp.Rational(1, 2)
    d1_4 = -sp.I * comm(Htot_4, phi_adj * phi_adj.H - phi_cz4 * phi_cz4.H)
    d2_4 = -sp.I * comm(Htot_4, d1_4)
    assert all(sp.simplify(sp.trace(E2_4[(m, n)] * d1_4)) == 0 for m in range(4) for n in range(4))
    d2_D = sp.Matrix(4, 4, lambda m, n: sp.simplify(sp.trace(E2_4[(m, n)] * d2_4)))
    assert d2_D[0, 1] == 2 * U * J and d2_D[2, 3] == 2 * U * J and d2_D[1, 2] == -U * J

    return {
        "exact_BBGKY_commutator_verified": True,
        "K_mn_in_V_cum_norm_sq_4": True,
        "weighted_B2_d1_D12": "(i*U/4)*sin(4*theta)",
        "B2_CZ4_chain_d2_D_01": str(d2_D[0, 1]),
    }


def main():
    print(json.dumps({
        "Theorem_1_Orthogonal_Splitting": verify_theorem1_splitting(),
        "Theorem_2_Invisibility_and_Redfield": verify_theorem2_invisibility_and_redfield(),
        "Sec_VIa_Three_Photons": verify_three_photon_filtration(),
        "Eq16_Bose_Hubbard_BBGKY_Witness": verify_bose_hubbard_witness(),
    }, indent=2))


if __name__ == "__main__":
    main()
