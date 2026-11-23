"""Shared Fock-space operators, Lindblad/Kraus propagators, metameric states, and graph models (Sec. II–III).

Spatial modes are indexed 0..N-1 in Python; optical port labels in the paper are 1-based (m + 1).
Two-boson states live in Sym^2(C^N) of dimension D_2 = N(N+1)/2 ordered lexicographically as m <= n.
"""
from __future__ import annotations

import itertools
import os
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def sym2_basis(N: int):
    """Lexicographic basis of Sym^2(C^N) and local occupation numbers n_{j,alpha}."""
    pairs = [(m, n) for m in range(N) for n in range(m, N)]
    idx_map = {}
    n_ops = np.zeros((N, len(pairs)), dtype=np.float64)
    for k, (m, n) in enumerate(pairs):
        idx_map[(m, n)] = idx_map[(n, m)] = k
        n_ops[m, k] += 1.0
        n_ops[n, k] += 1.0
    return pairs, idx_map, n_ops


def dephasing_matrix(n_ops: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Pure-dephasing damping matrix M_{ab} = (1/2) sum_j gamma_j (n_{j,a} - n_{j,b})^2."""
    M = np.zeros((n_ops.shape[1], n_ops.shape[1]), dtype=np.float64)
    for j, gj in enumerate(gamma):
        if gj != 0.0:
            diff = n_ops[j][:, None] - n_ops[j][None, :]
            M += 0.5 * gj * (diff ** 2)
    return M


def annihilation_sym2(N: int, pairs: list[tuple[int, int]]) -> list[np.ndarray]:
    """Annihilation operators b_m : Sym^2(C^N) -> C^N."""
    D2 = len(pairs)
    B = [np.zeros((N, D2), dtype=np.complex128) for _ in range(N)]
    for k, (m, n) in enumerate(pairs):
        if m == n:
            B[m][m, k] = np.sqrt(2.0)
        else:
            B[m][n, k] = 1.0
            B[n][m, k] = 1.0
    return B


def two_boson_model(
    adj: np.ndarray,
    J: float,
    gamma: np.ndarray,
    eps: np.ndarray | None = None,
    kappa: np.ndarray | None = None,
    U: float = 0.0,
):
    """Symmetric two-boson generator on Sym^2(C^N): H = -J sum A_{ij} b_i^dag b_j + sum eps_j n_j
    + (U/2) sum n_j(n_j - 1), optional non-Hermitian loss -i sum kappa_j n_j, and L_j = sqrt(gamma_j) n_j.
    """
    N = adj.shape[0]
    pairs, idx_map, n_ops = sym2_basis(N)
    B_ann = annihilation_sym2(N, pairs)
    D2 = len(pairs)

    H1 = -J * adj.astype(np.complex128)
    if eps is not None:
        H1 = H1 + np.diag(eps.astype(np.complex128))

    H2 = np.zeros((D2, D2), dtype=np.complex128)
    for i in range(N):
        for j in range(N):
            if H1[i, j] != 0.0:
                H2 += H1[i, j] * (B_ann[i].conj().T @ B_ann[j])
    if U != 0.0:
        H2 += np.diag((0.5 * U * np.sum(n_ops * (n_ops - 1.0), axis=0)).astype(np.complex128))

    H1_eff = H1.copy()
    H2_eff = H2.copy()
    if kappa is not None:
        H1_eff = H1_eff - 1j * np.diag(kappa.astype(np.complex128))
        H2_eff = H2_eff - 1j * np.diag((kappa @ n_ops).astype(np.complex128))

    M2 = dephasing_matrix(n_ops, gamma)
    M1 = 0.5 * (gamma[:, None] + gamma[None, :])
    np.fill_diagonal(M1, 0.0)
    return H1, H1_eff, M1, H2, H2_eff, M2, n_ops, pairs, idx_map, B_ann


def antisym_model(adj: np.ndarray, J: float, gamma: np.ndarray, kappa: np.ndarray | None = None):
    """Antisymmetric two-photon sector Lambda^2(C^N) of dimension D_- = N(N-1)/2 (Sec. V.B)."""
    N = adj.shape[0]
    apairs = [(m, n) for m in range(N) for n in range(m + 1, N)]
    aidx = {p: k for k, p in enumerate(apairs)}
    Da = len(apairs)

    H1 = -J * adj.astype(np.complex128)
    n_ops = np.zeros((N, Da), dtype=np.float64)
    for k, (m, n) in enumerate(apairs):
        n_ops[m, k] = n_ops[n, k] = 1.0

    Ha = np.zeros((Da, Da), dtype=np.complex128)
    for k, (m, n) in enumerate(apairs):
        Ha[k, k] += H1[m, m] + H1[n, n]
        for p in range(N):
            if p != n and H1[p, m] != 0.0:
                Ha[aidx[(p, n) if p < n else (n, p)], k] += H1[p, m] if p < n else -H1[p, m]
            if p != m and H1[p, n] != 0.0:
                Ha[aidx[(m, p) if m < p else (p, m)], k] += H1[p, n] if m < p else -H1[p, n]

    Ha_eff = Ha.copy()
    if kappa is not None:
        Ha_eff = Ha_eff - 1j * np.diag((kappa @ n_ops).astype(np.complex128))
    return Ha, Ha_eff, dephasing_matrix(n_ops, gamma), n_ops, apairs, aidx


def k_boson_model(adj: np.ndarray, J: float, gamma: np.ndarray, k_photons: int):
    """Symmetric k-boson generator on Sym^k(C^N), dimension D_k = binom(N + k - 1, k)."""
    N = adj.shape[0]
    tuples_k = list(itertools.combinations_with_replacement(range(N), k_photons))
    Dk = len(tuples_k)
    n_ops = np.zeros((N, Dk), dtype=np.int32)
    occ_to_idx = {}
    for idx, tup in enumerate(tuples_k):
        for m in tup:
            n_ops[m, idx] += 1
        occ_to_idx[tuple(n_ops[:, idx])] = idx

    H1 = -J * adj.astype(np.complex128)
    Hk = np.zeros((Dk, Dk), dtype=np.complex128)
    for idx in range(Dk):
        occ = n_ops[:, idx]
        for j in range(N):
            if occ[j] == 0:
                continue
            for i in range(N):
                if H1[i, j] == 0.0:
                    continue
                if i == j:
                    Hk[idx, idx] += H1[i, i] * occ[i]
                else:
                    new_occ = list(occ)
                    new_occ[j] -= 1
                    new_occ[i] += 1
                    Hk[occ_to_idx[tuple(new_occ)], idx] += H1[i, j] * np.sqrt(float(occ[j] * (occ[i] + 1)))

    return Hk, dephasing_matrix(n_ops.astype(np.float64), gamma), n_ops, tuples_k, occ_to_idx


def clements_layer(N: int, layer_idx: int, theta: float) -> np.ndarray:
    """Single even/odd Clements MZI layer with symmetric beam-splitter angle theta (Sec. II.A)."""
    U = np.eye(N, dtype=np.complex128)
    c, s = np.cos(theta), 1j * np.sin(theta)
    for m in range(layer_idx % 2, N - 1, 2):
        U[m, m] = U[m + 1, m + 1] = c
        U[m, m + 1] = U[m + 1, m] = s
    return U


def lift_sym2(U1: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    """Second-quantized unitary Sym^2(U1) on Sym^2(C^N) via two-mode permanents."""
    p = np.array([a for a, _ in pairs], dtype=np.int32)
    q = np.array([b for _, b in pairs], dtype=np.int32)
    norms = np.where(p == q, np.sqrt(2.0), 1.0)
    U_mp = U1[p[:, None], p[None, :]]
    U_nq = U1[q[:, None], q[None, :]]
    U_mq = U1[p[:, None], q[None, :]]
    U_np = U1[q[:, None], p[None, :]]
    return (U_mp * U_nq + U_mq * U_np) / (norms[:, None] * norms[None, :])


def rk4(H: np.ndarray, M_deph: np.ndarray, rho0: np.ndarray, dz: float, n_steps: int, hermitian: bool = True) -> np.ndarray:
    """4th-order Runge-Kutta integration of drho/dz = -i(H rho - rho H^dag) - M_deph (*) rho."""
    H_dag = H if hermitian else H.conj().T

    def rhs(r):
        return -1j * (H @ r - r @ H_dag) - M_deph * r

    traj = np.empty((n_steps + 1,) + rho0.shape, dtype=np.complex128)
    traj[0] = rho0
    r = rho0.copy()
    for s in range(1, n_steps + 1):
        k1 = rhs(r)
        k2 = rhs(r + 0.5 * dz * k1)
        k3 = rhs(r + 0.5 * dz * k2)
        k4 = rhs(r + dz * k3)
        r = r + (dz / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        traj[s] = r
    return traj


def hbt_observables(traj: np.ndarray, n_ops: np.ndarray, pairs: list[tuple[int, int]], N: int):
    """Mean intensities I_m = <n_m> and HBT coincidence matrix Gamma_{qr} = <b_q^dag b_r^dag b_r b_q>."""
    pops = np.real(np.diagonal(traj, axis1=1, axis2=2))
    intensities = pops @ n_ops.T
    G = np.zeros((pops.shape[0], N, N), dtype=np.float64)
    for k, (m, n) in enumerate(pairs):
        if m == n:
            G[:, m, m] += 2.0 * pops[:, k]
        else:
            G[:, m, n] += pops[:, k]
            G[:, n, m] += pops[:, k]
    return intensities, G


def hbt_antisym(traj: np.ndarray, n_ops_a: np.ndarray, apairs: list[tuple[int, int]], N: int):
    """Intensities and coincidences on Lambda^2(C^N) (Gamma_{qq} = 0 by Pauli exclusion)."""
    pops = np.real(np.diagonal(traj, axis1=1, axis2=2))
    intensities = pops @ n_ops_a.T
    G = np.zeros((pops.shape[0], N, N), dtype=np.float64)
    for k, (m, n) in enumerate(apairs):
        G[:, m, n] += pops[:, k]
        G[:, n, m] += pops[:, k]
    return intensities, G


def rdm_from_rho(rho: np.ndarray, B_ann: list[np.ndarray]) -> np.ndarray:
    """First-order coherence matrix D_{mn} = Tr(b_m^dag b_n rho) on Sym^2(C^N)."""
    N = len(B_ann)
    D = np.zeros((N, N), dtype=np.complex128)
    for m in range(N):
        Bm_dag = B_ann[m].conj().T
        for n in range(N):
            D[m, n] = np.sum((Bm_dag @ B_ann[n]) * rho.T)
    return D


def rdm_diff_shot(U: np.ndarray, W1: np.ndarray, W2: np.ndarray) -> float:
    """Single-realization ||D^(1,w) - D^(2,w)||_F for pure symmetric states: D = 2 (W W^dag)^*."""
    W1_out = U @ W1 @ U.T
    W2_out = U @ W2 @ U.T
    D1 = 2.0 * (W1_out @ W1_out.conj().T).conj()
    D2 = 2.0 * (W2_out @ W2_out.conj().T).conj()
    return float(np.linalg.norm(D1 - D2, "fro"))


def gamma_ss(N: int) -> np.ndarray:
    """Uniform maximally mixed steady-state coincidence matrix Gamma^(inf)_{qr} = 2(1 + delta_{qr}) / (N(N+1))."""
    G = np.full((N, N), 2.0 / (N * (N + 1)), dtype=np.float64)
    np.fill_diagonal(G, 4.0 / (N * (N + 1)))
    return G


def state_b1(N: int, a: int, b: int, noon_sign: float = -1.0):
    """Two-port metameric pair (|Psi_HOM>, |Psi_NOON^+/->) on modes (a, b) (Sec. II.C)."""
    W_hom = np.zeros((N, N), dtype=np.complex128)
    W_hom[a, b] = W_hom[b, a] = 1.0 / np.sqrt(2.0)
    W_noon = np.zeros((N, N), dtype=np.complex128)
    W_noon[a, a] = 1.0 / np.sqrt(2.0)
    W_noon[b, b] = noon_sign / np.sqrt(2.0)
    return W_hom, W_noon


def state_b2(N: int, a: int, b: int, c: int, sign: float = 1.0):
    """Three-port metameric pair (|Psi_1^+/->, |Psi_2>) on modes (a, b, c) (Sec. II.C)."""
    W1 = np.zeros((N, N), dtype=np.complex128)
    W1[a, a] = 1.0 / np.sqrt(2.0)
    v = sign * 0.5 / np.sqrt(2.0)
    W1[b, b] = W1[c, c] = W1[b, c] = W1[c, b] = v

    W2 = np.zeros((N, N), dtype=np.complex128)
    W2[a, b] = W2[b, a] = 0.5
    W2[a, c] = W2[c, a] = 0.5
    return W1, W2


def state_cz4(N: int, a: int, b: int, c: int, d: int):
    """Doublon-free four-port metameric pair (|Phi_adj>, |Phi_CZ4>) on modes (a, b, c, d) (Sec. II.C)."""
    W_adj = np.zeros((N, N), dtype=np.complex128)
    W_adj[a, b] = W_adj[b, a] = 0.5
    W_adj[c, d] = W_adj[d, c] = 0.5

    W_cz4 = np.zeros((N, N), dtype=np.complex128)
    for p, q, sgn in ((a, c, 1.0), (a, d, 1.0), (b, c, 1.0), (b, d, -1.0)):
        W_cz4[p, q] = W_cz4[q, p] = sgn / (2.0 * np.sqrt(2.0))
    return W_adj, W_cz4


def wf_to_fock(W: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    """Normalized Fock amplitudes psi_{(m,n)} = W_{mm} (m=n) or sqrt(2) W_{mn} (m<n)."""
    psi = np.zeros(len(pairs), dtype=np.complex128)
    for k, (m, n) in enumerate(pairs):
        psi[k] = W[m, m] if m == n else np.sqrt(2.0) * W[m, n]
    return psi


def graph_chain(N: int) -> np.ndarray:
    """Open 1D nearest-neighbor chain of N sites."""
    adj = np.zeros((N, N), dtype=np.float64)
    for i in range(N - 1):
        adj[i, i + 1] = adj[i + 1, i] = 1.0
    return adj


def graph_small_world(N: int) -> np.ndarray:
    """Deterministic chorded ring: unit ring plus chords of weight 0.85 (Sec. III.C)."""
    adj = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        j = (i + 1) % N
        adj[i, j] = adj[j, i] = 1.0
    if N == 8:
        adj[0, 3] = adj[3, 0] = 0.85
        adj[2, 6] = adj[6, 2] = 0.85
    else:
        for c in range(0, N // 2, 3):
            t = (c + N // 2 - 1) % N
            adj[c, t] = adj[t, c] = 0.85
    return adj


def graph_modular_bridge(N: int, J_bridge: float = 0.15) -> np.ndarray:
    """Two-community modular bottleneck graph: two N/2-site chorded rings joined by J_bridge (Sec. III.C)."""
    M = N // 2
    adj = np.zeros((N, N), dtype=np.float64)
    for base in (0, M):
        for i in range(M):
            j = (i + 1) % M
            adj[base + i, base + j] = adj[base + j, base + i] = 1.0
        if M == 8:
            adj[base + 0, base + 4] = adj[base + 4, base + 0] = 0.8
            adj[base + 2, base + 6] = adj[base + 6, base + 2] = 0.8
        elif M == 4:
            adj[base + 0, base + 2] = adj[base + 2, base + 0] = 0.8
    adj[M - 1, M] = adj[M, M - 1] = J_bridge
    return adj


def crossing_stats(d1: np.ndarray, d2: np.ndarray, grid: np.ndarray, tol: float = 1e-7):
    """Crossing coordinates and inversion gap where the initially farther trajectory overtakes the closer one."""
    if d1[0] >= d2[0]:
        d_far, d_close, order = d1, d2, "1_overtakes_2"
    else:
        d_far, d_close, order = d2, d1, "2_overtakes_1"

    diff = d_far - d_close
    if diff[0] <= tol:
        return None

    crossings = []
    for k in range(len(grid) - 1):
        if diff[k] * diff[k + 1] < 0.0 or (diff[k] == 0.0 and k > 0):
            denom = diff[k + 1] - diff[k]
            z_star = grid[k] - diff[k] * (grid[k + 1] - grid[k]) / denom if abs(denom) > 1e-15 else grid[k]
            crossings.append(round(float(z_star), 6))

    min_diff = float(np.min(diff))
    if not crossings or min_diff >= -tol:
        return None
    idx_min = int(np.argmin(diff))
    step = float(grid[1] - grid[0]) if len(grid) > 1 else 1.0
    inv_mask = diff < 0.0
    return {
        "order": order,
        "d0_far": round(float(d_far[0]), 6),
        "d0_close": round(float(d_close[0]), 6),
        "first_crossing": crossings[0],
        "n_crossings": len(crossings),
        "all_crossings": crossings[:10],
        "max_inversion_gap": round(float(-min_diff), 6),
        "rel_inversion_pct": round(100.0 * float(-min_diff) / max(float(d_close[idx_min]), 1e-12), 3),
        "coord_at_max_gap": round(float(grid[idx_min]), 5),
        "inverted_window": round(float(np.sum(inv_mask) * step), 4),
        "inverted_area": round(float(np.sum(-diff[inv_mask]) * step), 6),
    }
