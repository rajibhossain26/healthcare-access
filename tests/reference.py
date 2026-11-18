"""Slow, loop-based reference implementations written directly from the
published equations, kept deliberately separate from the vectorised code in
``healthaccess.models``. The tests compare the two on many random problems.

Notation (Luo & Wang 2003; Luo & Qi 2009; Wan et al. 2012; Delamater 2013):
P_i population of demand location i, S_j capacity of provider j,
d_ij travel cost, W_ij = f(d_ij) the distance-decay weight (0 outside the
catchment or when unreachable).
"""

import math


def weights(cost_rows, f):
    return [[0.0 if (c is None or math.isnan(c) or math.isinf(c)) else f(c) for c in row] for row in cost_rows]


def ref_2sfca(P, S, W):
    """R_j = S_j / sum_k P_k W_kj ;  A_i = sum_j R_j W_ij."""
    n, m = len(P), len(S)
    R = []
    for j in range(m):
        denom = sum(P[k] * W[k][j] for k in range(n))
        R.append(S[j] / denom if denom > 0 else 0.0)
    return [sum(R[j] * W[i][j] for j in range(m)) for i in range(n)]


def ref_3sfca(P, S, W):
    """G_ij = W_ij / sum_l W_il ;  R_j = S_j / sum_k G_kj P_k W_kj ;
    A_i = sum_j G_ij R_j W_ij  (Wan, Zou & Sternberg 2012)."""
    n, m = len(P), len(S)
    G = []
    for i in range(n):
        tot = sum(W[i])
        G.append([W[i][j] / tot if tot > 0 else 0.0 for j in range(m)])
    R = []
    for j in range(m):
        denom = sum(G[k][j] * P[k] * W[k][j] for k in range(n))
        R.append(S[j] / denom if denom > 0 else 0.0)
    return [sum(G[i][j] * R[j] * W[i][j] for j in range(m)) for i in range(n)]


def ref_m2sfca(P, S, W):
    """R_j = S_j / sum_k P_k W_kj ;  A_i = sum_j R_j W_ij^2  (Delamater 2013)."""
    n, m = len(P), len(S)
    R = []
    for j in range(m):
        denom = sum(P[k] * W[k][j] for k in range(n))
        R.append(S[j] / denom if denom > 0 else 0.0)
    return [sum(R[j] * W[i][j] ** 2 for j in range(m)) for i in range(n)]


# Decay functions, written from the papers.

def binary(d0):
    return lambda d: 1.0 if d <= d0 else 0.0


def e2sfca_zones(d):
    """Luo & Qi (2009): 0-10 min 1.00, 10-20 min 0.68, 20-30 min 0.22."""
    if d <= 10:
        return 1.0
    if d <= 20:
        return 0.68
    if d <= 30:
        return 0.22
    return 0.0


def dai_gaussian(d0):
    """Dai (2010): (exp(-1/2 (d/d0)^2) - exp(-1/2)) / (1 - exp(-1/2)) for d <= d0."""
    e = math.exp(-0.5)
    return lambda d: (math.exp(-0.5 * (d / d0) ** 2) - e) / (1 - e) if d <= d0 else 0.0


def gravity(beta, min_cost=1.0):
    """Joseph & Bantock (1982): d^-beta, with d floored at min_cost."""
    return lambda d: max(d, min_cost) ** (-beta)
