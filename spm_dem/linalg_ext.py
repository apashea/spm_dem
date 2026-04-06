"""spm_speye, spm_en, spm_svd, spm_inv, spm_pinv, spm_sqrtm."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse import diags, csr_matrix, eye as speye_fn


def spm_speye(m, n=None, k=0, c=0, o=1):
    """Sparse matrix with ones on k-th diagonal (spm_speye.m)."""
    if n is None:
        n = m
    if c != 0:
        raise NotImplementedError("spm_speye c!=0 not needed for Lorenz closure")
    D = diags(np.ones(m), k, shape=(m, n), format="csr")
    if m == n and o != 1:
        D = D**o
    return D


def spm_en(X, p=None):
    """Column Euclidean normalisation (spm_en.m, single-arg path)."""
    if p is not None:
        raise NotImplementedError("spm_en with detrend not used in Lorenz closure")
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        X = X.reshape(-1, 1)
    out = np.zeros_like(X)
    for i in range(X.shape[1]):
        col = X[:, i]
        if np.any(col):
            out[:, i] = col / np.sqrt(np.sum(col**2))
    return out


def spm_svd(X, Uthresh=1e-6):
    """Truncated SVD — dense path matching spm_svd.m intent for small blocks."""
    if Uthresh >= 1:
        Uthresh = Uthresh - 1e-6
    if Uthresh <= 0:
        Uthresh = 64 * np.finfo(float).eps
    if sp.issparse(X):
        Xf = X.toarray()
    else:
        Xf = np.asarray(X, dtype=float)
    M, N = Xf.shape
    if M == 0 or N == 0:
        return csr_matrix((M, 0)), csr_matrix((0, 0)), csr_matrix((N, 0))
    uu, s, vv = np.linalg.svd(Xf, full_matrices=False)
    s2 = s**2
    if s2.size == 0:
        return csr_matrix((M, 0)), csr_matrix((0, 0)), csr_matrix((N, 0))
    jk = np.flatnonzero(s2 * len(s2) / (np.sum(s2) + 1e-300) > Uthresh)
    if jk.size == 0:
        return csr_matrix((M, 0)), csr_matrix((0, 0)), csr_matrix((N, 0))
    uu = uu[:, jk]
    s = s[jk]
    vv = vv[jk, :]
    S = diags(s, 0, format="csr")
    return csr_matrix(uu), S, csr_matrix(vv.T)


def spm_inv(A, TOL=None):
    """Regularised inverse (spm_inv.m)."""
    A = A.toarray() if sp.issparse(A) else np.asarray(A, dtype=float)
    m, n = A.shape
    if A.size == 0:
        return sp.csr_matrix((n, m))
    if TOL is None:
        TOL = max(np.finfo(float).eps * np.linalg.norm(A, ord=np.inf) * max(m, n), np.exp(-32))
    return np.linalg.inv(A + np.eye(m, n) * TOL)


def spm_pinv(A, TOL=None):
    """Sparse pseudoinverse (spm_pinv.m)."""
    if sp.issparse(A):
        Af = A.toarray()
        m, n = A.shape
    else:
        Af = np.asarray(A, dtype=float)
        m, n = Af.shape
    if Af.size == 0:
        return sp.csr_matrix((n, m))
    if TOL is None:
        try:
            AtA = Af.T @ Af
            X = spm_inv(AtA)
            if np.all(np.isfinite(X)):
                return csr_matrix(X @ Af.T)
        except Exception:
            pass
    U, S, V = spm_svd(Af, 0)
    Sd = np.asarray(S.diagonal())
    if TOL is None:
        TOL = max(m, n) * np.finfo(float).eps * (np.max(np.abs(Sd)) if Sd.size else 0)
    r = int(np.sum(np.abs(Sd) > TOL))
    if r == 0:
        return sp.csr_matrix((n, m))
    idx = np.arange(r)
    Sinv = diags(1.0 / Sd[idx], 0, shape=(r, r), format="csr")
    return V[:, idx] @ Sinv @ U[:, idx].T


def spm_sqrtm(V):
    """Matrix square root via SVD (spm_sqrtm.m)."""
    if sp.issparse(V):
        Vf = V.toarray()
    else:
        Vf = np.asarray(V, dtype=float)
    u, s, _ = spm_svd(Vf, 0)
    sd = np.sqrt(np.abs(np.asarray(s.diagonal())))
    m = len(sd)
    smat = diags(sd, 0, shape=(m, m), format="csr")
    return u @ smat @ u.T
