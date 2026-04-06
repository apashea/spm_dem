"""spm_DEM_embed — temporal embedding (spm_DEM_embed.m)."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def _y_submatrix(Y, rows, cols):
    """Y[rows][:, cols] as dense (rows slice or int, cols 0-based indices)."""
    if sp.issparse(Y):
        Y = Y.tocsr()
        if isinstance(rows, int):
            mat = Y[rows, cols].toarray()
            return mat.reshape(1, -1)
        return Y[rows, :][:, cols].toarray()
    if isinstance(rows, int):
        return np.asarray(Y[rows, cols], dtype=float).reshape(1, -1)
    return np.asarray(Y[rows, :][:, cols], dtype=float)


def spm_DEM_embed(Y, n, t, dt=None, d=0):
    if dt is None:
        dt = 1.0
    if sp.issparse(Y):
        q, N = Y.shape
    else:
        Y = np.asarray(Y, dtype=float)
        q, N = Y.shape
    n = int(n)
    y = [sp.csr_matrix((q, 1), dtype=float) for _ in range(n)]
    if q == 0:
        return y
    d = np.atleast_1d(np.asarray(d, dtype=float)).ravel()
    for p in range(len(d)):
        s = (t - d[p]) / dt
        k = np.arange(1, n + 1) + int(np.floor(s - (n + 1) / 2))
        x = s - np.min(k) + 1
        kk = k.astype(float)
        kk[kk < 1] = 1
        kk[kk > N] = N
        k = kk.astype(int)
        T = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                jm = j + 1
                if jm == 1:
                    den = 1.0
                else:
                    den = float(np.prod(np.arange(1, jm)))
                T[i, j] = (((i + 1) - x) * dt) ** (jm - 1) / den
        E = np.linalg.inv(T)
        cidx = k - 1
        if len(d) == q:
            for i in range(n):
                row = _y_submatrix(Y, p, cidx)
                val = row.ravel() @ E[i, :]
                y[i] = y[i].tolil()
                y[i][p, 0] = val
                y[i] = y[i].tocsr()
        else:
            for i in range(n):
                blk = _y_submatrix(Y, slice(None), cidx)
                yi = blk @ E[i, :].reshape(n, 1)
                y[i] = sp.csr_matrix(yi)
            return y
    return y
