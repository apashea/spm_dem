"""spm_vec, spm_unvec, spm_length, spm_cat — MATLAB column-major semantics."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def _as_dense_col_vec(X):
    if sp.issparse(X):
        return np.asarray(X.todense()).reshape(-1, 1, order="F")
    return np.asarray(X, dtype=float).reshape(-1, 1, order="F")


def spm_vec(X, *varargin):
    """Vectorise numeric / sparse / cell / struct-like dict (sorted field names)."""
    if varargin:
        X = [X] + list(varargin)
    if X is None:
        return np.zeros((0, 1))
    if isinstance(X, dict):
        vX = np.zeros((0, 1))
        for k in sorted(X.keys()):
            vX = np.vstack((vX, spm_vec(X[k])))
        return vX
    if isinstance(X, (list, tuple)):
        vX = np.zeros((0, 1))
        for item in X:
            vX = np.vstack((vX, spm_vec(item)))
        return vX
    if sp.issparse(X) or isinstance(X, np.ndarray):
        return _as_dense_col_vec(X)
    if isinstance(X, (int, float, np.floating, np.integer)):
        return np.array([[float(X)]])
    raise TypeError(f"spm_vec: unsupported type {type(X)}")


def spm_length(X) -> int:
    if X is None:
        return 0
    if isinstance(X, dict):
        return sum(spm_length(X[k]) for k in sorted(X.keys()))
    if isinstance(X, (list, tuple)):
        return sum(spm_length(x) for x in X)
    if sp.issparse(X) or isinstance(X, np.ndarray):
        return int(np.asarray(X).size)
    if isinstance(X, (int, float, np.floating, np.integer)):
        return 1
    return 0


def _to_sparse_block(a, shape):
    r, c = shape
    if a is None:
        return sp.csr_matrix((r, c))
    if isinstance(a, (int, float)) and a == 0:
        return sp.csr_matrix((r, c))
    if sp.issparse(a):
        aa = a.tocsr()
        if aa.shape != shape:
            out = sp.csr_matrix(shape)
            out[: aa.shape[0], : aa.shape[1]] = aa
            return out
        return aa
    arr = np.atleast_2d(np.asarray(a, dtype=float))
    if arr.shape != shape:
        out = np.zeros(shape)
        out[: arr.shape[0], : arr.shape[1]] = arr
        arr = out
    return sp.csr_matrix(arr)


def spm_cat(x, d=None):
    """Convert nested list-of-lists cell layout to block matrix (spm_cat.m)."""
    if not isinstance(x, (list, tuple)):
        return x
    rows = x
    if d is not None:
        n = len(rows)
        m = len(rows[0]) if n else 0
        if d == 1:
            return [spm_cat([rows[i][j] for i in range(n)]) for j in range(m)]
        if d == 2:
            return [spm_cat(list(rows[i])) for i in range(n)]
        raise ValueError("unknown option")

    n = len(rows)
    m = len(rows[0]) if n else 0
    work = [[None] * m for _ in range(n)]
    I = np.zeros((n, m), dtype=int)
    J = np.zeros((n, m), dtype=int)
    for i in range(n):
        for j in range(m):
            cell = rows[i][j]
            if isinstance(cell, (list, tuple)):
                cell = spm_cat(cell)
            work[i][j] = cell
            if cell is None:
                continue
            if isinstance(cell, (int, float)) and cell == 0:
                I[i, j], J[i, j] = 1, 1
            elif sp.issparse(cell):
                I[i, j], J[i, j] = cell.shape
            else:
                a = np.asarray(cell)
                I[i, j], J[i, j] = a.shape if a.ndim == 2 else (a.shape[0], 1)
    Irow = np.max(I, axis=1)
    Jcol = np.max(J, axis=0)
    for i in range(n):
        for j in range(m):
            cell = work[i][j]
            if cell is None or (
                isinstance(cell, (list, np.ndarray)) and np.asarray(cell).size == 0
            ):
                work[i][j] = sp.csr_matrix((int(Irow[i]), int(Jcol[j])))
            elif isinstance(cell, (int, float)) and cell == 0:
                work[i][j] = sp.csr_matrix((int(Irow[i]), int(Jcol[j])))
            elif sp.issparse(cell) and cell.nnz == 0 and cell.shape == (0, 0):
                work[i][j] = sp.csr_matrix((int(Irow[i]), int(Jcol[j])))
    row_blocks = []
    for i in range(n):
        cols = []
        for j in range(m):
            cols.append(work[i][j])
        row_blocks.append(sp.hstack(cols, format="csr"))
    return sp.vstack(row_blocks, format="csr")


def spm_unvec(vX, tmpl):
    """Unvectorise using template (ndarray, sparse, dict, or list)."""
    if sp.issparse(vX):
        vX = vX.toarray()
    vX = np.asarray(vX, dtype=float).reshape(-1, 1)
    if sp.issparse(tmpl):
        r, c = tmpl.shape
        need = r * c
        flat = vX[:need].ravel(order="F")
        return sp.csr_matrix(flat.reshape((r, c), order="F"))
    if isinstance(tmpl, np.ndarray):
        out = np.array(tmpl, dtype=float, copy=True)
        flat = vX[: out.size].ravel(order="F")
        out[:] = flat.reshape(out.shape, order="F")
        return out
    if isinstance(tmpl, dict):
        out = {}
        off = 0
        for k in sorted(tmpl.keys()):
            sub = tmpl[k]
            n = spm_length(sub)
            out[k] = spm_unvec(vX[off : off + n], sub)
            off += n
        return out
    if isinstance(tmpl, list):
        out = []
        off = 0
        for sub in tmpl:
            n = spm_length(sub)
            out.append(spm_unvec(vX[off : off + n], sub))
            off += n
        return out
    return tmpl
