"""spm_diff — numerical differentiation (spm_diff.m)."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, eye as speye_fn

from spm_dem.funcheck import spm_funcheck
from spm_dem.packing import spm_unvec, spm_vec
from spm_dem.packing import spm_length as _spm_length

GLOBAL_DX = np.exp(-8)


def spm_diff(*varargin):
    """First-order: spm_diff(f, x1, x2, ..., n) -> (J, f0)."""
    dx = GLOBAL_DX
    f = spm_funcheck(varargin[0])
    last = varargin[-1]
    if isinstance(last, (list, tuple)) and len(varargin) >= 3:
        x = list(varargin[1:-2])
        n = int(varargin[-2])
        V = list(last)
        q = True
    elif isinstance(last, (int, np.integer)):
        x = list(varargin[1:-1])
        n = int(last)
        V = [[] for _ in x]
        q = True
    elif isinstance(last, str):
        x = list(varargin[1:-2])
        n = int(varargin[-2])
        V = [[] for _ in x]
        q = False
    else:
        raise ValueError("Improper spm_diff call")

    m = n
    for i in range(len(x)):
        if (not V[i]) and (n == i + 1):
            li = _spm_length(x[i])
            V[i] = speye_fn(li, li, format="csr")
    xm = spm_vec(x[m - 1])
    Vm = V[m - 1]
    ncol = int(Vm.shape[1]) if sp.issparse(Vm) else int(np.asarray(Vm).shape[1])
    Jcells = []
    for i in range(ncol):
        xi = [np.array(a, copy=True) if isinstance(a, np.ndarray) else a for a in x]
        col = Vm[:, i].toarray().reshape(-1, 1) if sp.issparse(Vm) else np.asarray(Vm[:, i]).reshape(-1, 1)
        pert = xm + col * dx
        xi[m - 1] = spm_unvec(pert, x[m - 1])
        Jcells.append(_spm_dfdx(f(*xi), f(*x), dx))
    f0 = f(*x)
    fv = spm_vec(f0)
    if xm.size == 0:
        J = csr_matrix((max(len(fv), 1), 0))
    elif fv.size == 0:
        J = csr_matrix((0, xm.size))
    elif isinstance(f0, np.ndarray) and isinstance(Jcells, list) and q:
        J = _spm_dfdx_cat(Jcells)
    else:
        J = _spm_dfdx_cat(Jcells)
    return J, f0


def _spm_dfdx(f, f0, dx):
    if isinstance(f, (list, tuple)):
        return [_spm_dfdx(a, b, dx) for a, b in zip(f, f0)]
    if isinstance(f, dict):
        return (spm_vec(f) - spm_vec(f0)) / dx
    fa = f.toarray() if sp.issparse(f) else np.asarray(f, dtype=float)
    f0a = f0.toarray() if sp.issparse(f0) else np.asarray(f0, dtype=float)
    return (fa - f0a) / dx


def _spm_dfdx_cat(J):
    if not J:
        return csr_matrix((0, 0))
    j0 = np.asarray(J[0])
    if j0.ndim == 2 and (j0.shape[1] == 1 or j0.shape[0] == 1):
        if j0.shape[1] == 1:
            return sp.hstack([csr_matrix(np.atleast_2d(np.asarray(x))) for x in J], format="csr")
        return sp.vstack([csr_matrix(np.atleast_2d(np.asarray(x))) for x in J], format="csr")
    return sp.hstack([csr_matrix(np.atleast_2d(np.asarray(x))) for x in J], format="csr")
