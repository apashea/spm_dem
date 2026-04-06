"""spm_dx — local linearised update (spm_dx.m)."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.linalg import expm

from spm_dem.linalg_ext import spm_pinv
from spm_dem.packing import spm_unvec, spm_vec


def spm_dx(dfdx, f, t=None, _unused=None):
    if t is None:
        t = np.inf
    xf = f
    fvec = spm_vec(f)
    n = len(fvec)
    ts = np.atleast_1d(t)
    if np.min(ts) > np.exp(16):
        dxv = -spm_pinv(dfdx) @ fvec
    else:
        if ts.ndim == 1 and ts.size == n:
            tmat = np.diag(ts.flatten())
        elif ts.size == 1:
            tmat = np.eye(n) * float(ts.ravel()[0])
        else:
            tmat = np.asarray(ts, dtype=float)
            if tmat.ndim == 1:
                tmat = np.diag(tmat)
        A = dfdx.toarray() if sp.issparse(dfdx) else np.asarray(dfdx, dtype=float)
        J = np.zeros((n + 1, n + 1))
        tf = (tmat @ fvec.reshape(-1, 1)).ravel()
        J[1:, 0] = tf
        J[1:, 1:] = tmat @ A
        E = expm(J)
        dxv = E[1:, 0:1]
    return spm_unvec(np.real(dxv), xf)
