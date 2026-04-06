"""spm_DEM_z — innovations (spm_DEM_z.m)."""

from __future__ import annotations

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm as spnorm
from scipy.stats import norm

from spm_dem.linalg_ext import spm_sqrtm


def _randn_norminv_uniform(rows: int, cols: int) -> np.ndarray:
    """Standard normal draws from the current NumPy uniform stream.

    Uniforms are taken in the same linear order as MATLAB ``rand(rows, cols)``
    (column-major), then transformed like ``norminv(u)``. SPM12 ``spm_DEM_z``
    uses ``randn`` (ziggurat); for bit-for-bit parity with this Python path,
    replace those ``randn`` calls with ``norminv(rand(...))`` in MATLAB.
    """
    u = np.random.random(rows * cols).reshape((rows, cols), order="F")
    return norm.ppf(u)


def spm_DEM_z(M, N):
    s = float(M[0]["E"]["s"]) + float(np.exp(-16))
    dt = float(M[0]["E"]["dt"])
    t = (np.arange(1, N + 1) - 1) * dt
    col0 = np.exp(-(t**2) / (2 * s**2))
    K = la.toeplitz(col0)
    d = np.sqrt(np.maximum(np.diag(K @ K.T), 1e-300))
    K = K @ np.diag(1.0 / d)

    zlist = []
    wlist = []
    for i in range(len(M)):
        P = M[i]["V"]
        if not sp.issparse(P):
            P = csr_matrix(np.asarray(P, dtype=float))
        try:
            for j in range(len(M[i]["Q"])):
                P = P + M[i]["Q"][j] * float(np.exp(M[i]["hE"][j, 0]))
        except Exception:
            pass
        n1 = spnorm(P, 1) if sp.issparse(P) else np.linalg.norm(P, ord=1)
        if n1 == 0:
            zi = csr_matrix(_randn_norminv_uniform(M[i]["l"], N) @ K)
        elif n1 >= np.exp(16):
            zi = sp.csr_matrix((M[i]["l"], N))
        else:
            Pd = P.toarray() if sp.issparse(P) else np.asarray(P, dtype=float)
            Pi = np.linalg.inv(Pd)
            zi = csr_matrix(spm_sqrtm(Pi) @ _randn_norminv_uniform(M[i]["l"], N) @ K)

        zlist.append(zi)

        Pw = M[i]["W"]
        if not sp.issparse(Pw):
            Pw = csr_matrix(np.asarray(Pw, dtype=float))
        if Pw.shape[0] == 0 or Pw.shape[1] == 0:
            wlist.append(sp.csr_matrix((M[i]["n"], N)))
            continue
        try:
            for j in range(len(M[i]["R"])):
                Pw = Pw + M[i]["R"][j] * float(np.exp(M[i]["gE"][j, 0]))
        except Exception:
            pass
        if Pw.shape[0] == 0 and Pw.shape[1] == 0:
            wi = sp.csr_matrix((0, 0))
        else:
            n2 = spnorm(Pw, 1) if sp.issparse(Pw) else np.linalg.norm(Pw, ord=1)
            if n2 == 0:
                wi = csr_matrix(_randn_norminv_uniform(M[i]["n"], N) @ K * dt)
            elif n2 >= np.exp(16):
                wi = sp.csr_matrix((M[i]["n"], N))
            else:
                Pwd = Pw.toarray() if sp.issparse(Pw) else np.asarray(Pw, dtype=float)
                Pwi = np.linalg.inv(Pwd)
                wi = csr_matrix(spm_sqrtm(Pwi) @ _randn_norminv_uniform(M[i]["n"], N) @ K * dt)
        wlist.append(wi)
    return zlist, wlist
