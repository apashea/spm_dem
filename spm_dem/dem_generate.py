"""spm_DEM_generate — data generation (spm_DEM_generate.m)."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix

from spm_dem.dem_int import spm_DEM_int
from spm_dem.dem_m_set import spm_DEM_M_set
from spm_dem.dem_z import spm_DEM_z
from spm_dem.packing import spm_unvec, spm_vec


def spm_DEM_generate(M, U, P=None, h=None, g_hyp=None):
    M = spm_DEM_M_set([dict(m) for m in M])
    DEM = {"M": deepcopy(M)}

    if sp.issparse(U) or isinstance(U, np.ndarray):
        if max(U.shape) > 1:
            N = U.shape[1]
        else:
            N = int(np.asarray(U).ravel()[0])
            U = csr_matrix((M[-1]["l"], N))
    else:
        N = int(U)
        U = csr_matrix((M[-1]["l"], N))

    m = len(M)
    if P is None:
        P = [M[i]["pE"] for i in range(m)]
    elif not isinstance(P, (list, tuple)):
        P = [P]
    if h is None:
        h = []
    if g_hyp is None:
        g_hyp = []

    for i in range(m):
        try:
            Pi = P[i]
            M[i]["pE"] = spm_unvec(spm_vec(Pi), M[i]["pE"])
        except Exception:
            try:
                M[i]["pE"] = P[i]
            except Exception:
                pass

    for i in range(m):
        try:
            M[i]["hE"] = np.asarray(h[i], dtype=float).reshape(-1, 1)
        except Exception:
            he = np.asarray(M[i]["hE"], dtype=float)
            M[i]["hE"] = (he - he) + 32.0

    for i in range(m):
        try:
            M[i]["gE"] = np.asarray(g_hyp[i], dtype=float).reshape(-1, 1)
        except Exception:
            ge = np.asarray(M[i]["gE"], dtype=float)
            M[i]["gE"] = (ge - ge) + 32.0

    M = spm_DEM_M_set(M)
    DEM["G"] = deepcopy(M)
    z, w = spm_DEM_z(M, N)

    u = [csr_matrix((M[i]["l"], N)) for i in range(m)]
    u[-1] = U if sp.issparse(U) else csr_matrix(U)

    v, x, ztraj, wtraj = spm_DEM_int(M, z, w, u)

    DEM["Y"] = v[0]
    DEM["pU"] = {"v": v, "x": x, "z": ztraj, "w": wtraj}
    DEM["pP"] = {"P": [m["pE"] for m in M]}
    DEM["pH"] = {"h": [m["hE"] for m in M], "g": [m["gE"] for m in M]}
    return DEM
