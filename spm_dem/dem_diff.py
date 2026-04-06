"""spm_DEM_diff — hierarchical dynamics + Jacobians (spm_DEM_diff.m)."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix

from spm_dem.diff import spm_diff
from spm_dem.packing import spm_cat, spm_unvec, spm_vec


def _mul_or_zero(A, x):
    if sp.issparse(A) and A.shape[1] == 0:
        return csr_matrix((A.shape[0], 1))
    if A.shape[1] == 0:
        return np.zeros((A.shape[0], 1))
    return A @ x


def spm_DEM_diff(M, u):
    try:
        _ = M[0]["a"]
        ADEM = True
    except Exception:
        u["a"] = [sp.csr_matrix(c.shape) for c in u["v"]]
        ADEM = False

    nl = len(M)
    n = int(M[0]["E"]["n"]) + 1

    dgdvi = [[csr_matrix((0, 0)) for _ in range(nl)] for _ in range(nl)]
    dgdxi = [[csr_matrix((0, 0)) for _ in range(nl)] for _ in range(nl)]
    dgdai = [[csr_matrix((0, 0)) for _ in range(nl)] for _ in range(nl)]
    dfdvi = [[csr_matrix((0, 0)) for _ in range(nl)] for _ in range(nl)]
    dfdxi = [[csr_matrix((0, 0)) for _ in range(nl)] for _ in range(nl)]
    dfdai = [[csr_matrix((0, 0)) for _ in range(nl)] for _ in range(nl)]
    for i in range(nl):
        dgdvi[i][i] = csr_matrix((M[i]["l"], M[i]["l"]))
        dgdxi[i][i] = csr_matrix((M[i]["l"], M[i]["n"]))
        dgdai[i][i] = csr_matrix((M[i]["l"], M[i].get("k", 0)))
        dfdvi[i][i] = csr_matrix((M[i]["n"], M[i]["l"]))
        dfdxi[i][i] = csr_matrix((M[i]["n"], M[i]["n"]))
        dfdai[i][i] = csr_matrix((M[i]["n"], M[i].get("k", 0)))

    vi = list(spm_unvec(u["v"][0], [m["v"] for m in M]))
    xi = list(spm_unvec(u["x"][0], [m["x"] for m in M]))
    _ = spm_unvec(u["a"][0], [m.get("a", sp.csr_matrix((0, 1))) for m in M])
    zi = list(spm_unvec(u["z"][0], [m["v"] for m in M]))

    gi = []
    fi = []
    vi[nl - 1] = zi[nl - 1]

    for idx in range(nl - 2, -1, -1):
        i = idx
        if ADEM:
            raise NotImplementedError("ADEM path not in Lorenz closure")
        dgdx, g = spm_diff(M[i]["g"], xi[i], vi[i + 1], M[i]["pE"], 1)
        dfdx, f = spm_diff(M[i]["f"], xi[i], vi[i + 1], M[i]["pE"], 1)
        dgdv = spm_diff(M[i]["g"], xi[i], vi[i + 1], M[i]["pE"], 2)[0]
        dfdv = spm_diff(M[i]["f"], xi[i], vi[i + 1], M[i]["pE"], 2)[0]
        dgda = csr_matrix((M[i]["l"], 0))
        dfda = csr_matrix((M[i]["n"], 0))

        gi.append(g)
        fi.append(f)
        vi[i] = spm_vec(g) + spm_vec(zi[i])

        dgdxi[i][i] = dgdx if sp.issparse(dgdx) else csr_matrix(np.atleast_2d(dgdx))
        dgdvi[i][i + 1] = dgdv if sp.issparse(dgdv) else csr_matrix(np.atleast_2d(dgdv))
        dgdai[i][i + 1] = dgda
        dfdxi[i][i] = dfdx if sp.issparse(dfdx) else csr_matrix(np.atleast_2d(dfdx))
        dfdvi[i][i + 1] = dfdv if sp.issparse(dfdv) else csr_matrix(np.atleast_2d(dfdv))
        dfdai[i][i + 1] = dfda

    dg = {
        "da": spm_cat(dgdai),
        "dv": spm_cat(dgdvi),
        "dx": spm_cat(dgdxi),
    }
    df = {
        "da": spm_cat(dfdai),
        "dv": spm_cat(dfdvi),
        "dx": spm_cat(dfdxi),
    }

    u["v"][0] = spm_vec(vi)
    u["x"][1] = spm_vec(list(reversed(fi))) + u["w"][0]
    for ii in range(1, n - 1):
        u["v"][ii] = (
            _mul_or_zero(dg["dv"], u["v"][ii])
            + _mul_or_zero(dg["dx"], u["x"][ii])
            + _mul_or_zero(dg["da"], u["a"][ii])
            + u["z"][ii]
        )
        u["x"][ii + 1] = (
            _mul_or_zero(df["dv"], u["v"][ii])
            + _mul_or_zero(df["dx"], u["x"][ii])
            + _mul_or_zero(df["da"], u["a"][ii])
            + u["w"][ii]
        )
    return u, dg, df
