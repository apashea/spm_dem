"""spm_DEM_M_set — model normalization (spm_DEM_M_set.m)."""

from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, diags, eye as speye_fn

from spm_dem.funcheck import fcnchk
from spm_dem.packing import spm_vec


def _deal(M, key, value):
    for m in M:
        m[key] = value


def _all_fields(M, key):
    return all(key in m for m in M)


def _f_zero(x, v, P):
    return sp.csr_matrix((0, 1))


def spm_DEM_M_set(M):
    M = [dict(m) for m in M]
    g = len(M)

    if _all_fields(M, "f") and not _all_fields(M, "n") and not _all_fields(M, "x"):
        pass  # msgbox omitted

    try:
        fcnchk(M[g - 1]["g"])
        g += 1
        new = {
            "l": int(M[g - 2]["m"]),
            "m": 0,
            "n": 0,
        }
        M.append(new)
    except Exception:
        pass
    M[g - 1]["m"] = 0
    M[g - 1]["n"] = 0

    if not _all_fields(M, "f"):
        _deal(M, "f", _f_zero)
        _deal(M, "x", sp.csr_matrix((0, 1)))
        _deal(M, "n", 0)

    for i in range(g):
        if M[i].get("f") is None:
            M[i]["f"] = _f_zero
            M[i]["x"] = sp.csr_matrix((0, 1))
            M[i]["n"] = 0
            continue
        try:
            M[i]["f"] = fcnchk(M[i]["f"], "x", "v", "P")
        except Exception:
            M[i]["f"] = _f_zero
            M[i]["x"] = sp.csr_matrix((0, 1))
            M[i]["n"] = 0

    for i in range(g):
        if "pE" not in M[i]:
            M[i]["pE"] = sp.csr_matrix((0, 0))

    for i in range(g):
        if "pC" not in M[i]:
            np_ = spm_vec(M[i]["pE"]).size
            M[i]["pC"] = sp.csr_matrix((np_, np_))

    for i in range(g):
        np_ = spm_vec(M[i]["pE"]).size
        pC = M[i]["pC"]
        if pC is None or (sp.issparse(pC) and pC.shape == (0, 0) and np_ > 0):
            M[i]["pC"] = sp.csr_matrix((np_, np_))
            pC = M[i]["pC"]
        pC = M[i]["pC"]
        if np.isscalar(pC) or (isinstance(pC, (int, float))):
            M[i]["pC"] = speye_fn(np_, np_, format="csr") * float(pC)
        elif hasattr(pC, "shape") and len(pC.shape) == 2:
            if pC.shape[0] == 1 or pC.shape[1] == 1:
                v = spm_vec(pC).ravel()
                M[i]["pC"] = diags(v, 0, format="csr", shape=(np_, np_))
        maxdim = max(M[i]["pC"].shape) if sp.issparse(M[i]["pC"]) else max(M[i]["pC"].shape)
        if maxdim != np_:
            raise ValueError(f"please check: M({i+1}).pC")

    try:
        v = M[g - 1]["v"]
    except KeyError:
        v = sp.csr_matrix((0, 0))
    if v is None or (sp.issparse(v) and v.nnz == 0 and v.shape == (0, 0)):
        try:
            v = sp.csr_matrix((M[g - 2]["m"], 1))
        except Exception:
            v = sp.csr_matrix((0, 0))
    if v is None or (sp.issparse(v) and v.nnz == 0 and v.shape == (0, 0)):
        try:
            v = sp.csr_matrix((M[g - 1]["l"], 1))
        except Exception:
            v = sp.csr_matrix((0, 0))
    M[g - 1]["l"] = spm_vec(v).size
    M[g - 1]["v"] = v

    for idx in range(g - 2, -1, -1):
        i = idx
        try:
            x = M[i]["x"]
        except KeyError:
            x = sp.csr_matrix((M[i]["n"], 1))
        if (x is None or (sp.issparse(x) and x.nnz == 0)) and M[i]["n"]:
            x = sp.csr_matrix((M[i]["n"], 1))
        try:
            M[i]["f"] = fcnchk(M[i]["f"], "x", "v", "P")
        except Exception:
            pass
        try:
            fv = M[i]["f"](x, v, M[i]["pE"])
            if spm_vec(x).size != spm_vec(fv).size:
                raise ValueError(f"please check: M({i+1}).f(x,v,P)")
        except Exception as e:
            raise RuntimeError(f"evaluation failure: M({i+1}).f(x,v,P): {e}") from e
        try:
            M[i]["g"] = fcnchk(M[i]["g"], "x", "v", "P")
        except Exception:
            pass
        try:
            M[i]["m"] = spm_vec(v).size
            v = M[i]["g"](x, v, M[i]["pE"])
            M[i]["l"] = spm_vec(v).size
            M[i]["n"] = spm_vec(x).size
            M[i]["v"] = v
            M[i]["x"] = x
        except Exception as e:
            raise RuntimeError(f"evaluation failure: M({i+1}).g(x,v,P): {e}") from e

    if not _all_fields(M, "xP"):
        M[0]["xP"] = []
    if not _all_fields(M, "vP"):
        M[0]["vP"] = []
    for i in range(g):
        xp = M[i].get("xP", [])
        if isinstance(xp, np.ndarray) and xp.ndim == 1:
            M[i]["xP"] = diags(xp, 0, shape=(M[i]["n"], M[i]["n"]), format="csr")
        elif xp is None or (isinstance(xp, list) and len(xp) == 0) or (
            sp.issparse(xp) and xp.nnz == 0 and max(xp.shape) == 0
        ):
            M[i]["xP"] = sp.csr_matrix((M[i]["n"], M[i]["n"]))
        elif sp.issparse(xp) or isinstance(xp, np.ndarray):
            if max(xp.shape) != M[i]["n"]:
                try:
                    s0 = float(xp[0, 0]) if sp.issparse(xp) else float(np.asarray(xp).ravel()[0])
                    M[i]["xP"] = speye_fn(M[i]["n"], M[i]["n"], format="csr") * s0
                except Exception:
                    M[i]["xP"] = sp.csr_matrix((M[i]["n"], M[i]["n"]))
        vp = M[i].get("vP", [])
        if isinstance(vp, np.ndarray) and vp.ndim == 1:
            M[i]["vP"] = diags(vp, 0, shape=(M[i]["l"], M[i]["l"]), format="csr")
        elif vp is None or (isinstance(vp, list) and len(vp) == 0) or (
            sp.issparse(vp) and vp.nnz == 0 and max(vp.shape) == 0
        ):
            M[i]["vP"] = sp.csr_matrix((M[i]["l"], M[i]["l"]))
        elif sp.issparse(vp) or isinstance(vp, np.ndarray):
            if max(vp.shape) != M[i]["l"]:
                try:
                    s0 = float(vp[0, 0]) if sp.issparse(vp) else float(np.asarray(vp).ravel()[0])
                    M[i]["vP"] = speye_fn(M[i]["l"], M[i]["l"], format="csr") * s0
                except Exception:
                    M[i]["vP"] = sp.csr_matrix((M[i]["l"], M[i]["l"]))

    nx = int(np.sum(spm_vec(np.array([[m["n"]] for m in M], dtype=float))))

    for i in range(g):
        M[i].setdefault("Q", [])
        M[i].setdefault("R", [])
        M[i].setdefault("V", [])
        M[i].setdefault("W", [])
        M[i].setdefault("hE", [])
        M[i].setdefault("gE", [])
        M[i].setdefault("ph", [])
        M[i].setdefault("pg", [])

    pP = 1.0
    for i in range(g):
        if M[i]["Q"] is not None and M[i]["Q"] != [] and not isinstance(M[i]["Q"], (list, tuple)):
            M[i]["Q"] = [M[i]["Q"]]
        if M[i]["R"] is not None and M[i]["R"] != [] and not isinstance(M[i]["R"], (list, tuple)):
            M[i]["R"] = [M[i]["R"]]
        hE = np.asarray(spm_vec(M[i]["hE"]) if M[i].get("hE") is not None else np.zeros((0, 1)))
        gE = np.asarray(spm_vec(M[i]["gE"]) if M[i].get("gE") is not None else np.zeros((0, 1)))
        if hE.size == 0:
            hE = np.zeros((len(M[i]["Q"]), 1))
        if gE.size == 0:
            gE = np.zeros((len(M[i]["R"]), 1))
        M[i]["hE"] = hE
        M[i]["gE"] = gE
        try:
            _ = M[i]["hC"] @ M[i]["hE"]
        except Exception:
            M[i]["hC"] = speye_fn(len(M[i]["hE"]), len(M[i]["hE"]), format="csr") / pP
        try:
            _ = M[i]["gC"] @ M[i]["gE"]
        except Exception:
            M[i]["gC"] = speye_fn(len(M[i]["gE"]), len(M[i]["gE"]), format="csr") / pP
        if M[i]["hC"] is None or (sp.issparse(M[i]["hC"]) and M[i]["hC"].nnz == 0):
            M[i]["hC"] = speye_fn(len(M[i]["hE"]), len(M[i]["hE"]), format="csr") / pP
        if M[i]["gC"] is None or (sp.issparse(M[i]["gC"]) and M[i]["gC"].nnz == 0):
            M[i]["gC"] = speye_fn(len(M[i]["gE"]), len(M[i]["gE"]), format="csr") / pP

        if len(M[i]["Q"]) > len(M[i]["hE"]):
            M[i]["hE"] = np.zeros((len(M[i]["Q"]), 1)) + float(M[i]["hE"].flat[0] if M[i]["hE"].size else 0)
        if len(M[i]["Q"]) < len(M[i]["hE"]):
            M[i]["Q"] = [speye_fn(M[i]["l"], M[i]["l"], format="csr")]
            M[i]["hE"] = np.array([[float(M[i]["hE"].flat[0])]])
        if len(M[i]["hE"]) > M[i]["hC"].shape[0]:
            M[i]["hC"] = speye_fn(len(M[i]["Q"]), len(M[i]["Q"]), format="csr") * float(M[i]["hC"][0, 0])
        if len(M[i]["R"]) > len(M[i]["gE"]):
            M[i]["gE"] = np.zeros((len(M[i]["R"]), 1)) + float(M[i]["gE"].flat[0] if M[i]["gE"].size else 0)
        if len(M[i]["R"]) < len(M[i]["gE"]):
            M[i]["R"] = [speye_fn(M[i]["n"], M[i]["n"], format="csr")]
            M[i]["gE"] = np.array([[float(M[i]["gE"].flat[0])]])
        if len(M[i]["gE"]) > M[i]["gC"].shape[0]:
            M[i]["gC"] = speye_fn(len(M[i]["R"]), len(M[i]["R"]), format="csr") * float(M[i]["gC"][0, 0])

        for j, Qj in enumerate(M[i]["Q"]):
            if max(Qj.shape) != M[i]["l"]:
                raise ValueError(f"wrong size; M({i+1}).Q{{{j+1}}}")
        for j, Rj in enumerate(M[i]["R"]):
            if max(Rj.shape) != M[i]["n"]:
                raise ValueError(f"wrong size; M({i+1}).R{{{j+1}}}")

        V = M[i]["V"]
        if isinstance(V, np.ndarray) and V.ndim == 1:
            M[i]["V"] = diags(V, 0, format="csr")
            V = M[i]["V"]
        if sp.issparse(V):
            md = max(V.shape)
        else:
            md = max(np.asarray(V).shape) if hasattr(V, "shape") else M[i]["l"]
        if md != M[i]["l"]:
            try:
                s0 = float(V[0, 0]) if sp.issparse(V) else float(np.asarray(V).ravel()[0])
                M[i]["V"] = speye_fn(M[i]["l"], M[i]["l"], format="csr") * s0
            except Exception:
                hEe = M[i]["hE"]
                ph = M[i].get("ph", [])
                if (hEe is None or hEe.size == 0) and (not ph or (isinstance(ph, list) and len(ph) == 0)):
                    M[i]["V"] = speye_fn(M[i]["l"], M[i]["l"], format="csr")
                else:
                    M[i]["V"] = sp.csr_matrix((M[i]["l"], M[i]["l"]))

        W = M[i]["W"]
        if isinstance(W, np.ndarray) and W.ndim == 1:
            M[i]["W"] = diags(W, 0, format="csr")
            W = M[i]["W"]
        if sp.issparse(W):
            mdw = max(W.shape)
        else:
            mdw = max(np.asarray(W).shape) if hasattr(W, "shape") else M[i]["n"]
        if mdw != M[i]["n"]:
            try:
                s0 = float(W[0, 0]) if sp.issparse(W) else float(np.asarray(W).ravel()[0])
                M[i]["W"] = speye_fn(M[i]["n"], M[i]["n"], format="csr") * s0
            except Exception:
                gEe = M[i]["gE"]
                pg = M[i].get("pg", [])
                if (gEe is None or gEe.size == 0) and (not pg or (isinstance(pg, list) and len(pg) == 0)):
                    M[i]["W"] = speye_fn(M[i]["n"], M[i]["n"], format="csr")
                else:
                    M[i]["W"] = sp.csr_matrix((M[i]["n"], M[i]["n"]))

    try:
        _ = M[0]["E"]["s"]
    except Exception:
        M[0].setdefault("E", {})
        M[0]["E"]["s"] = 0.5 if nx else 0.0
    try:
        _ = M[0]["E"]["dt"]
    except Exception:
        M[0].setdefault("E", {})
        M[0]["E"]["dt"] = 1.0
    try:
        _ = M[0]["E"]["d"]
    except Exception:
        M[0].setdefault("E", {})
        M[0]["E"]["d"] = 2 if nx else 0
    try:
        _ = M[0]["E"]["n"]
    except Exception:
        M[0].setdefault("E", {})
        M[0]["E"]["n"] = 6 if nx else 0
    M[0]["E"]["d"] = min(M[0]["E"]["d"], M[0]["E"]["n"])
    try:
        _ = M[0]["E"]["nD"]
    except Exception:
        M[0]["E"]["nD"] = 1 if nx else 8
    try:
        _ = M[0]["E"]["nE"]
    except Exception:
        M[0]["E"]["nE"] = 8
    try:
        _ = M[0]["E"]["nM"]
    except Exception:
        M[0]["E"]["nM"] = 8
    try:
        _ = M[0]["E"]["nN"]
    except Exception:
        M[0]["E"]["nN"] = 8

    for i in range(g):
        for key in ("sv", "sw"):
            if key in M[i]:
                del M[i][key]
    for i in range(g):
        if "sv" not in M[i]:
            M[i]["sv"] = M[0]["E"]["s"]
        if "sw" not in M[i]:
            M[i]["sw"] = M[0]["E"]["s"]
        if not np.isscalar(M[i]["sv"]):
            M[i]["sv"] = M[0]["E"]["s"]
        if not np.isscalar(M[i]["sw"]):
            M[i]["sw"] = M[0]["E"]["s"]

    try:
        _ = M[0]["E"]["linear"]
    except Exception:
        M[0]["E"]["linear"] = 0

    from scipy.sparse.linalg import norm as spnorm

    Vend = M[g - 1]["V"]
    if not sp.issparse(Vend):
        Vend = csr_matrix(np.atleast_2d(np.asarray(Vend, dtype=float)))
    try:
        Qflag = spnorm(Vend, 1) == 0
    except Exception:
        Qflag = float(np.linalg.norm(Vend.toarray(), ord=1)) == 0
    for i in range(g - 1):
        pC = M[i]["pC"]
        P = (spnorm(pC, 1) if sp.issparse(pC) else np.linalg.norm(np.asarray(pC), ord=1)) > np.exp(8)
        if P and Qflag:
            warnings.warn("please use informative priors on causes or parameters")

    return M
