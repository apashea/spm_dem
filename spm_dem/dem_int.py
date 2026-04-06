"""spm_DEM_int — generalised-coordinate integration (spm_DEM_int.m)."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse import block_diag, csr_matrix
from scipy.sparse import kron as spkron

from spm_dem.dem_diff import spm_DEM_diff
from spm_dem.dem_embed import spm_DEM_embed
from spm_dem.dem_m_set import spm_DEM_M_set
from spm_dem.dx import spm_dx
from spm_dem.linalg_ext import spm_speye
from spm_dem.packing import spm_cat, spm_unvec, spm_vec


def _to_dense_col(c):
    if sp.issparse(c):
        return c.toarray()
    return np.asarray(c, dtype=float).reshape(-1, 1)


def _spm_vec_u(u):
    return np.vstack(
        [
            spm_vec([_to_dense_col(c) for c in u["v"]]),
            spm_vec([_to_dense_col(c) for c in u["x"]]),
            spm_vec([_to_dense_col(c) for c in u["z"]]),
            spm_vec([_to_dense_col(c) for c in u["w"]]),
        ]
    )


def _spm_unvec_u(bigv, u):
    n = len(u["v"])
    nv = u["v"][0].shape[0]
    nx = u["x"][0].shape[0]
    off = 0
    for _ in range(n):
        u["v"][_] = csr_matrix(bigv[off : off + nv])
        off += nv
    for _ in range(n):
        u["x"][_] = csr_matrix(bigv[off : off + nx])
        off += nx
    for _ in range(n):
        u["z"][_] = csr_matrix(bigv[off : off + nv])
        off += nv
    for _ in range(n):
        u["w"][_] = csr_matrix(bigv[off : off + nx])
        off += nx
    return u


def _vstack_csr(mats):
    rows = []
    for m in mats:
        a = m.toarray() if sp.issparse(m) else np.asarray(m, dtype=float)
        rows.append(a)
    return csr_matrix(np.vstack(rows))


def spm_DEM_int(M, z, w, c):
    M = spm_DEM_M_set(M)
    zmat = _vstack_csr(z)
    cmat = _vstack_csr(c)
    zmat = zmat + cmat
    wmat = _vstack_csr(w)

    nt = zmat.shape[1]
    nl = len(M)
    nv = int(np.sum(spm_vec(np.array([[m["l"]] for m in M], dtype=float))))
    nx = int(np.sum(spm_vec(np.array([[m["n"]] for m in M], dtype=float))))

    dt = float(M[0]["E"]["dt"])
    n = int(M[0]["E"]["n"]) + 1
    nD = int(M[0]["E"]["nD"])
    td = dt / nD

    u = {
        "v": [sp.csr_matrix((nv, 1)) for _ in range(n)],
        "x": [sp.csr_matrix((nx, 1)) for _ in range(n)],
        "z": [sp.csr_matrix((nv, 1)) for _ in range(n)],
        "w": [sp.csr_matrix((nx, 1)) for _ in range(n)],
        "a": [sp.csr_matrix((nv, 1)) for _ in range(n)],
    }

    vi = [m["v"] for m in M]
    xi = [m["x"] for m in M]
    u["v"][0] = csr_matrix(spm_vec(vi))
    u["x"][0] = csr_matrix(spm_vec(xi))

    Dx = spkron(spm_speye(n, n, 1), spm_speye(nx, nx, 0))
    Dv = spkron(spm_speye(n, n, 1), spm_speye(nv, nv, 0))
    D = block_diag((Dv, Dx, Dv, Dx), format="csr")
    dfdw = np.kron(np.eye(n), np.eye(nx))

    V = [sp.csr_matrix((M[i]["l"], nt)) for i in range(nl)]
    X = [sp.csr_matrix((M[i]["n"], nt)) for i in range(nl)]
    Z = [sp.csr_matrix((M[i]["l"], nt)) for i in range(nl)]
    W = [sp.csr_matrix((M[i]["n"], nt)) for i in range(nl)]

    mnx = any(len(m.get("pg", [])) > 0 for m in M)
    mnv = any(len(m.get("ph", [])) > 0 for m in M)
    Sz = 1
    Sw = 1

    for t in range(1, nt + 1):
        for iD in range(1, nD + 1):
            ts = (t + (iD - 1) / nD) * dt
            if mnx or mnv:
                raise NotImplementedError("state-dependent precision not in Lorenz closure")

            uz = spm_DEM_embed(Sz * zmat, n, ts, dt)
            uw = spm_DEM_embed(Sw * wmat, n, ts, dt)
            u["z"] = uz
            u["w"] = uw

            u, dg, df = spm_DEM_diff(M, u)

            dgdv = spkron(spm_speye(n, n, 1), dg["dv"])
            dgdx = spkron(spm_speye(n, n, 1), dg["dx"])
            dfdv = spkron(spm_speye(n, n, 0), df["dv"])
            dfdx = spkron(spm_speye(n, n, 0), df["dx"])

            def _flat(u0):
                if sp.issparse(u0):
                    return u0.toarray()
                return np.asarray(u0, dtype=float)

            vi = spm_unvec(_flat(u["v"][0]), [m["v"] for m in M])
            xi = spm_unvec(_flat(u["x"][0]), [m["x"] for m in M])
            zi = spm_unvec(_flat(u["z"][0]), [m["v"] for m in M])
            wi = spm_unvec(_flat(u["w"][0]), [m["x"] for m in M])
            if iD == 1:
                for i in range(nl):
                    Vi = V[i].tolil()
                    Xi = X[i].tolil()
                    Zi = Z[i].tolil()
                    Wi = W[i].tolil()
                    if M[i]["l"]:
                        Vi[:, t - 1] = spm_vec([vi[i]]).ravel()
                    if M[i]["n"]:
                        Xi[:, t - 1] = spm_vec([xi[i]]).ravel()
                    if M[i]["l"]:
                        Zi[:, t - 1] = spm_vec([zi[i]]).ravel()
                    if M[i]["n"]:
                        Wi[:, t - 1] = spm_vec([wi[i]]).ravel()
                    V[i] = Vi.tocsr()
                    X[i] = Xi.tocsr()
                    Z[i] = Zi.tocsr()
                    W[i] = Wi.tocsr()

            if nt == 1:
                break

            J = spm_cat(
                [
                    [dgdv, dgdx, Dv, None],
                    [dfdv, dfdx, None, csr_matrix(dfdw)],
                    [None, None, Dv, None],
                    [None, None, None, Dx],
                ]
            )
            vu = _spm_vec_u(u)
            du = spm_dx(J, D @ vu, td)
            u = _spm_unvec_u(vu + du, u)

        if nt == 1:
            break

    return V, X, Z, W
