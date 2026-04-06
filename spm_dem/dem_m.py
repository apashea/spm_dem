"""spm_DEM_M — model templates (spm_DEM_M.m), Lorenz only."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from spm_dem.dem_m_set import spm_DEM_M_set

LORENZ_F_STR = "[-P(1) P(1) 0; (P(3)-x(3)) -1 -x(1); x(2) x(1) P(2)]*x/32;"


def _g_lorenz_sum(x, v, P):
    from spm_dem.funcheck import _as_dense_col

    x = _as_dense_col(x)
    return np.array([[float(np.sum(x))]], dtype=float)


def spm_DEM_M(model, *varargin):
    m = str(model).lower().strip()
    if m == "lorenz":
        M = [
            {
                "E": {"linear": 3, "s": 1.0 / 8.0},
                "f": LORENZ_F_STR,
                "g": _g_lorenz_sum,
                "x": np.array([[0.9], [0.8], [30.0]], dtype=float),
                "pE": np.array([[18.0], [-4.0], [46.92]], dtype=float),
                "V": float(np.exp(0)),
                "W": float(np.exp(16)),
            },
            {
                "v": np.array([[0.0]], dtype=float),
                "V": float(np.exp(16)),
                "f": None,
                "pE": sp.csr_matrix((0, 0)),
            },
        ]
    else:
        raise ValueError(f"unknown model; please add to spm_DEM_M: {model!r}")
    return spm_DEM_M_set(M)
