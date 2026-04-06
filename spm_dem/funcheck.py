"""spm_funcheck — normalize callable / string to a Python callable."""

from __future__ import annotations

import functools
from collections.abc import Callable


def spm_funcheck(f):
    """Convert MATLAB-like function spec to f(*args)."""
    if f is None:
        return f
    if callable(f):
        return f
    if isinstance(f, str):
        s = f.strip()
        if not s:
            return f
        # Lorenz dynamics string from spm_DEM_M (evaluated as inline in MATLAB)
        if "P(1)" in s and "*x/32" in s:
            return _lorenz_f_from_string()
        raise ValueError(f"spm_funcheck: unsupported string function: {s[:80]}...")
    raise TypeError(f"spm_funcheck: unsupported type {type(f)}")


def _as_dense_col(a):
    import numpy as np
    import scipy.sparse as sp

    if sp.issparse(a):
        a = a.toarray()
    return np.asarray(a, dtype=float).reshape(-1, 1)


def _lorenz_f_from_string() -> Callable:
    """dx/dt for Lorenz template (same as MATLAB char f)."""

    def f(x, v, P):
        import numpy as np

        x = _as_dense_col(x)
        P = _as_dense_col(P)
        A = np.array(
            [
                [-P[0, 0], P[0, 0], 0.0],
                [(P[2, 0] - x[2, 0]), -1.0, -x[0, 0]],
                [x[1, 0], x[0, 0], P[1, 0]],
            ],
            dtype=float,
        )
        return (A @ x) / 32.0

    return f


def fcnchk(f, *_names):
    """MATLAB fcnchk shim — return callable (names ignored in Python)."""
    return spm_funcheck(f)
