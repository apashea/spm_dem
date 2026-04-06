"""
Assert output shapes match MATLAB R2024b for:

    rng(1)
    M = spm_DEM_M('Lorenz')
    N = 1024
    U = sparse(1, N)
    DEM = spm_DEM_generate(M, U)

Run: python verify_dims.py
"""

import numpy as np
import scipy.sparse as sp

from spm_dem import spm_DEM_M, spm_DEM_generate


def main():
    np.random.seed(1)
    M = spm_DEM_M("Lorenz")
    assert len(M) == 2
    assert (M[0]["l"], M[0]["n"], M[0]["m"]) == (1, 3, 1)
    assert (M[1]["l"], M[1]["n"], M[1]["m"]) == (1, 0, 0)
    N = 1024
    U = sp.csr_matrix((1, N))
    DEM = spm_DEM_generate(M, U)
    assert DEM["Y"].shape == (1, N)
    assert DEM["pU"]["v"][0].shape == (1, N)
    assert DEM["pU"]["v"][1].shape == (1, N)
    assert DEM["pU"]["x"][0].shape == (3, N)
    assert DEM["pU"]["x"][1].shape == (0, N)
    print("verify_dims: OK")


if __name__ == "__main__":
    main()
