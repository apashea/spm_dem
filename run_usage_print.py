"""Full printout for Lorenz DEM usage (compare with MATLAB)."""

import numpy as np
import scipy.sparse as sp

np.set_printoptions(precision=8, suppress=True, linewidth=200)

from spm_dem import spm_DEM_M, spm_DEM_generate

print("=== PYTHON ===")
np.random.seed(1)
M = spm_DEM_M("Lorenz")
print("numel(M) =", len(M))
for i in range(len(M)):
    mi = M[i]
    print(
        f"M({i + 1}).l={mi['l']} M({i + 1}).n={mi['n']} M({i + 1}).m={mi['m']}"
    )
    pe = np.asarray(mi["pE"])
    xx = mi["x"]
    if sp.issparse(xx):
        xs = xx.shape
    else:
        xs = np.asarray(xx).shape
    print("  pE shape", pe.shape, "x shape", xs)
N = 1024
U = sp.csr_matrix((1, N))
print("U shape", U.shape, "nnz", U.nnz)
DEM = spm_DEM_generate(M, U)
Y = DEM["Y"]
print("DEM.Y shape", Y.shape, "type", type(Y).__name__, "nnz", Y.nnz)
print("DEM.Y first 10 cols (full):", Y[:, :10].toarray().ravel())
print("DEM.Y cols 512:517:", Y[:, 512:517].toarray().ravel())
print("DEM.Y last 5 cols:", Y[:, -5:].toarray().ravel())
for name in ("v", "x"):
    pu = DEM["pU"][name]
    print(f"DEM.pU.{name} len={len(pu)} shapes:", [pu[j].shape for j in range(len(pu))])
print("pU.v{1} first 5:", DEM["pU"]["v"][0][:, :5].toarray().ravel())
print("pU.v{2} first 5:", DEM["pU"]["v"][1][:, :5].toarray().ravel())
print("pU.x{1} first col (3x1):", DEM["pU"]["x"][0][:, 0].toarray().ravel())
print("pU.x{1} col 100:", DEM["pU"]["x"][0][:, 99].toarray().ravel())
P0 = np.asarray(DEM["pP"]["P"][0]).ravel()
print("DEM.pP.P[0] (level1 pE) flat[:5]:", P0[: min(5, len(P0))])
h0 = np.asarray(DEM["pH"]["h"][0]).ravel()
print("DEM.pH.h[0] first 3:", h0[: min(3, len(h0))])
