"""Minimal SPM12 DEM closure for Lorenz: spm_DEM_M + spm_DEM_generate."""

from spm_dem.dem_m import spm_DEM_M
from spm_dem.dem_generate import spm_DEM_generate

__all__ = ["spm_DEM_M", "spm_DEM_generate"]
