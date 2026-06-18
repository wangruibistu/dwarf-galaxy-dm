#!/usr/bin/env python3
"""Download the Tolstoy/Battaglia VLT-FLAMES + Gaia DR3 Sculptor catalogue
(J/A+A/675/A49, "A 3D view of dwarf galaxies ... I. Sculptor"), select members
with clean Ca II-triplet [Fe/H] and line-of-sight velocities, and write a
processed member catalogue for the multi-population Jeans analysis.

This is the data upgrade over the public Walker+2009 Mg-index subsample (148
stars): ~1500 members with clean CaT metallicities, comparable to the sample
that the original split-population analyses used.
"""
from __future__ import annotations
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "processed" / "sculptor_flames_members.parquet"
RAW = ROOT / "data" / "external" / "sculptor_flames2023_raw.parquet"
RAW.parent.mkdir(parents=True, exist_ok=True); OUT.parent.mkdir(parents=True, exist_ok=True)

# Sculptor centre and distance (McConnachie 2012)
RA0, DE0, DIST_KPC = 15.0183, -33.7186, 86.0


def main():
    if RAW.exists():
        df = pd.read_parquet(RAW)
        print(f"[raw] {len(df)} rows (cached) <- {RAW}")
    else:
        from astroquery.vizier import Vizier
        V = Vizier(columns=["**"], row_limit=-1)
        df = V.get_catalogs("J/A+A/675/A49")["J/A+A/675/A49/tablee1"].to_pandas()
        df.to_parquet(RAW)                               # archive raw first
        print(f"[raw] {len(df)} rows -> {RAW}")

    feh = pd.to_numeric(df["[Fe/H]"], errors="coerce")
    efeh = pd.to_numeric(df["e_[Fe/H]"], errors="coerce")
    vlos = pd.to_numeric(df["Vlos"], errors="coerce")
    evlos = pd.to_numeric(df["e_Vlos"], errors="coerce")
    ra = pd.to_numeric(df["RAJ2000"], errors="coerce").values
    de = pd.to_numeric(df["DEJ2000"], errors="coerce").values
    dra = (ra - RA0) * np.cos(np.radians(DE0)); dde = de - DE0
    R_pc = np.radians(np.sqrt(dra ** 2 + dde ** 2)) * DIST_KPC * 1000.0

    mem = df["Mem"].astype(str).str.strip() == "m"
    # [Fe/H] > 0 is non-physical for Sculptor members (CaT calibration outliers)
    good = mem & feh.notna() & vlos.notna() & (feh < 0.0)
    n_pos = int((mem & feh.notna() & vlos.notna() & (feh >= 0.0)).sum())
    print(f"[cut] removed {n_pos} members with non-physical [Fe/H] >= 0")
    out = pd.DataFrame({
        "R_pc": R_pc[good.values], "vlos": vlos[good].values,
        "evlos": evlos[good].fillna(2.0).values,
        "feh": feh[good].values, "efeh": efeh[good].fillna(0.2).values,
    })
    out.to_parquet(OUT)
    print(f"[members] {len(out)} (Mem=m, finite feh+vlos) -> {OUT}")
    q = lambda a: np.round(np.percentile(a, [5, 50, 95]), 2)
    print(f"  [Fe/H] 5/50/95 = {q(out['feh'])}")
    print(f"  R_pc   5/50/95 = {np.round(np.percentile(out['R_pc'],[5,50,95]),0)}")
    print(f"  inside 150/300/500 pc: {(out['R_pc']<150).sum()}/{(out['R_pc']<300).sum()}/{(out['R_pc']<500).sum()}")
    print(f"  Vlos sys = {np.median(out['vlos']):.1f} +/- {out['vlos'].std():.1f} km/s")
    med = np.median(out["feh"])
    mr = out["feh"] > med
    print(f"  median-[Fe/H] split: MR R_e~{np.median(out['R_pc'][mr]):.0f}pc (N={mr.sum()}), "
          f"MP R_e~{np.median(out['R_pc'][~mr]):.0f}pc (N={(~mr).sum()})")
    print("  [compare] current Mg multipop=148, single-pop=793")


if __name__ == "__main__":
    main()
