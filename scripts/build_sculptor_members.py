#!/usr/bin/env python3
"""Build the Sculptor dSph member catalog.

Pipeline:
  1. VizieR fetch of Walker+2009 (J/AJ/137/3100) MMFS RV catalog
  2. Filter on Target prefix "Scl"
  3. Gaia DR3 1-deg cone around Sculptor center
  4. 1-arcsec position crossmatch
  5. PM-based membership (vs Battaglia+ 2022 systemic PM)
  6. Elliptical projected radius (Munoz+ 2018 geometry)
  7. Write `data/processed/sculptor_members_v0.parquet`

Refs:
  Walker, Mateo & Olszewski 2009 (AJ 137, 3100)
  Battaglia+ 2022 (A&A 657, A54) Gaia eDR3 dwarf PMs
  Munoz+ 2018 (ApJ 860, 66) structural parameters
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astroquery.gaia import Gaia
from astroquery.vizier import Vizier


ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROC.mkdir(parents=True, exist_ok=True)


# Sculptor structural parameters (Munoz+ 2018)
RA0 = 15.0392        # deg (J2000)
DEC0 = -33.7092      # deg
DIST_KPC = 86.0
ELLIP = 0.36         # 1 - b/a
PA_DEG = 92.0        # deg of major axis
RE_PC = 280.0        # Plummer half-light

# Gaia systemic PM for Sculptor (Battaglia+ 2022, eDR3 fit)
PM_RA_SYS = 0.099    # mas/yr
PM_DEC_SYS = -0.160
SIGMA_PM_MEMBER = 0.20   # mas/yr selection radius (Gaia DR3 typical)


GAIA_QUERY = f"""
SELECT
    source_id, ra, dec, parallax, parallax_over_error,
    pmra, pmra_error, pmdec, pmdec_error,
    phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, bp_rp,
    ruwe, phot_bp_rp_excess_factor
FROM gaiadr3.gaia_source
WHERE
    1 = CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {RA0}, {DEC0}, 1.0)
    )
    AND phot_g_mean_mag BETWEEN 13 AND 21
    AND ruwe < 1.4
"""


def query_gaia(out_path: Path) -> pd.DataFrame:
    print("[gaia] querying DR3 1-deg cone around Sculptor (~1-3 min)")
    job = Gaia.launch_job_async(GAIA_QUERY)
    tab: Table = job.get_results()
    df = tab.to_pandas()
    df.to_parquet(out_path, index=False)
    print(f"[gaia] wrote {out_path}: {len(df):,} rows")
    return df


def load_walker_sculptor() -> pd.DataFrame:
    """Pull Walker+ 2009 J/AJ/137/3100/stars and filter to Sculptor."""
    Vizier.ROW_LIMIT = -1
    cats = Vizier.get_catalogs("J/AJ/137/3100")
    tab: Table = cats["J/AJ/137/3100/stars"]
    df = tab.to_pandas()
    df["dwarf"] = df["Target"].str[:3]
    scl = df[df["dwarf"] == "Scl"].reset_index(drop=True).copy()
    print(f"[walker] Sculptor rows: {len(scl):,} of {len(df):,} total")

    # Parse HMS/DMS to decimal degrees
    coords = SkyCoord(ra=scl["RAJ2000"].values, dec=scl["DEJ2000"].values,
                      unit=(u.hourangle, u.deg))
    scl["ra"] = coords.ra.deg
    scl["dec"] = coords.dec.deg

    # Standardize column names (the `<>` in column names are awkward)
    rename = {}
    for raw, std in [("<HV>", "v_los_kms"), ("e_<HV>", "v_err_kms"),
                     ("<SigMg>", "sigmg"), ("e_<SigMg>", "sigmg_err"),
                     ("Mmb", "mmb")]:
        if raw in scl.columns:
            rename[raw] = std
    scl = scl.rename(columns=rename)
    return scl


def crossmatch_walker_gaia(walker: pd.DataFrame, gaia: pd.DataFrame) -> pd.DataFrame:
    sc_w = SkyCoord(ra=walker["ra"].values * u.deg, dec=walker["dec"].values * u.deg)
    sc_g = SkyCoord(ra=gaia["ra"].values * u.deg, dec=gaia["dec"].values * u.deg)
    idx, sep, _ = sc_w.match_to_catalog_sky(sc_g)
    ok = sep.arcsec < 1.5
    out = walker.loc[ok].reset_index(drop=True).copy()
    g_matched = gaia.iloc[idx[ok]].reset_index(drop=True)
    # Avoid column collision on ra/dec
    g_matched = g_matched.rename(columns={"ra": "ra_g", "dec": "dec_g"})
    out = pd.concat([out, g_matched], axis=1)
    print(f"[xmatch] {len(out):,} of {len(walker):,} Walker stars matched to Gaia within 1.5\"")
    return out


def elliptical_radius_pc(ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    cos_d = np.cos(np.deg2rad(DEC0))
    x = (ra - RA0) * cos_d
    y = dec - DEC0
    pa = np.deg2rad(PA_DEG)
    xr = x * np.cos(pa) + y * np.sin(pa)
    yr = -x * np.sin(pa) + y * np.cos(pa)
    r_ell_deg = np.sqrt(xr ** 2 + (yr / (1.0 - ELLIP)) ** 2)
    return r_ell_deg * np.deg2rad(1.0) * DIST_KPC * 1000.0


def member_probability(df: pd.DataFrame) -> np.ndarray:
    """Combine: foreground parallax cut × Gaussian PM × Walker membership prob."""
    # Parallax cut via parallax_over_error (foreground stars typically have |ϖ/σ| > 5)
    if "parallax_over_error" in df.columns:
        pov = df["parallax_over_error"].fillna(0.0).values
    else:
        pov = np.zeros(len(df))
    p_par = (np.abs(pov) < 5.0).astype(float)

    # PM Gaussian centred on Sculptor systemic motion
    dpm = np.sqrt((df["pmra"].fillna(99) - PM_RA_SYS) ** 2 +
                  (df["pmdec"].fillna(99) - PM_DEC_SYS) ** 2)
    p_pm = np.exp(-0.5 * (dpm / SIGMA_PM_MEMBER) ** 2)

    # Walker Mmb IS the membership probability (0-1, not counter)
    if "mmb" in df.columns:
        p_walker = pd.to_numeric(df["mmb"], errors="coerce").fillna(0.0).values
    else:
        p_walker = np.full(len(df), 0.8)

    # Require finite RV
    p_rv = df["v_los_kms"].notna().astype(float).values

    return np.clip(p_par * p_pm * p_walker * p_rv, 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-gaia", action="store_true")
    args = parser.parse_args()

    gaia_path = DATA_RAW / "sculptor_gaia.parquet"
    if not args.skip_gaia or not gaia_path.exists():
        gaia = query_gaia(gaia_path)
    else:
        gaia = pd.read_parquet(gaia_path)
        print(f"[load] cached gaia {len(gaia):,} rows from {gaia_path}")

    walker = load_walker_sculptor()
    df = crossmatch_walker_gaia(walker, gaia)

    df["R_pc"] = elliptical_radius_pc(df["ra"].values, df["dec"].values)
    df["P_member"] = member_probability(df)

    keep_cols = ["source_id", "ra", "dec", "R_pc",
                 "v_los_kms", "v_err_kms", "sigmg", "sigmg_err", "mmb",
                 "pmra", "pmra_error", "pmdec", "pmdec_error",
                 "parallax", "parallax_over_error",
                 "phot_g_mean_mag", "bp_rp",
                 "P_member"]
    out_df = df[[c for c in keep_cols if c in df.columns]].copy()

    out_path = DATA_PROC / "sculptor_members_v0.parquet"
    out_df.to_parquet(out_path, index=False)
    print(f"[done] wrote {out_path}: {len(out_df):,} stars, "
          f"{(out_df.P_member > 0.5).sum():,} P>0.5, "
          f"{(out_df.P_member > 0.95).sum():,} P>0.95")


if __name__ == "__main__":
    main()
