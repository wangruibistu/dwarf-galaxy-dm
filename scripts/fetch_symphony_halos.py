#!/usr/bin/env python3
"""Fetch a Symphony zoom-in subhalo catalog for diffusion-prior training.

The Symphony simulation suite (Nadler+ 2023) is publicly available at
https://web.stanford.edu/group/wechsler-lab/symphony. We pull the
dwarf-mass-host subset and extract ρ(r) profiles.

This script does NOT auto-run by default (Symphony files are large and the
download URLs may change). It is provided as a runbook documenting the steps;
edit and run manually when you're ready.

Roadmap:
  Phase 1: pull MWest hosts (10 zooms of MW-mass) + their subhalos
  Phase 2: filter subhalos to dwarf-mass range 10^9 - 10^11 Msun
  Phase 3: profile extraction with `colossus` or symlib
  Phase 4: replace `synthetic_halos.py` dataset in the diffusion training
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Where to download to
ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "data" / "external" / "symphony"
TARGET.mkdir(parents=True, exist_ok=True)


SYMPHONY_OVERVIEW = """
Symphony zoom-in simulation suite (Nadler+ 2023, ApJ 945, 159)

Public access:
  - Symphony web tool: https://web.stanford.edu/group/wechsler-lab/symphony
  - Public DR: globus collection 'Symphony' (~5 TB total for all suites)
  - Per-host subhalo catalogs at z=0 are ~100 MB each (downloadable as HDF5)

Recommended subsets for our diffusion prior training:
  - SymphonyMilkyWay   : 45 MW-analog hosts, each with ~10^4 subhalos
  - SymphonyDwarf      : 39 LMC-mass hosts
  - SymphonyMilkyWayLR : low-res version for quick prototyping (~1 GB)

What we need per halo:
  - rvir, mvir, vmax, c (concentration)
  - ρ(r) profile sampled at log-spaced bins from ~5 pc to ~3 kpc
  - tidal track variables (m_acc, t_infall)

For the Paper II production model:
  - Combine Symphony + FIRE-2 (Hopkins+ 2018, dwarf subset on Flathub)
    + EDGE (Rey+, by collaboration)
  - Total ~10^5 halo profiles for diffusion training

To run this script manually:
  1. Register at the Stanford Wechsler-lab Symphony portal
  2. Download SymphonyMilkyWayLR_LMC_sub_HiResolution_z=0.h5 (or similar)
  3. Save to data/external/symphony/
  4. Update --catalog flag below to point to your file
  5. Run with: python scripts/fetch_symphony_halos.py --catalog <path>
"""


def extract_profiles_from_symphony(catalog_path: Path, n_max: int = 5000):
    """Stub: extract (ρ(r), conditioning) tuples from a Symphony HDF5 catalog.

    Production version will use either `symlib` (Mansfield+ 2023) or direct
    h5py reads following the Symphony data dictionary.
    """
    import h5py  # noqa: F401
    print(f"[NYI] extract_profiles_from_symphony({catalog_path}, n_max={n_max})")
    print("      Implementation pending real Symphony catalog file.")
    print("      Expected output: data/external/symphony/profiles.npz")
    print("      Format: log_rho (n, 32), cond (n, 4)")
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", type=Path,
                   help="Path to Symphony HDF5 catalog (manually downloaded)")
    p.add_argument("--info", action="store_true", help="Print Symphony overview and exit")
    args = p.parse_args()

    if args.info or not args.catalog:
        print(SYMPHONY_OVERVIEW)
        if not args.catalog:
            sys.exit(0)

    if args.catalog and not args.catalog.exists():
        print(f"ERROR: catalog file {args.catalog} not found.")
        print("       Download from https://web.stanford.edu/group/wechsler-lab/symphony")
        sys.exit(1)

    extract_profiles_from_symphony(args.catalog)


if __name__ == "__main__":
    main()
