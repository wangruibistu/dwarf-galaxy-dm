#!/usr/bin/env python3
"""Fetch and profile-extract FIRE-2 dwarf-mass halos for diffusion-prior training.

This is the production data path for replacing `synthetic_halos.py` /
`realistic_halos.py` in Phase 1c (Paper I revision / Paper II training).

Workflow:
  1. Register at Flathub (manual, one-time)
  2. Use globus-cli (or boto3 for the S3 mirror) to bulk-download a target
     subset of FIRE-2 dwarf-mass zoom-ins (m10*, m11*, m12* hosts)
  3. For each snapshot at z=0:
       a. Use pynbody / caesar to find the main halo center
       b. Bin DM particle positions (PartType1) into log-spaced shells
       c. Compute (M_halo, M_star, SFH duration, tidal radius) for conditioning
  4. Write unified output `data/external/halos_unified.npz` with
     {log_rho, cond, source} arrays.

This script is a runbook stub. Actual data fetch and profile extraction
require pynbody, h5py, and Flathub credentials. To run:

    python scripts/fetch_fire2_halos.py --globus-endpoint <endpoint-id> \\
                                         --target-dir data/external/fire2

References:
  - FIRE-2 public data: https://flathub.flatironinstitute.org/fire
  - Hopkins+ 2018 (MNRAS 480, 800) — simulation suite paper
  - Sanderson+ 2020 (ApJS 246, 6) — m12 halo subset overview
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "data" / "external" / "fire2"
TARGET.mkdir(parents=True, exist_ok=True)


FIRE2_TARGETS = [
    # (sim_name, host_mass, comment)
    ("m10b_res250md", 1e10, "UFD-scale isolated dwarf"),
    ("m10c_res250md", 1e10, "UFD-scale isolated dwarf"),
    ("m11a_res2100",  1e11, "dwarf-mass isolated dwarf"),
    ("m11b_res2100",  1e11, "dwarf-mass isolated dwarf"),
    ("m12i_res7100",  1e12, "MW-mass host; we want its satellites"),
    ("m12f_res7100",  1e12, "MW-mass host; we want its satellites"),
    ("m12m_res7100",  1e12, "MW-mass host; we want its satellites"),
]


def extract_profile_from_snapshot(snapshot_path: Path, halo_idx: int = 0):
    """Stub: load a gizmo HDF5 snapshot, find halo, extract ρ(r)."""
    try:
        import pynbody  # noqa
    except ImportError:
        print("ERROR: pynbody required. pip install pynbody")
        return None
    print(f"[NYI] extract_profile_from_snapshot({snapshot_path}, halo={halo_idx})")
    return None


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--globus-endpoint",
                   help="Globus endpoint ID for Flathub download")
    p.add_argument("--target-dir", type=Path, default=TARGET)
    p.add_argument("--list-only", action="store_true",
                   help="Just print the target halos and exit")
    args = p.parse_args()

    if args.list_only or not args.globus_endpoint:
        print("FIRE-2 target halos for diffusion-prior training:")
        for name, M, comment in FIRE2_TARGETS:
            print(f"  {name:20s}  M_halo ~ {M:.0e}  ({comment})")
        print()
        print("To download via Globus (after Flathub registration):")
        print("  globus transfer <flathub-fire-endpoint> <your-endpoint> --batch")
        print()
        print("To extract profiles, this script needs pynbody + the actual snapshot files.")
        sys.exit(0)

    for name, M, _ in FIRE2_TARGETS:
        # placeholder — would launch globus transfer per simulation
        print(f"[transfer] {name}: would download via Globus {args.globus_endpoint}")

    # Extract profiles after download
    for name, M, _ in FIRE2_TARGETS:
        snap = args.target_dir / name / "snapshot_600.hdf5"
        if snap.exists():
            extract_profile_from_snapshot(snap)


if __name__ == "__main__":
    main()
