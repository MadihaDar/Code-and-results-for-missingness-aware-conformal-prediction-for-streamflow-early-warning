"""
04_extract_caravan_sources.py
Second pass over the Caravan archive: extract ONLY the sources you choose
(time series, attributes, basin shapes, licence) and stop reading as soon as
everything on the wish list has been extracted.
"""
import os
import argparse
import shutil
import tarfile
import time
from pathlib import Path

import pandas as pd

BASE = Path(os.environ.get("FLOOD_DATA_DIR", "data"))
ARCHIVE = BASE / "raw" / "Caravan" / "Caravan-nc-download.tar.gz"
MANIFEST = BASE / "processed" / "caravan_manifest.csv"
OUT_DIR = BASE / "raw" / "Caravan" / "extracted"
TOP = "Caravan-nc"          # top-level folder inside the archive


def wanted_names(manifest, sources):
    names = {}
    for src in sources:
        prefixes = (f"{TOP}/timeseries/netcdf/{src}/", f"{TOP}/attributes/{src}/",
                    f"{TOP}/shapefiles/{src}/", f"{TOP}/licenses/{src}/")
        sub = manifest[manifest["name"].str.startswith(prefixes)]
        if sub.empty:
            raise SystemExit(f"source {src!r} not found in the manifest")
        names.update(dict(zip(sub["name"], sub["size"])))
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["lamah", "camelsgb"])
    ap.add_argument("--archive", default=str(ARCHIVE))
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    archive, out = Path(args.archive), Path(args.out)
    manifest = pd.read_csv(args.manifest)
    want = wanted_names(manifest, args.sources)
    need_gb = sum(want.values()) / 1e9
    out.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(out).free / 1e9
    nc = sum(n.endswith(".nc") for n in want)
    print(f"sources: {args.sources} | files: {len(want)} ({nc} basin .nc files) | size: {need_gb:.1f} GB "
          f"| free disk: {free_gb:.1f} GB")
    if free_gb < need_gb * 1.2:
        raise SystemExit("not enough free disk space (need about 20% headroom) - aborting")
    if args.dry_run:
        return

    remaining, done, skipped, t0 = set(want), 0, 0, time.time()
    try:
        with tarfile.open(archive, mode="r|gz") as tar:
            for i, m in enumerate(tar, 1):
                if m.name in remaining:
                    target = out / m.name
                    if target.exists() and target.stat().st_size == m.size:
                        skipped += 1
                    else:
                        tar.extract(m, out, filter="data")
                        done += 1
                    remaining.discard(m.name)
                    if not remaining:
                        print("all wanted files found - stopping early")
                        break
                if i % 2000 == 0:
                    print(f"  scanned {i} entries | extracted {done} | skipped {skipped} | "
                          f"left {len(remaining)} | {(time.time() - t0) / 60:.1f} min", flush=True)
    except KeyboardInterrupt:
        print("\ninterrupted - re-run to continue (existing files are skipped)")

    print(f"\nextracted {done}, already present {skipped}, missing {len(remaining)} "
          f"in {(time.time() - t0) / 60:.1f} min")
    bad = [n for n in want if not (out / n).exists() or (out / n).stat().st_size != want[n]]
    print("verification: all files present with the expected size" if not bad
          else f"verification FAILED for {len(bad)} files, e.g. {bad[:3]}")
    print(f"data is in: {out / TOP}")


if __name__ == "__main__":
    main()
