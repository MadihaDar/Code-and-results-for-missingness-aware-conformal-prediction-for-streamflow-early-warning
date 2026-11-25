"""
03_caravan_manifest.py
"""
import os
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

DEFAULT = Path(os.environ.get("FLOOD_DATA_DIR", "data")) / "raw" / "Caravan" / "Caravan-nc-download.tar.gz"
ARCHIVE = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
SMALL_DIR = ARCHIVE.parent / "extracted_small"
MANIFEST = ARCHIVE.parent.parent.parent / "processed" / "caravan_manifest.csv"
MAX_SMALL_BYTES = 50 * 1024 ** 2          


def wanted(name, size):
    """Attribute tables, licences and readme-type files only."""
    parts = [p.lower() for p in Path(name).parts]
    if size > MAX_SMALL_BYTES:
        return False
    if "attributes" in parts or "licenses" in parts or "licences" in parts:
        return True
    return name.lower().endswith((".md", ".txt", ".pdf")) and size < 5 * 1024 ** 2


def summarise(df):
    if df.empty:
        print("nothing read")
        return
    # group by the first 4 path components (e.g. Caravan/timeseries/netcdf/camels)
    df["group"] = df["name"].apply(lambda n: "/".join(Path(n).parts[:4]))
    g = df.groupby("group").agg(files=("name", "size"), gigabytes=("size", lambda s: round(s.sum() / 1e9, 2)))
    print("\nWhat is inside (first four path levels):")
    print(g.sort_values("files", ascending=False).head(40).to_string())
    nc = df[df["name"].str.endswith(".nc")]
    print(f"\n.nc files: {len(nc)}  | total files: {len(df)}  | total size: {df['size'].sum() / 1e9:.1f} GB")


def main():
    if not ARCHIVE.exists():
        sys.exit(f"archive not found: {ARCHIVE}")
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    SMALL_DIR.mkdir(exist_ok=True)
    rows, extracted, t0 = [], 0, time.time()
    try:
        with tarfile.open(ARCHIVE, mode="r|gz") as tar:          # streaming mode: no seeking, no unpacking
            for i, m in enumerate(tar, 1):
                if not m.isfile():
                    continue
                rows.append((m.name, m.size))
                if wanted(m.name, m.size):
                    tar.extract(m, SMALL_DIR, filter="data")
                    extracted += 1
                if i % 2000 == 0:
                    gb = sum(s for _, s in rows) / 1e9
                    print(f"  {i} entries | {gb:.1f} GB read | {extracted} small files extracted | "
                          f"{(time.time() - t0) / 60:.1f} min", flush=True)
    except KeyboardInterrupt:
        print("\ninterrupted - saving what was read so far")
    df = pd.DataFrame(rows, columns=["name", "size"])
    df.to_csv(MANIFEST, index=False)
    print(f"\nmanifest saved: {MANIFEST} ({len(df)} files)")
    print(f"small files extracted to: {SMALL_DIR} ({extracted} files)")
    summarise(df)


if __name__ == "__main__":
    main()
