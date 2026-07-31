"""
Downloads the CWRU Bearing Dataset (12k Drive End) directly from the
official Case School of Engineering Bearing Data Center.

Source: https://engineering.case.edu/bearingdatacenter/download-data-file

Usage:
    python download_cwru.py --out ../data/raw

Notes:
    - Run this on your own machine (not a sandboxed environment) since it
      needs outbound access to engineering.case.edu.
    - If CWRU reorganizes their file IDs, cross-check against the table on
      their download page and update FILE_MAP below.
    - We pull 12k Drive End (12k_DE) data only, across all 4 loads (0-3 HP),
      for 4 classes: Normal, Inner Race, Outer Race (centered/6:00), Ball.
    - Fault diameter used: 0.007" (7 mil) - the mildest/most common severity,
      consistent across all 4 fault classes at all loads. You can extend
      FILE_MAP to include 0.014"/0.021" diameters later for a severity axis.
"""

import argparse
import os
import time
import urllib.request
import urllib.error

BASE_URL = "https://engineering.case.edu/sites/default/files/{}.mat"

# file_id -> (class_name, load_hp)
# IDs verified against the CWRU Bearing Data Center 12k Drive End table
# (0.007" fault diameter, DE accelerometer channel).
FILE_MAP = {
    # Normal baseline (no fault), 12k sampling
    "97":  ("normal", 0),
    "98":  ("normal", 1),
    "99":  ("normal", 2),
    "100": ("normal", 3),

    # Inner Race fault, 0.007", 12k DE
    "105": ("inner_race", 0),
    "106": ("inner_race", 1),
    "107": ("inner_race", 2),
    "108": ("inner_race", 3),

    # Ball fault, 0.007", 12k DE
    "118": ("ball", 0),
    "119": ("ball", 1),
    "120": ("ball", 2),
    "121": ("ball", 3),

    # Outer Race fault (centered / 6:00), 0.007", 12k DE
    "130": ("outer_race", 0),
    "131": ("outer_race", 1),
    "132": ("outer_race", 2),
    "133": ("outer_race", 3),
}


def download_file(file_id: str, out_path: str, retries: int = 3) -> bool:
    url = BASE_URL.format(file_id)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp, open(out_path, "wb") as f:
                f.write(resp.read())
            return True
        except urllib.error.HTTPError as e:
            print(f"  [{file_id}] HTTP {e.code} on attempt {attempt}/{retries}")
        except urllib.error.URLError as e:
            print(f"  [{file_id}] URL error: {e.reason} on attempt {attempt}/{retries}")
        time.sleep(1.5)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../data/raw", help="Output directory for .mat files")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    failed = []
    for file_id, (cls, load) in FILE_MAP.items():
        fname = f"{cls}_load{load}_{file_id}.mat"
        out_path = os.path.join(args.out, fname)
        if os.path.exists(out_path):
            print(f"[skip] {fname} already exists")
            continue
        print(f"[download] id={file_id} -> {fname}")
        ok = download_file(file_id, out_path)
        if not ok:
            failed.append(file_id)

    if failed:
        print("\nFailed to download file IDs:", failed)
        print("Cross-check these against https://engineering.case.edu/bearingdatacenter/download-data-file")
        print("and update FILE_MAP in this script if IDs have changed.")
    else:
        print(f"\nAll files downloaded to {args.out}")


if __name__ == "__main__":
    main()
