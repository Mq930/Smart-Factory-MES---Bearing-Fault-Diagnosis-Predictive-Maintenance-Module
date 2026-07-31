"""
Downloads the CWRU Bearing Dataset (12k Drive End) directly from the
official Case School of Engineering Bearing Data Center.

Source: https://engineering.case.edu/bearingdatacenter/download-data-file

Usage:
    python download_cwru.py --out ../data/raw

Notes:
    - Run this on your own machine (not a sandboxed environment) since it
      needs outbound access to engineering.case.edu.
    - We pull 12k Drive End (12k_DE) data only, across all 4 loads (0-3 HP),
      for 3 fault diameters (0.007", 0.014", 0.021") x 3 fault types
      (Inner Race, Ball, Outer Race @ 6:00/centered) = 12 fault classes,
      plus Normal baseline = 13 total recordings groups -> 12-class problem
      (normal collapses fault-free, so 1 + 3*3 = 10... see dataset.py
      CLASS_NAMES for the exact list actually used).
    - File IDs below are copied directly from the official table at
      https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data
      (verified July 2026). If CWRU reorganizes their file IDs, cross-check
      against that page and update FILE_MAP.
"""

import argparse
import http.client
import os
import time
import urllib.request
import urllib.error

BASE_URL = "https://engineering.case.edu/sites/default/files/{}.mat"

# file_id -> (class_name, load_hp)
# class_name encodes fault_type + severity, e.g. "inner_race_007".
# "normal" has no severity suffix (no fault).
# IDs verified against the official 12k Drive End table (Inner Race, Ball,
# and Outer Race Centered/6:00 columns only - Orthogonal/Opposite outer race
# positions are excluded to keep this a clean 4-fault-type x 3-severity grid).
FILE_MAP = {
    # ---- Normal baseline (no fault), 12k sampling ----
    "97":  ("normal", 0),
    "98":  ("normal", 1),
    "99":  ("normal", 2),
    "100": ("normal", 3),

    # ---- 0.007" fault diameter ----
    "105": ("inner_race_007", 0),
    "106": ("inner_race_007", 1),
    "107": ("inner_race_007", 2),
    "108": ("inner_race_007", 3),

    "118": ("ball_007", 0),
    "119": ("ball_007", 1),
    "120": ("ball_007", 2),
    "121": ("ball_007", 3),

    "130": ("outer_race_007", 0),
    "131": ("outer_race_007", 1),
    "132": ("outer_race_007", 2),
    "133": ("outer_race_007", 3),

    # ---- 0.014" fault diameter ----
    "169": ("inner_race_014", 0),
    "170": ("inner_race_014", 1),
    "171": ("inner_race_014", 2),
    "172": ("inner_race_014", 3),

    "185": ("ball_014", 0),
    "186": ("ball_014", 1),
    "187": ("ball_014", 2),
    "188": ("ball_014", 3),

    "197": ("outer_race_014", 0),
    "198": ("outer_race_014", 1),
    "199": ("outer_race_014", 2),
    "200": ("outer_race_014", 3),

    # ---- 0.021" fault diameter ----
    "209": ("inner_race_021", 0),
    "210": ("inner_race_021", 1),
    "211": ("inner_race_021", 2),
    "212": ("inner_race_021", 3),

    "222": ("ball_021", 0),
    "223": ("ball_021", 1),
    "224": ("ball_021", 2),
    "225": ("ball_021", 3),

    "234": ("outer_race_021", 0),
    "235": ("outer_race_021", 1),
    "236": ("outer_race_021", 2),
    "237": ("outer_race_021", 3),
}


def download_file(file_id: str, out_path: str, retries: int = 5) -> bool:
    url = BASE_URL.format(file_id)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            if len(data) < 100_000:  # real CWRU .mat files are several hundred KB+
                raise ValueError(f"Downloaded file suspiciously small ({len(data)} bytes) - likely truncated or an error page")
            with open(out_path, "wb") as f:
                f.write(data)
            return True
        except urllib.error.HTTPError as e:
            print(f"  [{file_id}] HTTP {e.code} on attempt {attempt}/{retries}")
        except urllib.error.URLError as e:
            print(f"  [{file_id}] URL error: {e.reason} on attempt {attempt}/{retries}")
        except (http.client.IncompleteRead, ConnectionError, TimeoutError, ValueError) as e:
            print(f"  [{file_id}] Download issue on attempt {attempt}/{retries}: {e}")
        # Clean up any partial file so a failed attempt never leaves corrupt
        # data behind that could be mistaken for a successful download later.
        if os.path.exists(out_path):
            os.remove(out_path)
        time.sleep(2.0 * attempt)  # backoff a bit more each retry
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
