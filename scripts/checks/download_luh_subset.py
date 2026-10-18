"""Download the LUH/KIT milling dataset (subset) from Mendeley Data.

Fetches ``filelist.csv`` plus every N-th run's HDF5 file from
https://data.mendeley.com/datasets/zpxs87bjt8 (DOI 10.17632/zpxs87bjt8,
CC BY) into ``data/raw/luh_milling/``. The full dataset is 6,418 files
(~10.6 GB, use ``every=1``); the default every-4th-run mirror is
~1,600 files (~2.6 GB). Already-downloaded files are skipped, so
re-running resumes.

Usage (from the project root)::

    .\\.venv\\Scripts\\python.exe scripts/checks/download_luh_subset.py
    # full set: every=1, workers=8
    .\\.venv\\Scripts\\python.exe scripts/checks/download_luh_subset.py 1 8
    # optional: every=2, workers=8   (N-th run, concurrent fetches)
    .\\.venv\\Scripts\\python.exe scripts/checks/download_luh_subset.py 2 8
"""

from __future__ import annotations

import concurrent.futures as cf
import csv
import sys
import time
import urllib.request
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
OUT_DIR = ROOT / "data" / "raw" / "luh_milling"
H5_DIR = OUT_DIR / "h5"
FILELIST = OUT_DIR / "filelist.csv"
API = "https://data.mendeley.com/public-api/datasets/zpxs87bjt8"
FILELIST_ID = "01d97477-db7e-480d-b448-1e6c3fc8fe08"
FILELIST_URL = (
    "https://data.mendeley.com/public-files/datasets/zpxs87bjt8/files/"
    f"{FILELIST_ID}/file_downloaded"
)
UA = {"User-Agent": "Mozilla/5.0"}


def _fetch(url: str, dest: Path, timeout: int = 180) -> None:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as fh:
        fh.write(r.read())


def main() -> None:
    every = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8

    import json

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not FILELIST.exists():
        print(f"fetching {FILELIST} ...")
        _fetch(FILELIST_URL, FILELIST)

    req = urllib.request.Request(API, headers=UA)
    with urllib.request.urlopen(req, timeout=90) as r:
        meta = json.load(r)
    url_by_name = {}
    for f in meta["files"]:
        cd = f.get("content_details") or {}
        if cd.get("download_url"):
            url_by_name[f["filename"]] = cd["download_url"]

    selected: list[str] = []
    with open(FILELIST, "r", encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            if (int(row["run"]) - 1) % every == 0:
                selected.append(row["filename"])

    H5_DIR.mkdir(parents=True, exist_ok=True)
    todo = [
        n for n in selected
        if not ((H5_DIR / n).exists() and (H5_DIR / n).stat().st_size > 1000)
    ]
    print(f"selected={len(selected)} todo={len(todo)} every={every} workers={workers}")

    t0, done, failed = time.time(), 0, []

    def fetch(name: str) -> tuple[str, str]:
        dest = H5_DIR / name
        for attempt in range(3):
            try:
                _fetch(url_by_name[name], dest.with_suffix(".part"))
                if dest.with_suffix(".part").stat().st_size > 1000:
                    dest.with_suffix(".part").replace(dest)
                    return name, "ok"
            except Exception as exc:  # noqa: BLE001
                last = str(exc)
                time.sleep(1.0 + attempt)
        return name, f"FAIL: {last}"

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for name, status in ex.map(fetch, todo):
            done += 1
            if status != "ok":
                failed.append((name, status))
            if done % 200 == 0 or done == len(todo):
                rate = done / max(time.time() - t0, 1e-9)
                print(f"  {done}/{len(todo)}  {rate:.1f} files/s", flush=True)

    print(f"finished {done} in {time.time() - t0:.0f}s")
    if failed:
        print(f"FAILURES: {len(failed)} (re-run to retry)")
        for n, s in failed[:10]:
            print("  ", n, s)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
