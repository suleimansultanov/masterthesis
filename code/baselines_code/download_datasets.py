#!/usr/bin/env python3
"""
download_datasets.py
====================
Reliable downloader for the two benchmark datasets used in the thesis:

  * OPPORTUNITY  (UCI, ~292 MB single zip)  -> data/opportunity/
  * CASAS        (Zenodo record, text logs) -> data/casas/

Features that matter on a slow/flaky connection:
  - streaming download (constant memory),
  - HTTP range *resume* (re-run to continue a partial download),
  - automatic retries with backoff,
  - a live progress line (MB + %),
  - automatic unzip.

Usage
-----
    pip install requests tqdm        # tqdm optional; plain progress used if absent
    python download_datasets.py                 # both datasets, defaults
    python download_datasets.py --only opportunity
    python download_datasets.py --only casas --casas-record 15712834
    python download_datasets.py --casas-record 15708568   # a different CASAS set

Notes
-----
- OPPORTUNITY comes from UCI; the server is genuinely slow. The resume logic
  means a dropped connection does not start over -- just run the script again.
- CASAS is hosted on Zenodo as several records. The script asks the Zenodo API
  for the file list of the chosen record and downloads each file. Browse records
  at https://zenodo.org/communities/casas and pass the numeric id via
  --casas-record. Default below is a single-resident scripted-ADL record.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import zipfile

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")

OPPORTUNITY_URL = "https://archive.ics.uci.edu/static/public/226/opportunity+activity+recognition.zip"
DEFAULT_CASAS_RECORD = "15712834"  # CASAS scripted-ADL record on Zenodo

CHUNK = 1 << 16          # 64 KB
MAX_RETRIES = 8
TIMEOUT = 60


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def download(url: str, dest: str) -> str:
    """Stream-download `url` to `dest` with resume + retries. Returns dest path."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"

    for attempt in range(1, MAX_RETRIES + 1):
        existing = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        headers = {"Range": f"bytes={existing}-"} if existing else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=TIMEOUT) as r:
                # 206 = partial (resume accepted); 200 = full (server ignored range)
                if existing and r.status_code == 200:
                    existing = 0  # server won't resume; restart cleanly
                    mode = "wb"
                elif existing and r.status_code == 206:
                    mode = "ab"
                else:
                    mode = "wb"
                r.raise_for_status()

                total = int(r.headers.get("Content-Length", 0)) + existing
                done = existing
                last = time.time()
                with open(tmp, mode) as f:
                    for chunk in r.iter_content(CHUNK):
                        if not chunk:
                            continue
                        f.write(chunk)
                        done += len(chunk)
                        if time.time() - last > 0.3:
                            pct = f"{100*done/total:.1f}%" if total else "?"
                            sys.stdout.write(
                                f"\r  {os.path.basename(dest)}: {_human(done)} / "
                                f"{_human(total) if total else '?'} ({pct})   "
                            )
                            sys.stdout.flush()
                            last = time.time()
            sys.stdout.write("\n")
            os.replace(tmp, dest)
            return dest
        except (requests.RequestException, OSError) as e:
            wait = min(2 ** attempt, 30)
            print(f"\n  ! attempt {attempt}/{MAX_RETRIES} failed: {e}. "
                  f"retrying in {wait}s (partial file kept for resume)...")
            time.sleep(wait)

    raise RuntimeError(f"Failed to download {url} after {MAX_RETRIES} attempts.")


def unzip(zip_path: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    print(f"  unzipping {os.path.basename(zip_path)} -> {out_dir}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)


def get_opportunity() -> None:
    print("== OPPORTUNITY (UCI, ~292 MB) ==")
    out = os.path.join(DATA_DIR, "opportunity")
    zip_path = os.path.join(out, "opportunity.zip")
    download(OPPORTUNITY_URL, zip_path)
    unzip(zip_path, out)
    # the archive nests .dat files under OpportunityUCIDataset/dataset/
    print("  done. .dat files are under data/opportunity/OpportunityUCIDataset/dataset/")


def get_casas(record_id: str) -> None:
    print(f"== CASAS (Zenodo record {record_id}) ==")
    out = os.path.join(DATA_DIR, "casas")
    api = f"https://zenodo.org/api/records/{record_id}"
    meta = requests.get(api, timeout=TIMEOUT).json()
    files = meta.get("files", [])
    if not files:
        raise RuntimeError(
            f"No files listed for Zenodo record {record_id}. "
            f"Check the id at https://zenodo.org/communities/casas"
        )
    for fobj in files:
        fname = fobj.get("key") or fobj.get("filename")
        link = fobj["links"].get("self") or fobj["links"].get("download")
        dest = os.path.join(out, fname)
        print(f"  -> {fname} ({_human(fobj.get('size', 0))})")
        download(link, dest)
        if fname.lower().endswith(".zip"):
            unzip(dest, out)
    print("  done. CASAS files are under data/casas/")


def main() -> None:
    ap = argparse.ArgumentParser(description="Download OPPORTUNITY and CASAS datasets.")
    ap.add_argument("--only", choices=["opportunity", "casas"],
                    help="download just one dataset (default: both)")
    ap.add_argument("--casas-record", default=DEFAULT_CASAS_RECORD,
                    help=f"Zenodo record id for CASAS (default {DEFAULT_CASAS_RECORD})")
    args = ap.parse_args()

    if args.only in (None, "opportunity"):
        get_opportunity()
    if args.only in (None, "casas"):
        get_casas(args.casas_record)

    print("\nAll requested downloads complete.")


if __name__ == "__main__":
    main()
