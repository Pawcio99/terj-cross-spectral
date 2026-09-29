#!/usr/bin/env python3
"""Acquire and integrity-check Riemann zero tables for TERJ Cross-Spectral.

Primary source:
https://www-users.cse.umn.edu/~odlyzko/zeta_tables/

The 10^22 block is treated as a strict holdout: by default this program may
download and hash it, but will not parse, summarize, window, or convert it.

Author: Paweł Majsterek
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BASE_URL = "https://www-users.cse.umn.edu/~odlyzko/zeta_tables"

TABLES = {
    "first_2m": {
        "url": f"{BASE_URL}/zeros6.gz",
        "start_index": 1,
        "count": 2_001_052,
        "base": 0,
        "compressed": "gzip",
        "holdout": False,
        "accuracy_note": "Odlyzko index page: accurate to within 4e-9",
    },
    "1e12": {
        "url": f"{BASE_URL}/zeros3",
        "start_index": 10**12 + 1,
        "count": 10_000,
        "base": 267653395647,
        "compressed": None,
        "holdout": False,
        "accuracy_note": "header: guaranteed accurate to within 1e-8",
    },
    "1e21": {
        "url": f"{BASE_URL}/zeros4",
        "start_index": 10**21 + 1,
        "count": 10_000,
        "base": 144176897509546973000,
        "compressed": None,
        "holdout": False,
        "accuracy_note": "header: probably accurate to within 1e-6",
    },
    "1e22": {
        "url": f"{BASE_URL}/zeros5",
        "start_index": 10**22 + 1,
        "count": 10_000,
        "base": 1370919909931995300000,
        "compressed": None,
        "holdout": True,
        "accuracy_note": "header: probably accurate to within 1e-6",
    },
}

FLOAT_RE = re.compile(r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)\s*$")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=dest.name + ".", suffix=".partial", dir=dest.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "TERJ-cross-spectral/1.0"})
        with urllib.request.urlopen(req, timeout=120) as r, tmp.open("wb") as out:
            while True:
                chunk = r.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            tmp.unlink()


def numeric_lines(path: Path, compression: str | None) -> list[float]:
    opener = gzip.open if compression == "gzip" else open
    vals: list[float] = []
    with opener(path, "rt", encoding="utf-8", errors="strict") as f:
        for line in f:
            m = FLOAT_RE.match(line)
            if m:
                vals.append(float(m.group(1)))
    return vals


def first_text_lines(path: Path, compression: str | None, n: int = 12) -> list[str]:
    opener = gzip.open if compression == "gzip" else open
    out: list[str] = []
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        for _ in range(n):
            line = f.readline()
            if not line:
                break
            out.append(line.rstrip("\n"))
    return out


def verify_high_header(name: str, lines: list[str], base: int) -> None:
    if name not in {"1e12", "1e21", "1e22"}:
        return
    header = " ".join(lines[:3]).replace(",", "")
    if str(base) not in header:
        raise RuntimeError(f"{name}: expected base {base} not found in header")


def convert_table(name: str, src: Path, dest: Path) -> dict:
    spec = TABLES[name]
    if spec["holdout"]:
        raise RuntimeError("strict holdout conversion is forbidden without explicit unlock")
    if dest.exists():
        raise FileExistsError(f"refusing to overwrite converted file: {dest}")

    lines = first_text_lines(src, spec["compressed"])
    verify_high_header(name, lines, int(spec["base"]))
    vals = np.asarray(numeric_lines(src, spec["compressed"]), dtype=np.float64)
    if vals.size != int(spec["count"]):
        raise RuntimeError(f"{name}: expected {spec['count']} values, found {vals.size}")
    if not np.all(np.diff(vals) > 0):
        raise RuntimeError(f"{name}: values are not strictly increasing")

    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        dest,
        base=str(spec["base"]),
        x=vals,
        start_index=str(spec["start_index"]),
        source_url=spec["url"],
        source_sha256=sha256_file(src),
    )
    return {
        "converted": str(dest),
        "n": int(vals.size),
        "offset_first": float(vals[0]),
        "offset_last": float(vals[-1]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument(
        "--tables",
        nargs="+",
        choices=sorted(TABLES),
        default=["first_2m", "1e12", "1e21", "1e22"],
    )
    ap.add_argument(
        "--unlock-holdout",
        action="store_true",
        help="allow conversion of 1e22; do not use before final Stage A freeze",
    )
    args = ap.parse_args()

    work = Path(args.workspace)
    original = work / "raw" / "riemann" / "odlyzko" / "original"
    converted = work / "raw" / "riemann" / "odlyzko" / "converted"
    holdout = work / "raw" / "riemann" / "holdout_1e22"
    manifest_dir = work / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for name in args.tables:
        spec = TABLES[name]
        suffix = ".gz" if spec["compressed"] == "gzip" else ".txt"
        base_dir = holdout if spec["holdout"] else original
        src = base_dir / f"odlyzko_{name}{suffix}"

        print(f"[download] {name}: {spec['url']}")
        atomic_download(str(spec["url"]), src)
        digest = sha256_file(src)
        size = src.stat().st_size
        if spec["holdout"] and not args.unlock_holdout:
            # Strict holdout: do not parse or preview local numeric content.
            lines = []
        else:
            lines = first_text_lines(src, spec["compressed"], n=7)
            verify_high_header(name, lines, int(spec["base"]))

        rec = {
            "name": name,
            "url": spec["url"],
            "downloaded_path": str(src),
            "sha256": digest,
            "bytes": size,
            "start_index": str(spec["start_index"]),
            "count_expected": int(spec["count"]),
            "base": str(spec["base"]),
            "holdout": bool(spec["holdout"]),
            "accuracy_note": spec["accuracy_note"],
            "header_preview": lines[:7] if not spec["holdout"] else ["WITHHELD: strict holdout"],
            "holdout_slice_start": 1808 if spec["holdout"] else None,
            "holdout_slice_count": 8192 if spec["holdout"] else None,
        }

        if spec["holdout"] and not args.unlock_holdout:
            rec["action"] = "downloaded_and_hashed_only"
            print(f"[holdout] {name}: downloaded and hashed; content not parsed")
        else:
            if spec["holdout"]:
                print("WARNING: holdout has been explicitly unlocked")
            dest = converted / f"odlyzko_{name}.npz"
            if spec["holdout"]:
                # Final-analysis unlock: parse the source only now and persist only
                # the predeclared unseen tail used as the strict holdout.
                lines = first_text_lines(src, spec["compressed"], n=7)
                verify_high_header(name, lines, int(spec["base"]))
                vals = np.asarray(numeric_lines(src, spec["compressed"]), dtype=np.float64)
                if vals.size != int(spec["count"]):
                    raise RuntimeError(f"{name}: expected {spec['count']} values, found {vals.size}")
                if not np.all(np.diff(vals) > 0):
                    raise RuntimeError(f"{name}: values are not strictly increasing")
                start = 1808
                tail = vals[start:]
                if tail.size != 8192:
                    raise RuntimeError(f"{name}: strict holdout tail size mismatch: {tail.size}")
                dest = converted / "odlyzko_1e22_strict_tail8192.npz"
                if dest.exists():
                    raise FileExistsError(f"refusing to overwrite converted holdout: {dest}")
                np.savez(
                    dest,
                    base=str(spec["base"]),
                    x=tail,
                    start_index=str(int(spec["start_index"]) + start),
                    source_url=spec["url"],
                    source_sha256=sha256_file(src),
                    source_offset_start=np.asarray(start, dtype=np.int64),
                )
                conv = {
                    "converted": str(dest),
                    "n": int(tail.size),
                    "offset_first": float(tail[0]),
                    "offset_last": float(tail[-1]),
                }
            else:
                conv = convert_table(name, src, dest)
            rec["action"] = "downloaded_hashed_converted"
            rec.update(conv)
            print(f"[converted] {name}: n={conv['n']}")

        records.append(rec)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = manifest_dir / f"riemann_acquisition_{stamp}.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "terj-riemann-acquisition-v1",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "records": records,
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"manifest: {manifest}")


if __name__ == "__main__":
    main()
