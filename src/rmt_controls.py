#!/usr/bin/env python3
"""Production RMT controls for TERJ Cross-Spectral Stage A.

This implementation follows manifests/rmt_matching_v1.json and separates
matrix dimension from the number of analyzed spectral levels.

Primary nulls:
- Riemann: GUE beta=2 bulk window
- Neural: real Wishart/Laguerre beta=1 central 50%

Sensitivity/universality controls:
- CUE beta=2
- GOE beta=1 bulk window
- Poisson non-repulsive control

Author: Paweł Majsterek
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np

ENSEMBLES = ("gue", "goe", "wishart", "cue", "poisson")


def eig_gue(rng: np.random.Generator, n: int) -> np.ndarray:
    z = (
        rng.normal(size=(n, n))
        + 1j * rng.normal(size=(n, n))
    ) / math.sqrt(2 * n)
    h = (z + z.conj().T) / 2
    return np.linalg.eigvalsh(h).real


def eig_goe(rng: np.random.Generator, n: int) -> np.ndarray:
    a = rng.normal(size=(n, n)) / math.sqrt(n)
    h = (a + a.T) / 2
    return np.linalg.eigvalsh(h)


def eig_wishart(rng: np.random.Generator, n: int) -> np.ndarray:
    x = rng.normal(size=(n, n)) / math.sqrt(n)
    return np.linalg.eigvalsh(x.T @ x)


def eig_cue(rng: np.random.Generator, n: int) -> np.ndarray:
    z = (
        rng.normal(size=(n, n))
        + 1j * rng.normal(size=(n, n))
    ) / math.sqrt(2)
    q, r = np.linalg.qr(z)
    ph = np.diag(r)
    ph = np.where(np.abs(ph) > 0, ph / np.abs(ph), 1)
    u = q * ph.conj()
    return np.sort(np.mod(np.angle(np.linalg.eigvals(u)), 2 * np.pi))


def eig_poisson(rng: np.random.Generator, n: int) -> np.ndarray:
    return np.cumsum(rng.exponential(size=n))


GEN = {
    "gue": eig_gue,
    "goe": eig_goe,
    "wishart": eig_wishart,
    "cue": eig_cue,
    "poisson": eig_poisson,
}


def atomic_npy(path: Path, arr: np.ndarray) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".partial",
        dir=path.parent,
    )
    os.close(fd)
    p = Path(tmp)
    try:
        with p.open("wb") as f:
            np.save(
                f,
                np.asarray(arr, dtype=np.float64),
                allow_pickle=False,
            )
            f.flush()
            os.fsync(f.fileno())
        os.replace(p, path)
    finally:
        if p.exists():
            p.unlink()


def central_slice(levels: np.ndarray, analysis_size: int) -> np.ndarray:
    n = len(levels)
    if analysis_size <= 0 or analysis_size > n:
        raise ValueError("invalid analysis_size")
    if (n - analysis_size) % 2 != 0:
        raise ValueError(
            "matrix_size-analysis_size must be even for symmetric central slice"
        )
    lo = (n - analysis_size) // 2
    hi = lo + analysis_size
    out = np.asarray(levels[lo:hi], dtype=np.float64)
    if len(out) != analysis_size:
        raise RuntimeError("central slice length mismatch")
    return out


def cue_central_arc(levels: np.ndarray, analysis_size: int) -> np.ndarray:
    """Select a deterministic contiguous arc away from the wrap boundary.

    CUE has no physical spectral edge, but its eigenangles live on a circle.
    Taking the central sorted arc avoids introducing the 0/2pi cut into the
    linearized feature vector.
    """
    return central_slice(levels, analysis_size)


def choose_matrix_size(
    ensemble: str,
    analysis_size: int,
    matrix_size: int | None,
) -> int:
    if ensemble == "poisson":
        return analysis_size if matrix_size is None else matrix_size
    if matrix_size is None:
        return 2 * analysis_size
    if matrix_size < analysis_size:
        raise ValueError("matrix_size must be >= analysis_size")
    return matrix_size


def extract_analysis_levels(
    ensemble: str,
    full_levels: np.ndarray,
    analysis_size: int,
) -> np.ndarray:
    if ensemble in {"gue", "goe", "wishart"}:
        return central_slice(full_levels, analysis_size)
    if ensemble == "cue":
        return cue_central_arc(full_levels, analysis_size)
    if ensemble == "poisson":
        if len(full_levels) == analysis_size:
            return np.asarray(full_levels, dtype=np.float64)
        return central_slice(full_levels, analysis_size)
    raise ValueError(ensemble)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate finite-size matched RMT control spectra"
    )
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--ensemble", choices=ENSEMBLES, required=True)
    ap.add_argument(
        "--analysis-size",
        type=int,
        required=True,
        help="number of levels saved and passed to the common feature extractor",
    )
    ap.add_argument(
        "--matrix-size",
        type=int,
        default=None,
        help=(
            "full random-matrix dimension; defaults to 2*analysis-size "
            "for matrix ensembles and analysis-size for Poisson"
        ),
    )
    ap.add_argument("--reps", type=int, default=5000)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    if args.analysis_size < 12:
        raise ValueError("analysis-size must be >= 12")
    if args.reps < 1:
        raise ValueError("reps must be >= 1")

    root = Path(args.workspace).resolve()
    matrix_size = choose_matrix_size(
        args.ensemble,
        args.analysis_size,
        args.matrix_size,
    )

    if args.ensemble in {"gue", "goe", "wishart", "cue"}:
        if (matrix_size - args.analysis_size) % 2 != 0:
            raise ValueError(
                "matrix-size minus analysis-size must be even"
            )

    out = (
        root
        / "controls"
        / args.ensemble
        / f"matrix{matrix_size}_analysis{args.analysis_size}"
    )

    if out.exists() and any(out.iterdir()):
        raise RuntimeError(
            f"refusing existing populated dir: {out}"
        )
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    for i in range(args.reps):
        full = GEN[args.ensemble](rng, matrix_size)
        levels = extract_analysis_levels(
            args.ensemble,
            full,
            args.analysis_size,
        )
        if not np.all(np.diff(levels) > 0):
            raise RuntimeError(
                f"{args.ensemble}: analysis levels not strictly increasing"
            )
        atomic_npy(
            out / f"sample_{i:05d}.npy",
            levels,
        )
        if (i + 1) % 100 == 0:
            print(
                f"{args.ensemble} "
                f"matrix={matrix_size} "
                f"analysis={args.analysis_size}: "
                f"{i + 1}/{args.reps}",
                flush=True,
            )

    lo = (matrix_size - args.analysis_size) // 2
    hi = lo + args.analysis_size

    manifest = {
        "schema": "terj-rmt-controls-v2",
        "ensemble": args.ensemble,
        "matrix_size": matrix_size,
        "analysis_size": args.analysis_size,
        "slice": [lo, hi],
        "reps": args.reps,
        "seed": args.seed,
        "saved_levels_only": True,
        "full_matrices_saved": False,
    }

    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        "done",
        out,
        f"(matrix={matrix_size}, analysis={args.analysis_size})",
    )


if __name__ == "__main__":
    main()
