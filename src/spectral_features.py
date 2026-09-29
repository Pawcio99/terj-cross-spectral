#!/usr/bin/env python3
"""Shared scale-free spectral features for TERJ Cross-Spectral Stage A.

The same functions are used for Riemann zero levels, neural Gram eigenvalues,
and random-matrix controls. The primary feature layer does not use unfolding.

Author: Paweł Majsterek
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

FEATURE_SCHEMA = "terj-common-spectral-features-v1"
K_VALUES = (1, 2, 3, 4)


def _f(x: Any) -> float:
    y = float(x)
    if not math.isfinite(y):
        raise ValueError(f"non-finite feature value: {y!r}")
    return y


def validate_levels(levels: np.ndarray) -> np.ndarray:
    x = np.asarray(levels, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("levels must be a one-dimensional vector")
    if x.size < 12:
        raise ValueError("at least 12 ordered levels are required")
    if not np.all(np.isfinite(x)):
        raise ValueError("levels contain NaN or infinity")
    d = np.diff(x)
    if not np.all(d > 0.0):
        bad = int(np.count_nonzero(d <= 0.0))
        raise ValueError(f"levels must be strictly increasing; bad gaps={bad}")
    return x


def consecutive_gap_ratios(levels: np.ndarray) -> np.ndarray:
    """r_i=min(s_i,s_{i+1})/max(s_i,s_{i+1}), s_i=x_{i+1}-x_i."""
    x = validate_levels(levels)
    s = np.diff(x)
    den = np.maximum(s[:-1], s[1:])
    num = np.minimum(s[:-1], s[1:])
    return num / den


def nonoverlap_k_gap_ratios(levels: np.ndarray, k: int) -> np.ndarray:
    """Scale-free ratio of adjacent non-overlapping k-spacings.

    For k>=1 define
        a_i = x_{i+k}   - x_i
        b_i = x_{i+2k} - x_{i+k}
        r_i^(k) = min(a_i,b_i)/max(a_i,b_i)

    This definition is explicit and identical in every domain.
    """
    x = validate_levels(levels)
    if k < 1:
        raise ValueError("k must be >=1")
    if x.size < 2 * k + 2:
        raise ValueError(f"not enough levels for k={k}")
    a = x[k:-k] - x[:-2*k]
    b = x[2*k:] - x[k:-k]
    den = np.maximum(a, b)
    num = np.minimum(a, b)
    return num / den


def lag1_corr(v: np.ndarray) -> float:
    v = np.asarray(v, dtype=np.float64)
    if v.size < 3:
        return 0.0
    a, b = v[:-1], v[1:]
    sa, sb = float(a.std()), float(b.std())
    if sa == 0.0 or sb == 0.0:
        return 0.0
    return _f(np.corrcoef(a, b)[0, 1])


def summarize_ratio_vector(prefix: str, r: np.ndarray) -> dict[str, float | int]:
    r = np.asarray(r, dtype=np.float64)
    if r.ndim != 1 or r.size == 0:
        raise ValueError(f"{prefix}: empty ratio vector")
    q10, q25, q50, q75, q90 = np.quantile(r, [0.10, 0.25, 0.50, 0.75, 0.90])
    return {
        f"{prefix}_count": int(r.size),
        f"{prefix}_mean": _f(r.mean()),
        f"{prefix}_sd": _f(r.std(ddof=1)) if r.size > 1 else 0.0,
        f"{prefix}_q10": _f(q10),
        f"{prefix}_q25": _f(q25),
        f"{prefix}_median": _f(q50),
        f"{prefix}_q75": _f(q75),
        f"{prefix}_q90": _f(q90),
        f"{prefix}_frac_lt_0_05": _f(np.mean(r < 0.05)),
        f"{prefix}_frac_lt_0_10": _f(np.mean(r < 0.10)),
        f"{prefix}_frac_lt_0_20": _f(np.mean(r < 0.20)),
        f"{prefix}_lag1_corr": lag1_corr(r),
    }


def spectral_features(levels: np.ndarray) -> dict[str, Any]:
    """Compute the frozen no-unfolding Stage A common feature layer."""
    x = validate_levels(levels)
    out: dict[str, Any] = {
        "schema": FEATURE_SCHEMA,
        "n_levels": int(x.size),
    }

    r1 = consecutive_gap_ratios(x)
    out.update(summarize_ratio_vector("r1", r1))

    for k in K_VALUES:
        rk = nonoverlap_k_gap_ratios(x, k)
        out.update(summarize_ratio_vector(f"rk{k}", rk))

    # A few raw-spacing diagnostics are kept for integrity only. They are not
    # primary cross-domain features because they depend on local scale.
    s = np.diff(x)
    out["integrity"] = {
        "n_spacings": int(s.size),
        "min_gap": _f(s.min()),
        "max_gap": _f(s.max()),
        "gap_dynamic_range": _f(s.max() / s.min()),
    }
    return out


def primary_feature_names() -> list[str]:
    """Explicit candidate vector used before null standardization.

    Count and raw-scale integrity fields are deliberately excluded.
    """
    names: list[str] = []
    for prefix in ("r1", "rk1", "rk2", "rk3", "rk4"):
        names.extend([
            f"{prefix}_mean",
            f"{prefix}_sd",
            f"{prefix}_q10",
            f"{prefix}_q25",
            f"{prefix}_median",
            f"{prefix}_q75",
            f"{prefix}_q90",
            f"{prefix}_frac_lt_0_05",
            f"{prefix}_frac_lt_0_10",
            f"{prefix}_frac_lt_0_20",
            f"{prefix}_lag1_corr",
        ])
    # r1 and rk1 are mathematically different definitions for k=1 only in
    # indexing convention? Here they are equivalent up to the same adjacent
    # gaps, so avoid duplicating them in the final vector.
    names = [n for n in names if not n.startswith("rk1_")]
    return names


def load_levels(path: str | Path, key: str | None = None) -> np.ndarray:
    p = Path(path)
    if p.suffix == ".npy":
        return np.load(p, allow_pickle=False).astype(np.float64)
    if p.suffix == ".npz":
        d = np.load(p, allow_pickle=False)
        if key is not None:
            return np.asarray(d[key], dtype=np.float64)
        for candidate in ("levels", "eigenvalues", "x"):
            if candidate in d.files:
                return np.asarray(d[candidate], dtype=np.float64)
        if len(d.files) == 1:
            return np.asarray(d[d.files[0]], dtype=np.float64)
        raise ValueError(f"{p}: specify --key; arrays={d.files}")
    raise ValueError("input must be .npy or .npz")


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute common scale-free spectral features")
    ap.add_argument("input")
    ap.add_argument("--key", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--print-primary", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)

    levels = load_levels(args.input, args.key)
    features = spectral_features(levels)
    features["primary_feature_names"] = primary_feature_names()
    out.write_text(json.dumps(features, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    if args.print_primary:
        print("\n".join(primary_feature_names()))
    print(f"wrote {out} ({len(levels)} levels)")


if __name__ == "__main__":
    main()
