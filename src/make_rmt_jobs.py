#!/usr/bin/env python3
"""Generate deterministic shell commands for frozen RMT production plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="manifests/rmt_production_v1.json")
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--out", default="manifests/rmt_jobs_v1.sh")
    args = ap.parse_args()

    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    total = 0

    for ens, cfg in plan["ensembles"].items():
        for a in plan["analysis_sizes"]:
            m = a if ens == "poisson" else 2 * a
            seed = int(cfg["seed_base"]) + a
            reps = int(cfg["reps_per_size"])
            total += reps

            lines += [
                f'echo "=== {ens} matrix={m} analysis={a} reps={reps} ==="',
                "python3 src/rmt_controls.py \\",
                '  --workspace "$WORKSPACE" \\',
                f"  --ensemble {ens} \\",
                f"  --matrix-size {m} \\",
                f"  --analysis-size {a} \\",
                f"  --reps {reps} \\",
                f"  --seed {seed} \\",
                "  --resume",
                "",
            ]

    if total != int(plan["total_samples"]):
        raise RuntimeError((total, plan["total_samples"]))

    out = Path(args.out)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out.chmod(0o755)

    print(
        f"wrote {out}: total_samples={total}, "
        f"jobs={len(plan['ensembles']) * len(plan['analysis_sizes'])}"
    )


if __name__ == "__main__":
    main()
