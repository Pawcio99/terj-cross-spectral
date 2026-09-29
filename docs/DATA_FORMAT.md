# Stage A data format

The program `cross_spectral.py` consumes one JSON file.

## Minimal schema

```json
{
  "schema": "terj-cross-spectral-v1",
  "features": ["mean_rtilde", "median_rtilde"],
  "groups": [
    {
      "domain": "riemann",
      "id": "R1",
      "observed": {
        "mean_rtilde": 0.60,
        "median_rtilde": 0.61
      },
      "null": {
        "mean_rtilde": {"mean": 0.599, "sd": 0.010},
        "median_rtilde": {"mean": 0.600, "sd": 0.012}
      }
    },
    {
      "domain": "neural",
      "id": "N1",
      "observed": {
        "mean_rtilde": 0.53,
        "median_rtilde": 0.53
      },
      "null": {
        "mean_rtilde": {"mean": 0.530, "sd": 0.012},
        "median_rtilde": {"mean": 0.531, "sd": 0.014}
      }
    }
  ]
}
```

## Required rules

- Every feature listed in `features` must occur in every group.
- Null standard deviations must be positive.
- Domains currently supported by the primary comparison are `riemann` and `neural`.
- Each group is one replication unit: a Riemann zero block or one neural-seed
  summary (or a predeclared seed aggregate).
- Raw individual gaps must not be entered as independent groups.
- Null statistics must be finite-size matched to the observed group.

## Output

The program writes JSON containing:

- null-standardized residual vectors,
- domain centroids,
- Euclidean centroid distance,
- energy distance,
- Gaussian-kernel MMD,
- feature-wise centroid differences,
- leave-one-feature-out sensitivity,
- deterministic Monte Carlo calibration of the centroid distance.

The Monte Carlo calibration is conditional on the supplied null means and
standard deviations. It does not account for uncertainty in those estimates
unless the input itself was constructed from an outer bootstrap.
