# Stage A protocol amendment 001 — strict Riemann holdout slice

Date: 2026-09-29  
Author: Paweł Majsterek

## Reason

During source verification for the Odlyzko 10^22 table, the file header and a
small prefix of numeric values were exposed while confirming the documented
base offset and accuracy statement.

Therefore the entire 10^22 table must not be described as completely unseen.

## Frozen correction

The strict numerical holdout is now the **tail 8192 zeros** of the Odlyzko
10^22 block:

- source block: zeros # 10^22+1 through 10^22+10^4,
- excluded prefix: first 1808 values,
- strict holdout slice: zero-table offsets 1808..9999 inclusive,
- holdout count: 8192.

No feature extraction, plots, descriptive statistics, or model-selection
decisions may use this tail before the final Stage A procedure is frozen.

The source file may be downloaded and cryptographically hashed before final
analysis. The acquisition tool must not parse the holdout file unless an
explicit unlock flag is supplied.

This amendment changes only the definition of the holdout. It does not change
the Stage A hypotheses, feature policy, null models, or primary endpoint.
