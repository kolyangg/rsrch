# CL39 24k identity-source crossing, inference seeds 1-3

This validation-only package repeats the accepted fixed-96 A/B/C/D crossing at
inference seeds `1`, `2`, and `3`, using one A100 per seed. The sole scientific
change from the accepted seed-0 crossing is
`datasets.val.manual_val.seeds=[seed]`.

The jobs use three independent byte-exact copies of the manifest-verified
historical CL39 crossing runtime from
`CL39_attribution_controls_20260826_r1/source_cross_historical` and the
immutable CL39 24k checkpoint. Independent copies prevent Hydra/debug outputs
from one seed mutating another seed's preflight source. Each arm fails unless
the composed seed is exact and all 96 images exist. Validation-only arms
inherit parent Comet key `b1ca0b3da679401c85b991f1bbdf0b2a` and intentionally
use the console writer.
