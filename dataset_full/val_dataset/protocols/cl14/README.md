# CL14 validation boxes

These are the immutable fixed-96 generation-mask inputs used by the CL14
validation runtime:

- `pm96_bboxes_new.json`: canonical manual protocol, SHA-256
  `a39645e22b68027175946a028e185b7c5393a7514f5d68c94cd74e7cc9f5e614`;
- `pm96_bboxes_new_auto.json`: historical CL10-CL14 automatic cache,
  SHA-256
  `b33cf02665cd875a738fba8f20c2ea95fcb0585358436e32691af0013a2f1c7d`.

Both JSON objects contain 96 keys. `Reading pa_jensen.png` is deliberately
marked `force_manual` in the manual protocol, so the trainer uses that manual
box and the remaining 95 routes use the automatic cache. Do not regenerate or
normalize these files; either action can change CL14 validation generations.
