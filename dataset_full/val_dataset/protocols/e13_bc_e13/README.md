# E13 / BC_E13 validation boxes

These are the immutable fixed-96 generation-mask inputs used by the successful
E13 and BC_E13 validation runtimes:

- `pm96_bboxes_new.json`: canonical manual protocol, SHA-256
  `a39645e22b68027175946a028e185b7c5393a7514f5d68c94cd74e7cc9f5e614`;
- `pm96_bboxes_new_auto.json`: historical E13/BC_E13 automatic cache,
  SHA-256
  `4db6344d0deb0af0ee7a25d839b774c9a4a0c5b8f6ff4cc00aaa9c0d6d85c099`.

Both JSON objects contain 96 keys. `Reading pa_jensen.png` is deliberately
marked `force_manual` in the manual protocol, so the trainer uses that manual
box and the remaining 95 routes use the automatic cache. Do not regenerate or
normalize these files; either action can change validation generations.
