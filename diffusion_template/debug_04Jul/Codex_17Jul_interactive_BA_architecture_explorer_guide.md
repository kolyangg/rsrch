# Interactive BA architecture explorer: usage and extension guide

Date: 17 July 2026

## Artifact

Open:

`debug_04Jul/ba_architecture_explorer/index.html`

The initial comparison is N3a versus the proposed NN1b:

- N3a is the runnable full-spatial branched-attention baseline now restored on
  `main_clean`.
- NN1b is the recommended architecture-preserving stability proposal: both
  branched processors remain active, but branched cross-attention weights are
  frozen.

The selectors also support:

- the original pre-numbered `cosm_new1` spatial run;
- N1 / N2 (`start_ba_ref_only_vast_N1.sh`, whose Comet run name is
  `ba_refonly_N2`);
- NN1a, NN1b, and NN1c as explicitly unimplemented proposals;
- N31, N32, N33, N36, N37, and N38 as historical post-N3a experiments.

## Recommended way to open it

From the repository root:

```bash
cd /home/kolyangg/rsrch/diffusion_template
python3 -m http.server 8765
```

Then open:

```text
http://127.0.0.1:8765/debug_04Jul/ba_architecture_explorer/
```

Opening `index.html` directly with `file://` also renders the diagrams, but a
local HTTP server makes relative source links behave more consistently.

No package installation, build step, or external web resource is required.

## Interaction

- Choose either comparison configuration from the two selectors.
- Enable **Highlight differences** to treat the left run as the baseline and
  mark changed configuration fields, architecture blocks, connections, and
  Q/K/V processor routes on the right.
- Click a diagram block to see its role, run-specific facts, and relevant code.
- Click an arrow to inspect the implementation of that connection.
- Use `⇄ Swap` to reverse the comparison.
- Click a row in the all-runs table to load it into the right diagram.
- The selected pair is stored in URL parameters:
  `?left=N3a&right=NN1b`.
- Difference mode is shareable with `diff=1`, for example:
  `?left=N3a&right=NN1b&diff=1`.
- `Enter` and `Space` activate focused diagram elements; `Escape` closes the
  inspector.

The diagrams intentionally distinguish:

- the absolute PhotoMaker epsilon path;
- legacy full-grid reference latents versus compact reference identity memory;
- detailed self- and cross-attention Q/K/V routing;
- target-coordinate BA residual attention;
- hard target-bbox localization;
- PM/BA epsilon composition;
- training-only supervision.

This distinction is important because N36-N38 are called "identity owner" in
their configs, but PhotoMaker still owns the absolute prediction in their
post-CFG composition.

### How difference highlighting is determined

The right-side highlight is a semantic configuration comparison, not a literal
text diff. It compares:

- reference/identity memory and preprocessing;
- U-Net and processor topology;
- PhotoMaker and face-prompt context;
- branched self-attention trainability;
- branched cross-attention route and trainability;
- active site counts, gates, and allowlists;
- mask routing;
- PM/BA epsilon composition;
- denoising schedule;
- training objective;
- recorded face-MAE and identity metrics.

This avoids highlighting an unchanged block merely because two run names or
descriptions use different wording. The legend above the diagrams lists every
changed semantic category. Selecting the same run on both sides produces no
highlights.

## Branch-aware source links

The repository is intentionally split:

- local links for Initial, N1/N2, N3a, and NN1 proposals resolve against
  `main_clean`, whose active model surface is the runnable N3a baseline;
- N31-N38 model and launcher links open the corresponding files on the pushed
  `main_clean_exp` GitHub branch, because those implementations are not part of
  the active N3a baseline;
- NN1 proposal links open
  `Jul_new_exp/2026-07-17_NN1_architecture_and_experiment_options.md`.

The archived N31-N38 launchers/configs under
`Jul_new_exp/archived_post_n3a_examples/` are documentary examples only.

## Historical reconstruction: Initial, N1/N2, and N3a

The active `main_clean` model/training/runtime surface is restored exactly to
the runnable N3a commit `e42c966`. Commit `2157ead` remains useful as the
earlier topology anchor, but it predates N3a's complete launcher, optimizer
grouping, unconditional face-prompt fix, and post-validation processor
reattachment.

The explorer therefore uses three evidence layers:

1. Commit `2157eada14824d14019e80f9416e6d736c837306` establishes the original
   doubled-latent runtime and exact spatial Q/K/V processor contract.
2. Commit `9b0dc27` and the `cosm_new1` launchers establish the original run.
3. Commits `ef04716` and `e42c966`, plus the preserved launch scripts, establish
   the N1/N2 and N3a run-specific weight modes, losses, and optimizer settings.

Useful source commands:

```bash
git show 2157eada14824d14019e80f9416e6d736c837306:diffusion_template/src/model/photomaker_branched/branched_runtime.py
git show 2157eada14824d14019e80f9416e6d736c837306:diffusion_template/src/model/photomaker_branched/attn_processor_cleanest.py
git show e42c966:diffusion_template/serv_new_runs/start_ba_nr_alt_vast_N3a.sh
```

The key historical distinction is structural:

- Initial, N1/N2, and N3a VAE-encode the full reference, noise it to the current
  timestep, concatenate `[target, reference]`, and execute one doubled U-Net.
- At every legacy self-attention site, target background queries use target K/V
  while target face queries use masked reference-grid K/V.
- Legacy cross-attention conditions the target half with the generation prompt
  and the reference half with the ID-only face prompt.
- The target half of this doubled prediction is returned directly. There is no
  separate ordinary PhotoMaker epsilon pass or outer hard PhotoMaker restoration.
- N31 and later compact-residual runs instead keep target coordinates in one
  stream, reduce the reference to identity tokens, and use those tokens only as
  K/V for a localized correction.

### Which branched processors are actually active?

The processor class names existing somewhere in Git history does not mean both
were installed for every configuration:

- Initial, N1/N2, and N3a install `BranchedAttnProcessor` at all 70 `attn1`
  self-attention sites and `BranchedCrossAttnProcessor` at all 70 `attn2`
  cross-attention sites.
- In the `main_clean_exp` implementation, N31–N38 inherit
  `model.ba_sa_mode: standard` from
  `one_id_ba_idtoken_ca_residual_N28.yaml`. In
  `patch_unet_attention_processors`, this sets `disable_sa`, and every `attn1`
  slot retains its original processor. `BranchedAttnProcessor` is therefore
  imported but is not installed or executed in these runs.
- N31–N33 do install `BranchedCrossAttnProcessor` in
  `target_face_residual` mode at all 70 `attn2` sites.
- N36–N38 install the same cross-attention processor at only 16 allowlisted
  `attn2` sites; the other 54 retain their original processors.

Consequently, N31–N38 are compact cross-attention-residual architectures, not
the full branched self-plus-cross-attention mechanism shown in the original
plan. The explorer now labels this explicitly and makes the standard `attn1`
block clickable so the inherited toggle and runtime branch can be inspected.
This distinction does not by itself explain why N36–N38 are flatter than
N31–N33: both groups have standard `attn1`. Within this compact family, the
more relevant route change is 70 active cross-attention sites in N31–N33 versus
16 in N36–N38, together with their changed gates, PM context, composition, and
objectives.

The detailed SVGs deliberately follow the visual grammar of
`/home/kolyangg/rsrch/_ba_scheme/ba_original_plan.pdf`: explicit `Q`, `K`, and
`V` projection blocks, attention outputs, mask multiplication, and merge arrows.

## Proposed NN1 records

NN1a/b/c are visualization and design records only. No NN1 launcher, Hydra
config, identity-loss backport, or model change has been created.

- NN1a: exact N3a two-GPU DDP parity control; branched SA and CA both train.
- NN1b: full spatial BA with branched SA training and branched CA
  forward-active but frozen.
- NN1c: NN1b plus a proposed decoded reference-identity loss at `t <= 400`.

All three retain the doubled `[target, reference]` U-Net batch, all 70
`BranchedAttnProcessor` sites, all 70 `BranchedCrossAttnProcessor` sites, full
reference-grid K/V, and direct target-half epsilon output. This is the main
architectural boundary the visualization is intended to protect.

## Files

| File | Purpose |
|---|---|
| `index.html` | semantic page structure and inspector panel |
| `styles.css` | original-plan-inspired teal diagram styling and responsive layout |
| `app.js` | run records, code snippets, SVG renderer, selectors, and interactions |

## Adding a new configuration

The explorer is data-driven but not fully automatic. Hydra inheritance,
checkpoint continuation, script overrides, and historical commit state make a
pure YAML parser unreliable for architecture semantics. Add one explicit
record after resolving the effective config.

In `app.js`, add a new entry to `CONFIGS`:

```javascript
N39: {
  short: "N39",
  title: "Short human-readable name",
  subtitle: "One-sentence result or experiment intent.",
  family: "pre-N34",
  topology: "compact_residual", // omit for compact residual; use legacy_spatial only with evidence
  status: "active", // active | mixed | failed | proposed
  statusLabel: "Anchor run",

  memory: {
    label: "8 face-patch tokens",
    detail: "What creates the tokens",
    tokens: 8,
  },

  sites: {
    count: 70,
    effective: 70,
    label: "All 70 CA sites",
    detail: "Gate/layer summary",
  },

  pmContext: "Full PM identity at all sites",
  composition: "Full composition description",
  compositionShort: "pre-CFG hard merge",
  objective: "Full training objective",
  objectiveShort: "diffusion + decoded ID",
  schedule: "PM at 10 · BOTH at 15",

  faceMae: 0.0, // use null if no comparable fixed-set measurement exists
  idScore: 0.0, // use null if unavailable
  metricStep: "step 0",
  architectureNote: "The central interpretation.",

  details: {
    memory: { /* inspector record */ },
    sites: { /* inspector record */ },
    objective: { /* inspector record */ },
    compose: { /* inspector record */ },
  },
},
```

Each inspector record has this form:

```javascript
{
  title: "Inspector title",
  description: "What this element does and why it matters.",
  facts: {
    "Fact label": "Value",
  },
  code: [
    code(
      "../../src/path/to/file.py",
      123,
      `short, relevant code excerpt`,
    ),
  ],
}
```

Only four run-specific inspector records are usually needed:

- `memory`
- `sites`
- `objective`
- `compose`

The shared input, target, PhotoMaker, residual, mask, scheduler, and connection
records come from `COMMON_DETAILS`.

## How to resolve a new run accurately

Before adding its record:

1. Resolve the full Hydra config, including parent defaults and launch-script
   overrides.
2. Determine `ba_identity_memory_mode`, token count, and image preprocessing.
3. Determine whether `BranchedAttnProcessor`, `BranchedCrossAttnProcessor`, or
   both are actually installed. Do not infer this merely because both classes
   are imported; resolve `ba_sa_mode`, `ba_ca_mode`, and disable toggles.
4. Determine whether the run is a compact target residual or a full spatial
   doubled-latent topology. Historical spatial entries require commit evidence,
   not only a current YAML.
5. Count actual patched/trainable `attn1` and `attn2` processors from the startup
   log.
6. Record per-layer gate initializations and calculate effective gate sites:
   `sum(gate_i)` across selected processors.
7. Determine whether composition is legacy pre-CFG hard merge or post-CFG
   conditional delta. For legacy spatial runs, verify whether there is any
   separate PhotoMaker pass at all.
8. Check whether PhotoMaker identity context is full, attenuated, or removed at
   each selected site.
9. Trace every loss that consumes `wrong_identity_pred`; do not assume the
   configured objective is the only objective applied.
10. Add fixed full-validation face MAE and identity score only after a complete
   96-image checkpoint exists.

Useful runtime checks:

```bash
rg -n \
  'BA Architecture|BA Validation|BA Runtime|Switch|manual_val/id_sim' \
  path/to/training.log
```

Historical config/code:

```bash
git show COMMIT:diffusion_template/src/configs/CONFIG.yaml
git diff COMMIT..HEAD -- diffusion_template/src/model/photomaker_branched
```

## Visual conventions

The style follows the older manual BA plan:

- dark teal boxes for core tensors/components;
- blue path for PhotoMaker;
- green path for BA identity;
- ochre path for target mask;
- dashed purple path for training-only supervision.

Keep the diagram focused on ownership and information flow. Add a new node only
when a run introduces a genuinely new state or transformation. A config-only
change normally belongs in an existing node's inspector and label.

## Validation after an update

Syntax-check JavaScript:

```bash
node --check debug_04Jul/ba_architecture_explorer/app.js
```

Serve the page and verify:

- both selectors redraw the correct labels and metrics;
- `?left=...&right=...` restores the pair;
- every block and arrow opens the inspector;
- source links point to the intended file/line;
- the page works at desktop and narrow widths;
- the all-runs matrix contains the new row.
