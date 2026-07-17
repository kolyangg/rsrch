"use strict";

const PRE_N34 = "3c06eed7bb11744d87e2b816dc3a889808f051ba";
const LEGACY_SPATIAL = "2157eada14824d14019e80f9416e6d736c837306";
const N3A_COMMIT = "e42c966";
const EXP_SOURCE_ROOT =
  "https://github.com/kolyangg/rsrch/blob/main_clean_exp/diffusion_template";
const NN1_PROPOSAL =
  "../../Jul_new_exp/2026-07-17_NN1_architecture_and_experiment_options.md";

const code = (path, line, snippet, label = "Open source") => ({
  path,
  line,
  snippet,
  label,
});

const COMMON_DETAILS = {
  reference: {
    title: "Reference image",
    description:
      "The reference is never used as a target-coordinate spatial grid in this family. It is reduced to compact identity memory, which prevents direct copying of reference pose, hands, hair layout, and background.",
    facts: { Role: "Identity evidence", Coordinates: "Reference only" },
    code: [
      code(
        "../../src/model/photomaker_branched/lora2_helpers.py",
        386,
        `for i, (prompt, refs, bbox) in enumerate(...):\n    ...\n    faces = analyze_faces(model.face_analyzer, img_np)\n    embedding = torch.from_numpy(faces[0]["embedding"]).float()`,
      ),
    ],
  },
  target: {
    title: "Target noisy latent",
    description:
      "The same target latent and timestep feed both the ordinary PhotoMaker prediction and the BA-modified prediction. Target hidden-state queries retain pose, expression, head placement, and scene coordinates.",
    facts: { Shared: "PM and BA predictions", Geometry: "Target-owned" },
    code: [
      code(
        "../../src/model/photomaker_branched/lora2.py",
        688,
        `noise = torch.randn_like(latents)\ntimesteps = t_scalar.repeat(batch_size)\nnoisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)`,
      ),
    ],
  },
  prompt: {
    title: "PhotoMaker prompt context",
    description:
      "PhotoMaker injects its identity-conditioned prompt into the normal target path. Older N31-N33 runs retain this context at every residual site. N36-N38 attenuate it only inside their 16 selected BA processors; the external PM baseline remains fully identity-conditioned.",
    facts: { Baseline: "PhotoMaker V2", Risk: "PM identity remains dominant" },
    code: [
      code(
        "../../src/model/photomaker_branched/attn_processor_cleanest.py",
        983,
        `pm_context = encoder_hidden_states\nif self.ba_pm_identity_context_scale < 1.0:\n    pm_context = text_context + scale * (\n        encoder_hidden_states - text_context\n    )`,
      ),
    ],
  },
  pmPass: {
    title: "Ordinary PhotoMaker target pass",
    description:
      "A frozen/original-processor UNet pass produces the absolute epsilon baseline. This path is the source of the stable pose, background, lighting, and rendering that the hard merge protects.",
    facts: { Output: "Absolute PM epsilon", Trainability: "Frozen baseline" },
    code: [
      code(
        "../../src/model/photomaker_branched/lora2_helpers.py",
        706,
        `set_branched_training_mode(model, branched_active=False)\nwith torch.no_grad():\n    photomaker_pred = model.unet(\n        noisy_latents, timesteps,\n        encoder_hidden_states=prompt_embeds,\n    )[0]`,
      ),
      code(
        "../../src/pipelines/br_pipeline_helpers.py",
        1055,
        `if pipeline.ba_pm_preservation_mode == "hard_epsilon_merge":\n    set_validation_unet_mode(pipeline, branched_active=False)\n    photomaker_pred = pipeline.unet(...)[0]`,
      ),
    ],
  },
  baPass: {
    title: "BA-modified target pass",
    description:
      "Selected cross-attention processors run the normal target attention and add a compact-memory face residual. The query comes from target hidden states; only K/V come from reference identity memory.",
    facts: { Query: "Target hidden state", "Identity K/V": "Compact reference memory" },
    code: [
      code(
        "../../src/model/photomaker_branched/attn_processor_cleanest.py",
        1063,
        `query = attn.to_q(normalized)\nkey = _linear_forward(self.target_id_to_k, id_tokens)\nvalue = _linear_forward(self.target_id_to_v, id_tokens)\nface_hidden = F.scaled_dot_product_attention(query, key, value)`,
      ),
    ],
  },
  standardSelfAttention: {
    title: "Standard self-attention in N31–N38",
    description:
      "These compact-residual runs do not install BranchedAttnProcessor at attn1. They inherit ba_sa_mode=standard from N28, so the runtime keeps each original self-attention processor. The class is still imported and available in the source, but it is not on the active forward path. This separates the whole N31–N38 family from legacy spatial BA; it does not by itself explain N36–N38 versus N31–N33 because both groups share this setting.",
    facts: {
      "attn1 processor": "Original / standard",
      BranchedAttnProcessor: "Not installed",
      "Inherited toggle": "ba_sa_mode: standard",
    },
    code: [
      code(
        "../../src/configs/one_id_ba_idtoken_ca_residual_N28.yaml",
        15,
        `model:\n  ba_sa_mode: standard\n  ba_ca_mode: target_face_residual`,
        "Open inherited N28 architecture config",
      ),
      code(
        "../../src/model/photomaker_branched/branched_runtime.py",
        94,
        `disable_sa = ... or ba_sa_mode == "standard"\n...\nif name.endswith("attn1.processor"):\n    if disable_sa:\n        new_procs[name] = pipeline._original_attn_processors[name]`,
        "Open processor installation path",
      ),
    ],
  },
  mask: {
    title: "Hard target face bbox",
    description:
      "The target bbox gates each attention-resolution residual and the final epsilon merge. It keeps changes local, but it does not distinguish identity from expression, color, texture, or lighting inside the face.",
    facts: { Space: "Target image / latent", Resize: "Area-preserving + hard threshold" },
    code: [
      code(
        "../../src/model/photomaker_branched/branched_runtime.py",
        354,
        `mask = hard_mask[:, :1].to(...)\nmask = F.interpolate(mask, size=branched_pred.shape[-2:], mode="nearest")\nmask = (mask > 0).to(dtype=branched_pred.dtype)`,
      ),
    ],
  },
  residual: {
    title: "Target-face identity residual",
    description:
      "Identity attention is projected through a zero-initialized low-rank output adapter, multiplied by a gate, and added to the normal target attention only inside the target face mask.",
    facts: { Form: "PM attention + gated delta", Initialization: "Zero output projection" },
    code: [
      code(
        "../../src/model/photomaker_branched/attn_processor_cleanest.py",
        1073,
        `face_hidden = F.scaled_dot_product_attention(...)\ngate = self.effective_face_residual_gate()\nface_delta = self.face_delta_out(face_hidden) * gate\nface_delta = face_delta * mask_gate.squeeze(1)\nreturn pm_out + face_delta`,
      ),
    ],
  },
  output: {
    title: "Scheduler epsilon",
    description:
      "The merged prediction is passed to the normal diffusion scheduler. Repeating a small face correction over later denoising steps can change pixels outside the original crop indirectly, even though each epsilon merge is hard-masked.",
    facts: { Consumer: "Diffusion scheduler", Scope: "One denoising step" },
    code: [
      code(
        "../../src/pipelines/br_pipeline_helpers.py",
        1360,
        `latents = pipeline.scheduler.step(\n    noise_pred, t, latents, **extra_step_kwargs\n)[0]`,
      ),
    ],
  },
  memoryFlow: {
    title: "Reference → compact memory",
    description:
      "This connection is the identity information bottleneck. N31/N33 use two frozen PhotoMaker QFormer tokens, N32 uses eight learned face-patch queries, and N37 combines QFormer with canonical part tokens.",
    facts: { "Spatial grid": "No", "Target alignment": "Deferred to target queries" },
    code: [
      code(
        "../../src/model/photomaker_branched/lora2_helpers.py",
        470,
        `memory_mode = model.ba_identity_memory_mode\n...\nextracted_id_features = model.ba_identity_resampler(...)\n# or PhotoMaker QFormer tokens`,
      ),
    ],
  },
  pmFlow: {
    title: "PhotoMaker absolute path",
    description:
      "PhotoMaker supplies an absolute epsilon prediction, not merely pose features. Hard preservation guarantees this prediction outside the face and makes it the starting point inside the face too.",
    facts: { Authority: "Global and absolute", Safety: "High" },
    code: [
      code(
        "../../src/model/photomaker_branched/branched_runtime.py",
        354,
        `return photomaker_pred * (1.0 - mask) + branched_pred * mask`,
      ),
    ],
  },
  residualFlow: {
    title: "Compact memory → target-face K/V",
    description:
      "Reference memory becomes K/V while target hidden states remain Q. This is the key safety improvement over old spatial-reference BA, where reference and target coordinates were mixed directly.",
    facts: { Q: "Target", "K / V": "Reference identity tokens" },
    code: [
      code(
        "../../src/model/photomaker_branched/attn_processor_cleanest.py",
        1063,
        `query = attn.to_q(normalized)\nkey = self.target_id_to_k(id_tokens)\nvalue = self.target_id_to_v(id_tokens)`,
      ),
    ],
  },
  maskFlow: {
    title: "BBox localization",
    description:
      "The same target mask localizes attention residuals and the outer epsilon merge. It protects scene structure but cannot make an incorrectly supervised residual identity-specific.",
    facts: { Applied: "Per selected CA site + final merge", Boundary: "Hard" },
    code: [
      code(
        "../../src/model/photomaker_branched/attn_processor_cleanest.py",
        1080,
        `face_delta = self.face_delta_out(face_hidden) * gate\nface_delta = face_delta * mask_gate.squeeze(1)`,
      ),
    ],
  },
  trainFlow: {
    title: "Training-only supervision",
    description:
      "The objective updates memory, K/V, residual, and gate parameters. It is not present during validation, so inference depends entirely on what these parameters learned.",
    facts: { Inference: "Absent", "Gradient target": "BA trainables" },
    code: [
      code(
        "../../src/trainer/sdxl_trainers.py",
        452,
        `causal_identity_loss = batch.get("causal_identity_loss")\nif causal_identity_loss is not None:\n    batch["loss"] += weight * causal_identity_loss`,
      ),
    ],
  },
};

const CONFIGS = {
  Initial: {
    short: "Initial",
    title: "Original spatial noise + reference BA",
    subtitle:
      "The pre-numbered cosm_new1 run: strong reference transfer and target integration, but severe color, geometry, and prop drift.",
    family: "legacy spatial",
    topology: "legacy_spatial",
    status: "failed",
    statusLabel: "Original · artifact-prone",
    sourceCommit: "9b0dc27",
    memory: {
      label: "Full noised reference latent",
      detail: "VAE reference grid carried through a second U-Net batch half",
      tokens: null,
    },
    sites: legacySites("noise_and_ref"),
    weightMode: "noise_and_ref",
    pmContext: "Target prompt + ID-only reference-half prompt",
    composition: "One doubled BA U-Net pass; target half returned directly",
    compositionShort: "single-pass spatial replacement",
    objective: "Masked alternating diffusion MSE",
    objectiveShort: "alternating face/full MSE",
    schedule: "Text-only 0–9 · PM 10–14 · BA 15–49",
    faceMae: null,
    idScore: 0.2890,
    metricStep: "28k · 24 imgs",
    architectureNote:
      "BA has absolute authority at late steps; raw reference geometry improves integration in some poses but also transfers nuisance content.",
    details: legacyDetails("Initial", {
      sourceCommit: "9b0dc27",
      launcher: "../../serv_new_runs/start_ba_cosm_new1_vast.sh",
      launcherLine: 46,
      weightMode: "noise_and_ref",
      objective: "masked_alternating",
      optimizer: "LR 1e-4 · no gradient clipping · weight decay 0",
      result: "Matched 24-image ID mean ≈0.289 at epoch 14",
    }),
  },
  N1: {
    short: "N1 / N2",
    title: "Reference-only spatial BA",
    subtitle:
      "Called N1 by its launcher and analysis, but logged as ba_refonly_N2; cleaner content with weak face/body co-adaptation.",
    family: "legacy spatial",
    topology: "legacy_spatial",
    status: "mixed",
    statusLabel: "Cleaner · motion smear",
    sourceCommit: "ef04716",
    memory: {
      label: "Full noised reference latent",
      detail: "Same spatial reference grid; only reference-side branch clones train",
      tokens: null,
    },
    sites: legacySites("ref_only"),
    weightMode: "ref_only",
    pmContext: "Target prompt + ID-only reference-half prompt",
    composition: "One doubled BA U-Net pass; target half returned directly",
    compositionShort: "single-pass spatial replacement",
    objective: "Blended full-image and face diffusion MSE",
    objectiveShort: "blended masked MSE",
    schedule: "Text-only 0–9 · PM 10–14 · BA 15–49",
    faceMae: null,
    idScore: 0.3515,
    metricStep: "20k · 24 imgs",
    architectureNote:
      "Freezing target/noise clones removes much drift, but the inserted reference face does not co-adapt to difficult target motion and hair.",
    details: legacyDetails("N1 / N2", {
      sourceCommit: "ef04716",
      launcher: "../../serv_new_runs/start_ba_ref_only_vast_N1.sh",
      launcherLine: 4,
      weightMode: "ref_only",
      objective: "blended_masked · λface 0.2",
      optimizer: "LR 5e-5 · clip 1.0 · weight decay 1e-2",
      result: "Matched 24-image ID mean ≈0.352 at epoch 10",
      naming:
        "Launcher/comment: N1; saved analysis: ba_refonly1; writer.run_name: ba_refonly_N2",
    }),
  },
  N3a: {
    short: "N3a",
    title: "Damped spatial noise + reference BA",
    subtitle:
      "The original full-grid topology with optimizer hygiene; training still moves sharply away from the clean step-zero branch.",
    family: "legacy spatial",
    topology: "legacy_spatial",
    status: "failed",
    statusLabel: "Historical · destructive",
    sourceCommit: N3A_COMMIT,
    memory: {
      label: "Full noised reference latent",
      detail: "Spatial reference grid with crop jitter; no compact identity bottleneck",
      tokens: null,
    },
    sites: legacySites("noise_and_ref", 0.25),
    weightMode: "noise_and_ref · noise LR × 0.25",
    pmContext: "Target prompt + ID-only reference-half prompt",
    composition: "One doubled BA U-Net pass; target half returned directly",
    compositionShort: "single-pass spatial replacement",
    objective: "Masked alternating diffusion MSE",
    objectiveShort: "alternating face/full MSE",
    schedule: "Text-only 0–9 · PM 10–14 · BA 15–49",
    faceMae: 0.20616,
    idScore: 0.1710,
    metricStep: "10k · 96 imgs",
    architectureNote:
      "Very large face movement is not useful identity control: the spatial branch copies reference appearance and corrupts target-aligned content.",
    details: legacyDetails("N3a", {
      sourceCommit: N3A_COMMIT,
      launcher: "../../serv_new_runs/start_ba_nr_alt_vast_N3a.sh",
      launcherLine: 4,
      weightMode: "noise_and_ref · target/noise clone LR × 0.25",
      objective: "masked_alternating",
      optimizer: "LR 5e-5 · noise LR 1.25e-5 · clip 1.0 · weight decay 1e-2",
      result: "96-image ID 0.171; face MAE vs PM 0.206 at 10k",
    }),
  },
  NN1a: {
    short: "NN1a",
    title: "Exact N3a two-GPU control",
    subtitle:
      "Proposed reproducibility control: preserve the complete N3a forward and optimizer contract while changing only execution to two-GPU DDP.",
    family: "NN1 proposal",
    topology: "legacy_spatial",
    status: "proposed",
    statusLabel: "Proposal · not implemented",
    sourceCommit: N3A_COMMIT,
    memory: {
      label: "Full noised reference latent",
      detail: "Exact N3a spatial reference grid with crop jitter",
      tokens: null,
    },
    sites: legacySites("noise_and_ref", 0.25, true),
    weightMode: "noise_and_ref · SA/CA train · noise LR ×0.25",
    pmContext: "Target prompt + ID-only reference-half prompt",
    composition: "One doubled BA U-Net pass; target half returned directly",
    compositionShort: "exact N3a spatial pass",
    objective: "Masked alternating diffusion MSE",
    objectiveShort: "N3a masked alternating",
    schedule: "Text-only 0–9 · PM 10–14 · BA 15–49",
    faceMae: null,
    idScore: null,
    metricStep: "not run",
    architectureNote:
      "This is a parity gate, not the expected winner: NN1b/c should not be interpreted until NN1a reproduces N3a startup counts and step-zero behavior.",
    details: proposedNnDetails("NN1a", {
      line: 50,
      trainCa: true,
      weightMode: "noise_and_ref · target/noise clone LR ×0.25",
      objective: "masked_alternating",
      optimizer: "LR 5e-5 · noise LR 1.25e-5 · clip 1.0 · weight decay 1e-2",
      purpose: "Exact N3a forward and training control on two DDP processes",
      objectiveDescription:
        "Replay N3a's masked-alternating diffusion objective without an architectural change. Its purpose is to verify repository and DDP parity before testing improvements.",
    }),
  },
  NN1b: {
    short: "NN1b",
    title: "Stable full-BA self-attention training",
    subtitle:
      "Proposed stability anchor: both branched processors remain active, while only the spatial self-attention clones train.",
    family: "NN1 proposal",
    topology: "legacy_spatial",
    status: "proposed",
    statusLabel: "Recommended · not implemented",
    sourceCommit: N3A_COMMIT,
    memory: {
      label: "Full noised reference latent",
      detail: "N3a spatial reference grid without crop jitter",
      tokens: null,
    },
    sites: legacySites("noise_and_ref", 0.1, false),
    weightMode: "SA noise_and_ref · CA active/frozen",
    pmContext: "Target prompt + ID-only reference-half prompt",
    composition: "One doubled BA U-Net pass; target half returned directly",
    compositionShort: "full BA · frozen CA weights",
    objective: "Blended full-image and face diffusion MSE",
    objectiveShort: "blended masked λface 0.15",
    schedule: "Text-only 0–9 · PM 10–14 · BA 15–49",
    faceMae: null,
    idScore: null,
    metricStep: "not run",
    architectureNote:
      "This carries forward N11's strongest stability lesson without removing BranchedCrossAttnProcessor from the forward path.",
    details: proposedNnDetails("NN1b", {
      line: 77,
      trainCa: false,
      weightMode: "noise_and_ref for SA · CA forward active with frozen weights",
      objective: "blended_masked · λface 0.15",
      optimizer: "LR 1e-4 · noise LR 1e-5 · clip 1.0 · weight decay 1e-3",
      purpose: "Architecture-preserving stability anchor based on N11",
      objectiveDescription:
        "Use blended full-image and face diffusion MSE while training only branched self-attention clones. Cross-attention still splits target/reference prompts at all 70 sites, but its cloned weights are frozen.",
    }),
  },
  NN1c: {
    short: "NN1c",
    title: "Stable full BA plus identity supervision",
    subtitle:
      "Proposed highest-upside option: NN1b's stable full BA route plus a small low-timestep decoded identity loss.",
    family: "NN1 proposal",
    topology: "legacy_spatial",
    status: "proposed",
    statusLabel: "Highest upside · not implemented",
    sourceCommit: N3A_COMMIT,
    memory: {
      label: "Full noised reference latent",
      detail: "N3a spatial reference grid without crop jitter",
      tokens: null,
    },
    sites: legacySites("noise_and_ref", 0.1, false),
    weightMode: "SA noise_and_ref · CA active/frozen",
    pmContext: "Target prompt + ID-only reference-half prompt",
    composition: "One doubled BA U-Net pass; target half returned directly",
    compositionShort: "full BA · frozen CA weights",
    objective: "Blended diffusion MSE + decoded reference identity loss",
    objectiveShort: "blended + ID 0.1 at t≤400",
    schedule: "Text-only 0–9 · PM 10–14 · BA 15–49",
    faceMae: null,
    idScore: null,
    metricStep: "not run",
    architectureNote:
      "NN1c changes supervision, not attention math; the identity-loss plumbing is explicitly deferred until the experiment family is approved.",
    details: proposedNnDetails("NN1c", {
      line: 100,
      trainCa: false,
      weightMode: "noise_and_ref for SA · CA forward active with frozen weights",
      objective: "blended_masked + decoded ID 0.1 at t≤400",
      optimizer: "NN1b optimizer · identity target from trusted reference",
      purpose: "NN1b plus identity-directed supervision",
      objectiveDescription:
        "Keep NN1b's full branched-attention forward and add a flag-gated decoded face identity loss of 0.1 only for t≤400. No compact memory, residual route, allowlist, or alternate epsilon composition is introduced.",
    }),
  },
  N31: {
    short: "N31",
    title: "Identity-dependence QFormer",
    subtitle: "Strong branch use, but epsilon-ranking learns desaturation and expression shortcuts.",
    family: "pre-N34",
    status: "mixed",
    statusLabel: "Active · unsafe objective",
    memory: {
      label: "2 QFormer tokens",
      detail: "Frozen PhotoMaker identity tokens · full reference",
      tokens: 2,
    },
    sites: {
      count: 70,
      effective: 70,
      label: "All 70 CA sites",
      detail: "All target-face attn2 processors · unit scalar gate",
    },
    pmContext: "Full PM identity at all sites",
    composition: "Legacy pre-CFG hard epsilon merge",
    compositionShort: "pre-CFG hard merge",
    objective: "Diffusion + ID loss + wrong-reference epsilon ranking",
    objectiveShort: "epsilon correct/wrong rank",
    schedule: "PM at 10 · BOTH at 15",
    faceMae: 0.06683,
    idScore: 0.4640,
    metricStep: "2k",
    architectureNote: "Branch is active but optimizes nuisance cues rather than clean identity.",
    details: {
      memory: {
        title: "N31 QFormer memory",
        description:
          "N31 preserves the two distinct frozen PhotoMaker QFormer identity tokens. The memory is compact and safe, but it largely duplicates identity information already present in PhotoMaker.",
        facts: { Tokens: "2", Preprocessing: "Full reference" },
        code: [
          code(
            "../../src/configs/one_id_ba_qformer_idtokens_N29.yaml",
            8,
            `model:\n  ba_identity_token_count: 2\n  ba_identity_memory_mode: qformer_tokens\n  ba_identity_image_mode: full_reference`,
          ),
        ],
      },
      sites: allSiteDetail("N31"),
      objective: {
        title: "N31 epsilon identity-dependence objective",
        description:
          "Correct memory must predict target epsilon better than a wrong memory inside the face. The objective can be satisfied with color, contrast, or expression shortcuts and caused the visible grayscale drift.",
        facts: { Weight: "0.25", Margin: "0.02", Failure: "Nuisance shortcut" },
        code: [
          code(
            "../../src/configs/one_id_ba_identity_dependence_N31.yaml",
            7,
            `model:\n  ba_identity_dependence_mode: paired_wrong_reference\n  ba_identity_dependence_weight: 0.25\n  ba_identity_dependence_margin: 0.02`,
          ),
          code(
            "../../src/loss/diffusion_loss.py",
            35,
            `def identity_dependence_ranking_loss(...):\n    ...\n    return relu(margin + correct_face_loss - wrong_face_loss)`,
          ),
        ],
      },
      compose: legacyComposeDetail(),
    },
  },
  N32: {
    short: "N32",
    title: "Face-patch resampler",
    subtitle: "The strongest clean, visibly active pre-N34 branch; selected as the old-run anchor.",
    family: "pre-N34",
    status: "active",
    statusLabel: "Selected old anchor",
    memory: {
      label: "8 face-patch tokens",
      detail: "Learned InsightFace-conditioned queries over hard-bbox CLIP patches",
      tokens: 8,
    },
    sites: {
      count: 70,
      effective: 70,
      label: "All 70 CA sites",
      detail: "All target-face attn2 processors · unit scalar gate",
    },
    pmContext: "Full PM identity at all sites",
    composition: "Legacy pre-CFG hard epsilon merge",
    compositionShort: "pre-CFG hard merge",
    objective: "Diffusion + inherited low-timestep reference ID loss",
    objectiveShort: "diffusion + decoded ID",
    schedule: "PM at 10 · BOTH at 15",
    faceMae: 0.07763,
    idScore: 0.4453,
    metricStep: "16k",
    architectureNote: "Safe and active, but memory changes do not become consistently identity-improving.",
    details: {
      memory: {
        title: "N32 face-patch identity memory",
        description:
          "Eight learned queries use the 512-D face identity embedding to attend only CLIP patches whose centers fall inside the reference bbox. It carries more facial evidence than two QFormer tokens without a reference UNet grid.",
        facts: { Tokens: "8", Query: "InsightFace-conditioned", Patches: "Hard-bbox CLIP" },
        code: [
          code(
            "../../src/configs/one_id_ba_facepatch_resampler_N32.yaml",
            7,
            `model:\n  ba_identity_token_count: 8\n  ba_identity_memory_mode: face_patch_resampler\n  ba_identity_patch_padding: 0.0`,
          ),
          code(
            "../../src/model/photomaker_branched/identity_memory.py",
            169,
            `queries = self.query_proj(self.identity_norm(identity_embeds))\nattended, _ = self.cross_attn(\n    queries, patches, patches,\n    key_padding_mask=~patch_mask.bool(),\n)`,
          ),
        ],
      },
      sites: allSiteDetail("N32"),
      objective: {
        title: "N32 training objective",
        description:
          "N32 does not use N31's wrong-reference ranking. It inherits diffusion/masked loss and the low-timestep reference identity loss from the N28/N25 chain. The output remains active but identity improvement is not monotonic.",
        facts: { "Wrong-reference rank": "Off", "Direct causal swap test": "Absent" },
        code: [
          code(
            "../../src/configs/one_id_ba_idtoken_ca_residual_N28.yaml",
            15,
            `model:\n  ba_ca_mode: target_face_residual\n  ba_pm_preservation_mode: hard_epsilon_merge\n  id_loss_identity_source: reference`,
          ),
        ],
      },
      compose: legacyComposeDetail(),
    },
  },
  N33: {
    short: "N33",
    title: "Long QFormer continuation",
    subtitle: "Clean and stable, but more training of the two-token path remains PhotoMaker-like.",
    family: "pre-N34",
    status: "mixed",
    statusLabel: "Stable · plateaued",
    memory: {
      label: "2 QFormer tokens",
      detail: "Unchanged N29 memory continued beyond 10k",
      tokens: 2,
    },
    sites: {
      count: 70,
      effective: 70,
      label: "All 70 CA sites",
      detail: "All target-face attn2 processors · unit scalar gate",
    },
    pmContext: "Full PM identity at all sites",
    composition: "Legacy pre-CFG hard epsilon merge",
    compositionShort: "pre-CFG hard merge",
    objective: "Unchanged N29 diffusion + inherited ID loss",
    objectiveShort: "N29 continuation",
    schedule: "PM at 10 · BOTH at 15",
    faceMae: 0.06653,
    idScore: 0.4731,
    metricStep: "24k",
    architectureNote: "Parameter norms continue growing, but the identity representation is saturated.",
    details: {
      memory: {
        title: "N33 continued QFormer memory",
        description:
          "N33 resumes N29 without an architectural change. Its two PhotoMaker-derived tokens remain a safe but compressed identity bottleneck.",
        facts: { Tokens: "2", Initialization: "N29 10k checkpoint", Change: "Duration only" },
        code: [
          code(
            "../../serv_new_runs/start_ba_qformer_continue20k_serv_N33.sh",
            4,
            `# N33: continue unchanged N29 from 10k to 40k\nN29_CHECKPOINT=.../checkpoint-epoch5.pth`,
          ),
        ],
      },
      sites: allSiteDetail("N33"),
      objective: {
        title: "N33 unchanged objective",
        description:
          "No new identity-causal signal is introduced. The run tests whether duration alone can recover information missing from the two-token memory; the result says no.",
        facts: { "Architecture change": "None", Result: "Plateau" },
        code: [
          code(
            "../../src/configs/one_id_ba_qformer_continue20k_N33.yaml",
            1,
            `defaults:\n  - one_id_ba_qformer_idtokens_N29\n  - _self_`,
          ),
        ],
      },
      compose: legacyComposeDetail(),
    },
  },
  N36: {
    short: "N36",
    title: "Restricted identity-owner QFormer",
    subtitle: "Correctly wired at 4k-8k, but the 16-site residual remains too weak.",
    family: "post-N34",
    status: "failed",
    statusLabel: "Weak · failed",
    memory: {
      label: "2 QFormer tokens",
      detail: "Frozen PhotoMaker identity tokens · full reference",
      tokens: 2,
    },
    sites: restrictedSites(),
    pmContext: "54 full + 10 half + 6 text-only site equivalents",
    composition: "Guidance-scaled conditional delta added after CFG",
    compositionShort: "post-CFG delta × 5",
    objective: "Decoded causal loss + unintended epsilon ranking",
    objectiveShort: "causal + epsilon rank",
    schedule: "PM at 10 · BOTH at 15",
    faceMae: 0.03856,
    idScore: 0.4517,
    metricStep: "8k",
    architectureNote: "PhotoMaker remains the absolute owner; 16 selected sites cannot overcome 54 untouched sites.",
    details: restrictedDetails("N36", {
      title: "N36 QFormer memory",
      description:
        "The same two full-reference QFormer tokens as N29/N31/N33 feed the restricted identity-owner route.",
      facts: { Tokens: "2", "Image mode": "Full reference" },
      code: [
        code(
          "../../src/configs/one_id_ba_identity_owner_qformer_N36.yaml",
          1,
          `defaults:\n  - one_id_ba_causal_highres_qformer_N34\n...\n# inherited: qformer_tokens, token_count=2`,
        ),
      ],
    }),
  },
  N37: {
    short: "N37",
    title: "Restricted hybrid canonical memory",
    subtitle: "Ten richer tokens do not fix the downstream route; latest ID result is the worst.",
    family: "post-N34",
    status: "failed",
    statusLabel: "Richer memory · failed",
    memory: {
      label: "2 QFormer + 8 parts",
      detail: "Canonical face-part resampler appended to frozen QFormer tokens",
      tokens: 10,
    },
    sites: restrictedSites(),
    pmContext: "54 full + 10 half + 6 text-only site equivalents",
    composition: "Guidance-scaled conditional delta added after CFG",
    compositionShort: "post-CFG delta × 5",
    objective: "Decoded causal loss + unintended epsilon ranking",
    objectiveShort: "causal + epsilon rank",
    schedule: "PM at 10 · BOTH at 15",
    faceMae: 0.04620,
    idScore: 0.4472,
    metricStep: "8k",
    architectureNote: "Memory capacity is not useful while residual authority is restricted.",
    details: restrictedDetails("N37", {
      title: "N37 hybrid identity memory",
      description:
        "Two frozen QFormer tokens are concatenated with eight trainable canonical face-part tokens. The richer ten-token memory increases face movement slightly but does not improve identity.",
      facts: { Tokens: "10", Alignment: "Canonical landmarks / bbox fallback" },
      code: [
        code(
          "../../src/configs/one_id_ba_identity_owner_hybrid_N37.yaml",
          14,
          `model:\n  ba_identity_token_count: 10\n  ba_identity_memory_mode: qformer_plus_canonical_parts\n  ba_identity_canonical_size: 224`,
        ),
        code(
          "../../src/model/photomaker_branched/lora2_helpers.py",
          618,
          `canonical_tokens = model.ba_identity_resampler(...)\nqformer_tokens = torch.cat(canonical_qformer_list, dim=0)\nextracted_id_features = torch.cat(\n    [qformer_tokens, canonical_tokens], dim=1\n)`,
        ),
      ],
    }),
  },
  N38: {
    short: "N38",
    title: "Restricted cropped QFormer",
    subtitle: "Selected as the cleanest new-family representative; still only half N32's face movement.",
    family: "post-N34",
    status: "failed",
    statusLabel: "Selected new comparison",
    memory: {
      label: "2 cropped QFormer tokens",
      detail: "PhotoMaker QFormer over padded bbox-normalized reference crop",
      tokens: 2,
    },
    sites: restrictedSites(),
    pmContext: "54 full + 10 half + 6 text-only site equivalents",
    composition: "Guidance-scaled conditional delta added after CFG",
    compositionShort: "post-CFG delta × 5",
    objective: "Decoded causal loss + unintended epsilon ranking",
    objectiveShort: "causal + epsilon rank",
    schedule: "PM at 10 · BOTH at 15",
    faceMae: 0.03877,
    idScore: 0.4530,
    metricStep: "8k",
    architectureNote: "Cropping does not solve the weak route; PhotoMaker remains visually dominant.",
    details: restrictedDetails("N38", {
      title: "N38 cropped QFormer memory",
      description:
        "N38 changes only QFormer preprocessing relative to N36: a square bbox-centered crop with 15% padding reduces background/clothing leakage. Its output remains effectively as weak as N36.",
      facts: { Tokens: "2", "Image mode": "bbox_normalized", Padding: "0.15" },
      code: [
        code(
          "../../src/configs/one_id_ba_identity_owner_cropped_qformer_N38.yaml",
          8,
          `model:\n  ba_identity_token_count: 2\n  ba_identity_memory_mode: qformer_tokens\n  ba_identity_image_mode: bbox_normalized\n  ba_identity_crop_padding: 0.15`,
        ),
        code(
          "../../src/model/photomaker_branched/identity_memory.py",
          11,
          `def bbox_normalized_reference(image, bbox, padding=0.10):\n    ...\n    return image.crop((left, top, right, bottom))`,
        ),
      ],
    }),
  },
};

function legacySites(weightMode, noiseLrScale = null, caTrainable = true) {
  const training =
    weightMode === "ref_only"
      ? "reference-side clones train; target/noise projections stay at base"
      : noiseLrScale == null
        ? "reference and target/noise clones train at the same LR"
        : `reference clones train at base LR; target/noise clones at ×${noiseLrScale}`;
  const caTraining = caTrainable
    ? "branched CA weights train"
    : "branched CA forward remains active; its weights are frozen";
  return {
    count: 70,
    effective: 70,
    caTrainable,
    label: "70 SA + 70 CA sites",
    matrixLabel: "70 / 70 (+70 SA)",
    metricLabel: caTrainable ? "70 SA + 70 CA" : "70 SA train + 70 CA active/frozen",
    diagramLabel: caTrainable
      ? "70 SA + 70 CA processors"
      : "70 SA train · 70 CA active/frozen",
    effectiveLabel:
      weightMode === "ref_only"
        ? "70 ref-side"
        : caTrainable
          ? "70 spatial"
          : "70 SA train · CA frozen",
    detail: `All SDXL attn1 and attn2 processors · ${training} · ${caTraining}`,
  };
}

function legacyDetails(run, evidence) {
  const namingFacts = evidence.naming ? { Naming: evidence.naming } : {};
  const isProposal = Boolean(evidence.proposal);
  const caTrainable = evidence.trainCa !== false;
  const caTraining = caTrainable ? "Trainable" : "Forward active; weights frozen";
  return {
    history: {
      title: isProposal
        ? `${run}: proposal source and implementation status`
        : `${run}: source reconstruction`,
      description: isProposal
        ? "This is a visualized proposal only. It is grounded in the runnable N3a source now active on main_clean and the dated NN1 design document; no NN1 code, Hydra config, or launcher exists yet."
        : "main_clean now contains the runnable N3a-era spatial BA implementation. Commit 2157ead anchors the original doubled-latent topology; the run commit and launcher establish each historical optimizer and loss variant.",
      facts: {
        "Core topology": LEGACY_SPATIAL.slice(0, 8),
        "Run evidence": evidence.sourceCommit,
        "main_clean baseline": N3A_COMMIT,
        Status: isProposal ? "Not implemented" : "Historical run",
        ...namingFacts,
      },
      code: [
        code(
          "../../debug_04Jul/Codex_17Jul_interactive_BA_architecture_explorer_guide.md",
          72,
          `git show ${LEGACY_SPATIAL}:diffusion_template/src/model/photomaker_branched/branched_runtime.py\ngit show ${LEGACY_SPATIAL}:diffusion_template/src/model/photomaker_branched/attn_processor_cleanest.py`,
          "Open reconstruction notes",
        ),
        code(
          evidence.launcher,
          evidence.launcherLine,
          `# Run-specific evidence\nbranched_attn_weight_mode=${evidence.weightMode}\nloss_kind=${evidence.objective}\n# ${evidence.optimizer}`,
          "Open run launcher",
        ),
      ],
    },
    reference: {
      title: `${run}: full spatial reference image`,
      description:
        "The reference is not reduced to identity tokens. It is VAE-encoded as a full spatial latent, noised to the current target timestep, and carried through the U-Net as a second batch half. Reference pose, hair, clothing, lighting, and nearby objects therefore remain available to attention.",
      facts: {
        Representation: "Full VAE latent grid",
        Alignment: "Reference coordinates",
        Bottleneck: "None",
      },
      code: [
        code(
          "../../src/model/photomaker_branched/branched_runtime.py",
          403,
          `ref_noised = pipeline.scheduler.add_noise(\n    reference_latents,\n    pipeline._ref_noise[:reference_latents.shape[0]],\n    t_ref,\n)\nref_noised = pipeline.scheduler.scale_model_input(ref_noised, t_ref)`,
          "Open N3a runtime",
        ),
      ],
    },
    memory: {
      title: `${run}: noised reference latent`,
      description:
        "This is a spatial feature carrier rather than compact identity memory. It occupies the second half of the doubled U-Net batch and evolves through its own self- and cross-attention path.",
      facts: {
        Shape: "Same latent H×W as target",
        Timestep: "Matched to target",
        "Identity-only": "No",
      },
      code: [
        code(
          "../../src/model/photomaker_branched/branched_runtime.py",
          417,
          `# Create branched batch: [generation B, reference B].\nbatched_latents = torch.cat([latent_model_input, ref_noised], dim=0)\nt_batched = torch.cat([t_gen, t_ref], dim=0)`,
          "Open N3a runtime",
        ),
      ],
    },
    target: {
      title: `${run}: target noisy latent`,
      description:
        "The target latent forms the first half of the doubled batch. Its non-face hidden states remain target-owned, but face queries attend reference-grid K/V inside every branched self-attention layer.",
      facts: { "Batch half": "First", Geometry: "Target queries", Output: "Returned epsilon half" },
      code: [
        code(
          "../../src/model/photomaker_branched/attn_processor_cleanest.py",
          240,
          `noise_hidden = hidden_states[:batch_size]\nref_hidden = hidden_states[batch_size:]\nquery = self._q_noise(attn, noise_hidden)`,
          "Open BranchedAttnProcessor",
        ),
      ],
    },
    prompt: {
      title: `${run}: generation prompt`,
      description:
        "The target half of branched cross-attention uses the normal PhotoMaker identity-conditioned generation prompt. This controls the target/background stream.",
      facts: { Consumer: "Target-half cross-attention", Context: "PhotoMaker prompt" },
      code: [
        code(
          "../../src/model/photomaker_branched/attn_processor_cleanest.py",
          690,
          `# target/background half\nkey_bg = self._k_noise(attn, gen_prompt)\nvalue_bg = self._v_noise(attn, gen_prompt)\nhidden_bg = scaled_dot_product_attention(q_bg, key_bg, value_bg)`,
          "Open BranchedCrossAttnProcessor",
        ),
      ],
    },
    facePrompt: {
      title: `${run}: reference-half face prompt`,
      description:
        "The second U-Net half uses an ID-only face prompt in cross-attention. That conditioning updates the reference stream; target identity then reaches the target face mainly through spatial self-attention transfer.",
      facts: { Consumer: "Reference-half cross-attention", Mode: "ID-only" },
      code: [
        code(
          "../../src/model/photomaker_branched/branched_runtime.py",
          476,
          `face_prompt_mode = getattr(pipeline, "ba_face_prompt_mode", "id_only")\n...\nmasked_face_prompt_embeds = face_prompt_embeds * class_token_mask * id_scale`,
          "Open N3a face-prompt runtime",
        ),
        code(
          "../../src/model/photomaker_branched/attn_processor_cleanest.py",
          700,
          `# reference half\nkey_ref = self._k_ref(attn, face_prompt)\nvalue_ref = self._v_ref(attn, face_prompt)\nhidden_ref = scaled_dot_product_attention(q_ref, key_ref, value_ref)`,
          "Open BranchedCrossAttnProcessor",
        ),
      ],
    },
    mask: {
      title: `${run}: target and reference face masks`,
      description:
        "Separate target and reference bboxes select target face queries and reference face K/V. They localize transfer but do not geometrically align the two faces.",
      facts: {
        Target: "Selects target face queries / layer merge",
        Reference: "Selects reference-grid K/V",
        Alignment: "BBox only",
      },
      code: [
        code(
          "../../src/model/photomaker_branched/attn_processor_cleanest.py",
          327,
          `ref_mask = self._prepare_mask(self.mask_ref, seq_len, ref_batch_size)\nnoise_face_hidden = noise_hidden * target_mask\nref_face_hidden = ref_hidden * ref_mask`,
          "Open BranchedAttnProcessor masks",
        ),
      ],
    },
    baPass: {
      title: `${run}: single doubled U-Net pass`,
      description:
        "At active BA steps there is one U-Net call over [target, reference]. This is not a separate BA correction composed with an ordinary PhotoMaker prediction. It is the absolute prediction used for that denoising step.",
      facts: {
        "U-Net calls": "One at active BA step",
        Batch: "[target, reference]",
        Authority: "Absolute target-half epsilon",
      },
      code: [
        code(
          "../../src/model/photomaker_branched/branched_runtime.py",
          595,
          `noise_pred = pipeline.unet(\n    batched_latents,\n    t_batched,\n    encoder_hidden_states=torch.cat([prompt_embeds, face_prompt_embeds]),\n)[0]`,
          "Open N3a doubled U-Net call",
        ),
      ],
    },
    selfAttention: {
      title: `${run}: spatial branched self-attention`,
      description:
        "Target background queries attend target K/V. Target face queries attend K/V made from the masked reference spatial grid; pose adaptation is zero. The two absolute hidden outputs are hard-composed at every self-attention site, while the reference half continues independently.",
      facts: {
        Sites: "70 / 70 attn1",
        "Target face Q": "Target coordinates",
        "Face K/V": "Reference spatial grid",
        "Pose adapt": "0",
      },
      code: [
        code(
          "../../src/model/photomaker_branched/attn_processor_cleanest.py",
          336,
          `ref_face_hidden = ref_hidden * ref_mask\nface_hidden_mixed = ref_face_hidden  # pose ratio 0\nq_face = target_q * target_mask\nhidden_face = attention(q_face, K(ref_face), V(ref_face))`,
          "Open BranchedAttnProcessor face path",
        ),
        code(
          "../../src/model/photomaker_branched/attn_processor_cleanest.py",
          384,
          `merged = hidden_bg * (1 - target_mask) + hidden_face * target_mask\nhidden_states = torch.cat([merged, hidden_ref], dim=0)`,
          "Open BranchedAttnProcessor merge",
        ),
      ],
    },
    crossAttention: {
      title: `${run}: split cross-attention`,
      description:
        `All cross-attention sites process the target half with the generation prompt and the reference half with the face prompt. Cross-attention does not itself add a target-face residual; it conditions the two spatial streams that self-attention later couples. CA training state: ${caTraining.toLowerCase()}.`,
      facts: {
        Sites: "70 / 70 attn2",
        Target: "Generation prompt",
        Reference: "Face prompt",
        "CA weights": caTraining,
      },
      code: [
        code(
          "../../src/model/photomaker_branched/attn_processor_cleanest.py",
          690,
          `hidden_bg = attention(Q(target), K(gen_prompt), V(gen_prompt))\nhidden_ref = attention(Q(reference), K(face_prompt), V(face_prompt))\nhidden_states = torch.cat([hidden_bg, hidden_ref], dim=0)`,
          "Open BranchedCrossAttnProcessor",
        ),
      ],
    },
    sites: {
      title: `${run}: all spatial SA and CA sites`,
      description:
        "With ba_patch_top_k=1 and branched CA enabled, all 70 self-attention and all 70 cross-attention processors use the doubled spatial contract. The weight mode changes which cloned projections train, not which forward routes exist.",
      facts: {
        "Self-attention": "70 / 70",
        "Cross-attention": "70 / 70",
        "CA weights": caTraining,
        "Weight mode": evidence.weightMode,
      },
      code: [
        code(
          evidence.launcher,
          evidence.launcherLine,
          `branched_attn_weight_mode=${evidence.weightMode}\ntrain_branched_ca_lora=${caTrainable}\nba_patch_top_k=1.0\nba_train_top_k=1.0`,
          isProposal ? "Open proposal" : "Open run launcher",
        ),
      ],
    },
    compose: {
      title: `${run}: target half becomes the prediction`,
      description:
        "After the doubled U-Net, the first batch half is returned directly. The mask-based split below it is only for previews/diagnostics. There is no outer exact-PhotoMaker epsilon restoration comparable to N28-N38.",
      facts: {
        "Outer PM pass": "Absent",
        "Final hard epsilon merge": "Absent",
        Returned: "First / target batch half",
      },
      code: [
        code(
          "../../src/model/photomaker_branched/branched_runtime.py",
          629,
          `noise_pred_merged = noise_pred[:batch_size]\n...\nreturn noise_pred_merged, noise_face, noise_bg`,
          "Open N3a target-half return",
        ),
      ],
    },
    output: {
      title: `${run}: scheduler consumes absolute BA epsilon`,
      description:
        "The target half proceeds through CFG and the scheduler as the current step's model prediction. Its background is not guaranteed to equal an independent PhotoMaker baseline.",
      facts: { "PM preservation": "Only implicit in target branch", Scope: "Whole prediction" },
      code: [
        code(
          "../../src/pipelines/br_pipeline_helpers.py",
          1025,
          `if branched_active:\n    noise_pred, _, _ = run_branched_step(...)\nelse:\n    noise_pred = pipeline.unet(...)[0]`,
          "Open N3a validation route",
        ),
      ],
    },
    objective: {
      title: `${run}: ${evidence.objective}`,
      description:
        `The topology is supervised with ${evidence.objective}. ${evidence.optimizer}. This objective rewards denoising reconstruction, not reference identity causality. ${evidence.result}.`,
      facts: {
        Objective: evidence.objective,
        Optimizer: evidence.optimizer,
        Result: evidence.result,
      },
      code: [
        code(
          evidence.launcher,
          evidence.launcherLine,
          `loss_kind=${evidence.objective}\n# ${evidence.optimizer}`,
          isProposal ? "Open proposal" : "Open run launcher",
        ),
      ],
    },
    memoryFlow: {
      title: `${run}: reference image → spatial latent`,
      description:
        "VAE encoding preserves a dense spatial reference grid. Adding scheduler noise matches target signal level but does not remove pose or nuisance content.",
      facts: { Transform: "VAE encode + same-timestep noise", Output: "Spatial latent" },
      code: [
        code(
          "../../src/model/photomaker_branched/branched_runtime.py",
          403,
          `ref_noised = scheduler.add_noise(reference_latents, ref_noise, t_ref)`,
          "Open N3a runtime",
        ),
      ],
    },
    residualFlow: {
      title: `${run}: reference-grid face transfer`,
      description:
        "Target face queries attend masked reference-grid K/V at every self-attention layer. This gives high authority but exposes target generation to reference pose, hair, lighting, and objects.",
      facts: { Q: "Target face", "K / V": "Reference face grid", Form: "Absolute hidden output" },
      code: [
        code(
          "../../src/model/photomaker_branched/attn_processor_cleanest.py",
          336,
          `q_face = q * target_mask\nhidden_face = attention(q_face, K(ref_face_hidden), V(ref_face_hidden))`,
          "Open BranchedAttnProcessor",
        ),
      ],
    },
    maskFlow: {
      title: `${run}: bbox routing without canonical alignment`,
      description:
        "The two masks select face regions in different coordinate systems. They limit where transfer happens but cannot make reference features pose-compatible with target queries.",
      facts: { "Target mask": "Target coordinates", "Reference mask": "Reference coordinates" },
      code: [
        code(
          "../../src/model/photomaker_branched/attn_processor_cleanest.py",
          327,
          `target_mask = self._prepare_mask(self.mask, ...)\nref_mask = self._prepare_mask(self.mask_ref, ...)`,
          "Open BranchedAttnProcessor masks",
        ),
      ],
    },
    pmFlow: {
      title: `${run}: generation-prompt target stream`,
      description:
        "The target stream remains PhotoMaker-conditioned, but once BA is active it is part of the same doubled forward rather than a protected independent baseline.",
      facts: { Prompt: "PhotoMaker generation prompt", "Independent PM epsilon": "No" },
      code: [
        code(
          "../../src/model/photomaker_branched/attn_processor_cleanest.py",
          690,
          `hidden_bg = attention(Q(target), K(gen_prompt), V(gen_prompt))`,
          "Open BranchedCrossAttnProcessor",
        ),
      ],
    },
    trainFlow: {
      title: `${run}: training updates branch clones`,
      description:
        `The ${evidence.weightMode} mode selects which reference and target/noise projection clones receive gradients. The branch topology remains spatial in every case.`,
      facts: { "Weight mode": evidence.weightMode, Objective: evidence.objective },
      code: [
        code(
          "../../src/model/photomaker_branched/lora2_helpers.py",
          11,
          `mode = model.branched_attn_weight_mode\n# ref_only: train ref_to_*\n# noise_and_ref: train ref_to_* and noise_to_*`,
          "Open trainable selection",
        ),
      ],
    },
    scheduleFlow: {
      title: `${run}: denoising schedule switches absolute paths`,
      description:
        "Validation uses text-only steps 0–9, ordinary PhotoMaker steps 10–14, then the doubled BA pass from step 15. This is a temporal switch, not a simultaneous PM-plus-residual composition.",
      facts: { "Text-only": "0–9", PhotoMaker: "10–14", "Spatial BA": "15–49" },
      code: [
        code(
          "../../src/configs/pipeline/pm_br_09Feb_testing.yaml",
          11,
          `photomaker_start_step: 10\nmerge_start_step: 10\nbranched_attn_start_step: 15\nbranched_start_mode: both`,
          "Open schedule config",
        ),
      ],
    },
  };
}

function proposedNnDetails(run, evidence) {
  const details = legacyDetails(run, {
    sourceCommit: N3A_COMMIT,
    launcher: NN1_PROPOSAL,
    launcherLine: evidence.line,
    weightMode: evidence.weightMode,
    objective: evidence.objective,
    optimizer: evidence.optimizer,
    result: "Proposal only; no checkpoint or metric exists",
    proposal: true,
    trainCa: evidence.trainCa,
  });

  details.history.code = [
    code(
      NN1_PROPOSAL,
      evidence.line,
      `# ${run}: ${evidence.purpose}\n# Architecture proposal only; implementation awaits approval.`,
      "Open NN1 proposal",
    ),
    code(
      "../../Jul_new_exp/2026-07-17_branch_split_and_recovery.md",
      1,
      `main_clean      = runnable N3a behavioral baseline\nmain_clean_exp  = complete post-N3a experimental implementation`,
      "Open branch and recovery contract",
    ),
  ];
  details.objective.description = evidence.objectiveDescription;
  details.objective.facts = {
    Status: "Proposed; not implemented",
    Objective: evidence.objective,
    Optimizer: evidence.optimizer,
  };
  details.objective.code = [
    code(
      NN1_PROPOSAL,
      evidence.line,
      `# ${evidence.purpose}\nloss=${evidence.objective}\n# ${evidence.optimizer}`,
      "Open proposed experiment",
    ),
  ];
  return details;
}

function allSiteDetail(run) {
  return {
    title: `${run}: all 70 target-face CA sites`,
    description:
      "With no CA allowlist in the pre-N34 implementation, every SDXL cross-attention processor is replaced by the target-face residual processor. Self-attention remains standard. The scalar residual gate initializes to one at every site.",
    facts: {
      "CA processors": "70 / 70",
      "Gate equivalents": "70",
      "Historical code": PRE_N34.slice(0, 8),
    },
    code: [
      code(
        "../../src/model/photomaker_branched/branched_runtime.py",
        255,
        `elif name.endswith("attn2.processor"):\n    if disable_ca or not processor_name_matches_allowlist(...):\n        ...\n    else:\n        proc = BranchedCrossAttnProcessor(...)`,
      ),
    ],
  };
}

function restrictedSites() {
  return {
    count: 16,
    effective: 11,
    label: "16 of 70 CA sites",
    detail: "6 × gate 1.0 + 10 × gate 0.5 · 54 untouched PM sites",
  };
}

function legacyComposeDetail() {
  return {
    title: "Legacy pre-CFG hard epsilon merge",
    description:
      "The branch prediction is hard-merged with PhotoMaker before the pipeline's classifier-free guidance operation. A conditional BA delta therefore receives the same guidance multiplier as the PhotoMaker conditional prediction.",
    facts: { Composition: "Before CFG", "Outside bbox": "Exactly PM epsilon" },
    code: [
      code(
        "../../src/model/photomaker_branched/branched_runtime.py",
        354,
        `return photomaker_pred * (1.0 - mask) + branched_pred * mask`,
      ),
      code(
        "../../src/pipelines/br_pipeline_helpers.py",
        1152,
        `else:\n    noise_pred = hard_epsilon_merge(\n        photomaker_pred, noise_pred, mask4_for_merge\n    )`,
      ),
    ],
  };
}

function restrictedDetails(run, memoryDetail) {
  return {
    memory: memoryDetail,
    sites: {
      title: `${run}: restricted 16-site route`,
      description:
        "The allowlist selects six up_blocks.1 sites and ten late up_blocks.0.attentions.2 sites. Gates initialize to 1.0 and 0.5 respectively, giving only 11 unit-gate site equivalents versus N32's 70.",
      facts: {
        "CA processors": "16 / 70",
        "Unit-gate equivalents": "11",
        "Untouched PM sites": "54",
      },
      code: [
        code(
          "../../src/configs/one_id_ba_identity_owner_qformer_N36.yaml",
          15,
          `ba_ca_layer_allowlist:\n  - up_blocks.1\n  - up_blocks.0.attentions.2\n...\nba_face_gate_init: 1.0\nba_face_gate_init_overrides:\n  up_blocks.0.attentions.2: 0.5`,
        ),
        code(
          "../../src/model/photomaker_branched/branched_runtime.py",
          16,
          `def processor_name_matches_allowlist(name, allowlist):\n    ...\n    return any(name.startswith(pattern) for pattern in patterns)`,
        ),
      ],
    },
    objective: {
      title: `${run}: decoded causal plus inherited epsilon rank`,
      description:
        "The decoded correct/null/wrong objective reports essentially zero correct identity gain. In addition, the trainer sees wrong_identity_pred and also applies N31's epsilon ranking with inherited weight 0.25, unintentionally coupling both objectives.",
      facts: {
        "Causal weight": "0.5",
        "Low-timestep frequency": "≈30%",
        "Correct gain": "≈0",
        "Extra epsilon rank": "Implicitly on",
      },
      code: [
        code(
          "../../src/configs/one_id_ba_identity_owner_qformer_N36.yaml",
          38,
          `ba_causal_identity_weight: 0.5\nba_causal_margin: 0.05\nba_causal_direct_weight: 0.5`,
        ),
        code(
          "../../src/trainer/sdxl_trainers.py",
          477,
          `wrong_identity_pred = batch.get("wrong_identity_pred")\nif wrong_identity_pred is not None:\n    dependence_loss = identity_dependence_ranking_loss(...)\n    batch["loss"] += dependence_weight * dependence_loss`,
        ),
      ],
    },
    compose: {
      title: `${run}: guidance-scaled post-CFG correction`,
      description:
        "The fully guided PhotoMaker prediction is formed first. The conditional BA-minus-PM correction is then hard-masked and added with guidance gain five. This fixes the earlier missing-CFG-strength bug, but PhotoMaker remains the absolute baseline.",
      facts: {
        Composition: "After PM CFG",
        "Applied gain": "5",
        "Absolute owner": "PhotoMaker prediction",
      },
      code: [
        code(
          "../../src/model/photomaker_branched/branched_runtime.py",
          375,
          `pm_uncond, pm_cond = photomaker_pred.chunk(2)\n_, ba_cond = branched_pred.chunk(2)\ndelta_cond = hard_epsilon_merge(pm_cond, ba_cond, mask) - pm_cond\nguided = pm_uncond + guidance_scale * (pm_cond - pm_uncond)\nreturn guided + residual_scale * guidance_scale * delta_cond`,
        ),
      ],
    },
  };
}

function detailFor(config, key) {
  if (config.details && config.details[key]) return config.details[key];
  return COMMON_DETAILS[key] || {
    title: key,
    description: "No detail record has been added for this element yet.",
    facts: {},
    code: [],
  };
}

function topologyValue(config) {
  return config.topology === "legacy_spatial"
    ? "full spatial doubled [target, reference] BA"
    : "compact target-face residual BA";
}

function selfTrainingValue(config) {
  if (config.topology !== "legacy_spatial") {
    return "standard self-attention; BranchedAttnProcessor absent";
  }
  if (String(config.weightMode || "").startsWith("ref_only")) {
    return "BranchedAttnProcessor active; reference clones train, target/noise frozen";
  }
  return "BranchedAttnProcessor active; reference and target/noise clones train";
}

function crossTrainingValue(config) {
  if (config.topology !== "legacy_spatial") {
    return "target-face residual BranchedCrossAttnProcessor weights train";
  }
  return config.sites.caTrainable === false
    ? "BranchedCrossAttnProcessor forward active; cloned weights frozen"
    : "BranchedCrossAttnProcessor forward active; cloned weights train";
}

function objectiveOptimizerValue(config) {
  return detailFor(config, "objective").facts?.Optimizer || "not recorded";
}

const COMPARISON_GROUPS = [
  {
    id: "memory",
    label: "identity representation",
    keys: ["reference", "memory", "memoryFlow"],
    fields: [
      { label: "Representation", codeName: "identity_memory", value: (config) => config.memory.label },
      {
        label: "Tokenization",
        codeName: "identity_tokens",
        value: (config) =>
          config.memory.tokens == null ? "full spatial latent grid" : `${config.memory.tokens} tokens`,
      },
    ],
  },
  {
    id: "topology",
    label: "processor topology",
    keys: [
      "baPass",
      "selfAttention",
      "standardSelfAttention",
      "crossAttention",
      "residual",
      "residualFlow",
      "mask",
      "maskFlow",
    ],
    fields: [
      { label: "Topology", codeName: "ba_topology", value: topologyValue },
    ],
  },
  {
    id: "prompt-context",
    label: "prompt context",
    keys: ["prompt", "facePrompt", "pmFlow"],
    fields: [
      {
        label: "Prompt context",
        codeName: "pm_context",
        value: (config) => config.pmContext,
      },
    ],
  },
  {
    id: "self-training",
    label: "self-attention trainability",
    keys: ["selfAttention", "trainFlow"],
    fields: [
      {
        label: "Self-attention",
        codeName: "self_attention_training",
        value: selfTrainingValue,
      },
    ],
  },
  {
    id: "cross-training",
    label: "cross-attention trainability",
    keys: ["crossAttention", "sites"],
    fields: [
      {
        label: "Cross-attention",
        codeName: "cross_attention_training",
        value: crossTrainingValue,
      },
    ],
  },
  {
    id: "sites",
    label: "active sites / gates",
    keys: ["sites", "baPass", "residual", "residualFlow"],
    fields: [
      {
        label: "Active CA sites",
        codeName: "active_ca_sites",
        value: (config) => `${config.sites.count} / 70`,
      },
      {
        label: "Effective gated sites",
        codeName: "effective_ca_sites",
        value: (config) => String(config.sites.effective),
      },
    ],
  },
  {
    id: "composition",
    label: "PM / BA composition",
    keys: ["pmPass", "compose", "output", "pmFlow"],
    fields: [
      {
        label: "Composition",
        codeName: "epsilon_composition",
        value: (config) => config.composition,
      },
    ],
  },
  {
    id: "schedule",
    label: "denoising schedule",
    keys: ["scheduleFlow", "output"],
    fields: [
      {
        label: "Schedule",
        codeName: "denoising_schedule",
        value: (config) => config.schedule,
      },
    ],
  },
  {
    id: "objective",
    label: "training objective / optimizer",
    keys: ["objective", "trainFlow"],
    fields: [
      {
        label: "Objective",
        codeName: "training_objective",
        value: (config) => config.objective,
      },
      {
        label: "Optimizer",
        codeName: "optimizer",
        value: objectiveOptimizerValue,
      },
    ],
  },
];

function comparisonDifferences(left, right) {
  const changedGroups = [];
  const byKey = new Map();

  COMPARISON_GROUPS.forEach((group) => {
    const changes = group.fields
      .map((field) => ({
        label: field.label,
        codeName: field.codeName,
        left: String(field.value(left)),
        right: String(field.value(right)),
      }))
      .filter((change) => change.left !== change.right);
    if (changes.length === 0) return;

    changedGroups.push({ ...group, changes });
    group.keys.forEach((key) => {
      const existing = byKey.get(key) || [];
      byKey.set(key, [...existing, ...changes]);
    });
  });

  return {
    changedGroups,
    keys: new Set(byKey.keys()),
    byKey,
  };
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function node(x, y, w, h, key, title, subtitle, className = "") {
  const titleLines = Array.isArray(title) ? title : [title];
  const titleStart = y + h / 2 - (titleLines.length - 1) * 9 - (subtitle ? 7 : 0);
  const titleSvg = titleLines
    .map(
      (line, index) =>
        `<text x="${x + w / 2}" y="${titleStart + index * 18}">${escapeHtml(line)}</text>`,
    )
    .join("");
  const subtitleSvg = subtitle
    ? `<text class="small" x="${x + w / 2}" y="${y + h - 13}">${escapeHtml(subtitle)}</text>`
    : "";
  return `
    <g class="node clickable ${className}" data-inspect="${key}" tabindex="0" role="button">
      <rect x="${x}" y="${y}" width="${w}" height="${h}" rx="5"></rect>
      ${titleSvg}
      ${subtitleSvg}
    </g>`;
}

function edge(path, key, className, markerId) {
  return `
    <g class="edge-group clickable" data-inspect="${key}" tabindex="0" role="button">
      <path class="edge ${className}" d="${path}" marker-end="url(#${markerId}-${className || "base"})"></path>
      <path class="edge-hit" d="${path}"></path>
    </g>`;
}

function edgeLabel(x, y, text, className = "") {
  return `<text class="edge-label ${className}" x="${x}" y="${y}">${escapeHtml(text)}</text>`;
}

function markerDefs(markerId) {
  return ["base", "pm", "ba", "mask", "train"]
    .map(
      (kind) => `
      <marker id="${markerId}-${kind}" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
        <path d="M0,0 L8,4 L0,8 z" fill="${
          kind === "pm"
            ? "var(--pm)"
            : kind === "ba"
              ? "var(--ba)"
              : kind === "mask"
                ? "var(--mask)"
                : kind === "train"
                  ? "var(--train)"
                  : "var(--teal)"
        }"></path>
      </marker>`,
    )
    .join("");
}

function renderSvg(config, panelId) {
  if (config.topology === "legacy_spatial") {
    return renderLegacyOverviewSvg(config, panelId);
  }
  return renderCompactOverviewSvg(config, panelId);
}

function renderLegacyOverviewSvg(config, panelId) {
  const markerId = `arrow-${panelId}`;
  const caSubtitle =
    config.sites.caTrainable === false
      ? "forward active · cloned weights frozen"
      : "target↔gen · reference↔face";
  const siteLabel = config.sites.diagramLabel || "70 SA + 70 CA processors";
  return `
  <svg class="architecture-svg" viewBox="0 0 920 650" aria-label="${escapeHtml(config.short)} legacy spatial architecture overview">
    <defs>${markerDefs(markerId)}</defs>

    <text class="group-label" x="28" y="28">Spatial inputs</text>
    <text class="group-label" x="392" y="28">One doubled U-Net call</text>
    <text class="group-label" x="725" y="28">Absolute output</text>

    ${edge("M194 77 C215 77 220 77 242 77", "memoryFlow", "ba", markerId)}
    ${edge("M382 77 C414 77 414 145 436 145", "memoryFlow", "ba", markerId)}
    ${edge("M194 215 C300 215 330 145 436 145", "target", "pm", markerId)}
    ${edge("M194 330 C315 330 326 365 436 365", "prompt", "pm", markerId)}
    ${edge("M194 420 C315 420 326 395 436 395", "facePrompt", "ba", markerId)}
    ${edge("M194 535 C320 535 330 287 436 287", "maskFlow", "mask", markerId)}
    ${edge("M556 185 C556 205 556 217 556 232", "residualFlow", "ba", markerId)}
    ${edge("M556 322 C556 337 556 345 556 357", "crossAttention", "ba", markerId)}
    ${edge("M676 390 C708 390 708 260 730 260", "compose", "ba", markerId)}
    ${edge("M815 300 C815 330 815 350 815 372", "pmFlow", "pm", markerId)}
    ${edge("M800 120 C725 120 710 145 676 145", "scheduleFlow", "train", markerId)}
    ${edge("M556 568 C556 525 556 495 556 470", "trainFlow", "train", markerId)}

    ${node(28, 46, 166, 62, "reference", ["Reference", "image"], "full spatial evidence", "ba")}
    ${node(242, 42, 140, 70, "memory", ["VAE + noise", "xref,t"], "reference grid", "ba")}
    ${node(28, 180, 166, 70, "target", ["Target noisy", "latent xt"], "first batch half", "pm")}
    ${node(28, 295, 166, 68, "prompt", ["Generation", "PM prompt"], "target-half CA", "pm")}
    ${node(28, 385, 166, 68, "facePrompt", ["ID-only", "face prompt"], "reference-half CA", "ba")}
    ${node(28, 500, 166, 70, "mask", ["Target mask M", "Reference Mref"], "two coordinate grids", "mask")}

    <g class="unet-shell clickable" data-inspect="baPass" tabindex="0" role="button">
      <rect x="416" y="88" width="280" height="408" rx="9"></rect>
      <text class="shell-title" x="556" y="116">Doubled batch [target, reference]</text>
    </g>
    ${node(436, 128, 240, 58, "baPass", ["Target + reference streams"], "one absolute BA prediction", "")}
    ${node(436, 232, 240, 90, "selfAttention", ["Branched self-attention"], "Qtarget face → K/Vreference face", "ba")}
    ${node(436, 357, 240, 72, "crossAttention", ["Split cross-attention"], caSubtitle, "pm")}
    <rect class="site-chip" x="436" y="444" width="240" height="24" rx="12"></rect>
    <text class="site-chip-text" x="556" y="460">${escapeHtml(siteLabel)}</text>

    ${node(730, 220, 170, 80, "compose", ["Return target", "epsilon half"], "no outer PM merge", "ba")}
    ${node(730, 372, 170, 76, "output", ["CFG → scheduler", "0–9 text · 10–14 PM", "15–49 spatial BA"], "", "output")}
    ${node(716, 72, 184, 96, "scheduleFlow", ["Temporal switch", "0–9 text · 10–14 PM", "15–49 spatial BA"], "", "train")}
    ${node(426, 530, 260, 76, "objective", ["Training objective"], config.objectiveShort, "train")}
    ${node(214, 530, 176, 76, "history", ["Historical evidence"], `${LEGACY_SPATIAL.slice(0, 8)} + ${config.sourceCommit}`, "")}
  </svg>`;
}

function renderCompactOverviewSvg(config, panelId) {
  const markerId = `arrow-${panelId}`;
  const memoryLines =
    config.memory.tokens >= 10
      ? ["Compact identity", "memory · 10 tokens"]
      : ["Compact identity", `memory · ${config.memory.tokens} tokens`];
  const siteChip =
    config.sites.count === 70
      ? `<rect class="site-chip" x="371" y="421" width="112" height="24" rx="12"></rect>
         <text class="site-chip-text" x="427" y="437">70 / 70 CA sites</text>`
      : `<rect class="site-chip" x="354" y="421" width="146" height="24" rx="12"></rect>
         <text class="site-chip-text" x="427" y="437">16 / 70 · ≈11 effective</text>`;

  return `
  <svg class="architecture-svg" viewBox="0 0 920 650" aria-label="${escapeHtml(config.short)} architecture diagram">
    <defs>${markerDefs(markerId)}</defs>

    <text class="group-label" x="28" y="28">Inputs</text>
    <text class="group-label" x="285" y="28">Two target-coordinate predictions</text>
    <text class="group-label" x="676" y="28">Arbitration</text>

    ${edge("M194 82 C222 82 236 82 262 82", "memoryFlow", "ba", markerId)}
    ${edge("M472 92 C548 92 548 348 292 383", "residualFlow", "ba", markerId)}
    ${edge("M194 236 C232 236 242 224 292 224", "pmFlow", "pm", markerId)}
    ${edge("M194 246 C240 282 245 370 292 383", "residualFlow", "ba", markerId)}
    ${edge("M194 397 C250 352 242 268 292 248", "pmFlow", "pm", markerId)}
    ${edge("M194 410 C238 410 252 410 292 410", "residualFlow", "ba", markerId)}
    ${edge("M194 548 C255 548 252 469 292 450", "maskFlow", "mask", markerId)}
    ${edge("M194 560 C430 606 570 538 659 420", "maskFlow", "mask", markerId)}
    ${edge("M548 228 C608 228 618 280 659 314", "pmFlow", "pm", markerId)}
    ${edge("M548 410 C610 410 610 385 659 374", "residualFlow", "ba", markerId)}
    ${edge("M770 421 C790 458 792 479 792 506", "pmFlow", "pm", markerId)}
    ${edge("M427 556 C427 524 427 506 427 489", "trainFlow", "train", markerId)}

    ${node(28, 48, 166, 68, "reference", ["Reference", "image"], "identity evidence", "ba")}
    ${node(262, 48, 210, 88, "memory", memoryLines, `reference → ${config.memory.tokens} × 2048-D`, "ba")}
    ${node(28, 202, 166, 72, "prompt", ["PhotoMaker", "prompt context"], "identity-conditioned", "pm")}
    ${node(28, 372, 166, 72, "target", ["Target noisy", "latent xₜ"], "pose + scene coordinates", "")}
    ${node(28, 522, 166, 64, "mask", ["Hard target", "face bbox"], "localization", "mask")}

    ${node(292, 176, 256, 104, "pmPass", ["Ordinary PhotoMaker", "target pass"], "absolute ε_PM baseline", "pm")}
    ${node(292, 350, 256, 140, "baPass", ["BA-modified target pass", "BranchedCrossAttnProcessor"], config.sites.label, "ba")}
    ${siteChip}

    ${node(659, 280, 221, 142, "compose", ["PM / BA", "epsilon composition"], config.compositionShort, "pm")}
    <text class="formula" x="769" y="391">${
      config.family === "pre-N34"
        ? "ε = (1−M) εPM + M εBA"
        : "ε = CFG(PM) + 5 · M · ΔBA"
    }</text>
    ${node(695, 506, 194, 62, "output", ["Merged ε → scheduler"], config.schedule, "output")}
    ${node(292, 538, 256, 76, "objective", ["Training objective"], config.objectiveShort, "train")}
  </svg>`;
}

function renderMechanismSvg(config, panelId) {
  return config.topology === "legacy_spatial"
    ? renderLegacyMechanismSvg(config, panelId)
    : renderResidualMechanismSvg(config, panelId);
}

function renderLegacyMechanismSvg(config, panelId) {
  const markerId = `mechanism-${panelId}`;
  const caTrainingText =
    config.sites.caTrainable === false
      ? "CA forward active at all 70 sites; weights frozen."
      : "CA forward active and trainable at all 70 sites.";
  return `
  <svg class="mechanism-svg" viewBox="0 0 920 840" aria-label="${escapeHtml(config.short)} detailed branched self and cross attention">
    <defs>${markerDefs(markerId)}</defs>

    <text class="mechanism-title" x="24" y="30">A · BranchedAttnProcessor — spatial self-attention at every attn1 site</text>
    <text class="mechanism-note" x="24" y="54">Target face Q attends reference-grid K/V; target background remains target-owned.</text>

    ${node(24, 82, 142, 72, "target", ["target_hidden"], "first batch half", "pm")}
    ${node(24, 276, 142, 72, "memory", ["ref_hidden"], "second batch half", "ba")}

    ${node(214, 70, 126, 56, "selfAttention", ["Qtarget"], "to_q", "pm")}
    ${node(214, 146, 126, 56, "selfAttention", ["Ktarget / Vtarget"], "to_k · to_v", "pm")}
    ${node(214, 270, 126, 66, "selfAttention", ["Kref / Vref"], "to_k · to_v", "ba")}
    ${node(214, 354, 126, 56, "selfAttention", ["Qref / Kref / Vref"], "reference continuation", "ba")}

    ${node(392, 66, 116, 54, "maskFlow", ["q_bg"], "Qtarget × (1−M)", "pm")}
    ${node(392, 136, 116, 54, "maskFlow", ["q_face"], "Qtarget × M", "ba")}
    ${node(392, 214, 116, 54, "selfAttention", ["k_bg / v_bg"], "target K/V", "pm")}
    ${node(392, 284, 116, 64, "residualFlow", ["k_face / v_face"], "reference × Mref", "ba")}

    ${node(570, 98, 132, 64, "selfAttention", ["hidden_bg"], "Attn(q_bg,k_bg,v_bg)", "pm")}
    ${node(570, 244, 132, 70, "residualFlow", ["hidden_face"], "Attn(q_face,k_face,v_face)", "ba")}
    ${node(570, 350, 132, 56, "selfAttention", ["hidden_ref"], "reference self-attn", "ba")}
    ${node(754, 164, 142, 76, "compose", ["merged target"], "(1−M)·bg + M·face", "ba")}
    ${node(754, 330, 142, 64, "baPass", ["output batch"], "[merged, hidden_ref]", "")}

    ${edge("M166 108 C184 108 194 98 214 98", "selfAttention", "pm", markerId)}
    ${edge("M166 120 C188 138 192 174 214 174", "selfAttention", "pm", markerId)}
    ${edge("M166 312 C188 312 194 303 214 303", "residualFlow", "ba", markerId)}
    ${edge("M166 324 C186 358 194 382 214 382", "selfAttention", "ba", markerId)}
    ${edge("M340 98 C360 98 370 93 392 93", "maskFlow", "mask", markerId)}
    ${edge("M340 104 C365 122 368 163 392 163", "maskFlow", "mask", markerId)}
    ${edge("M340 174 C365 174 370 241 392 241", "selfAttention", "pm", markerId)}
    ${edge("M340 303 C360 303 370 316 392 316", "residualFlow", "ba", markerId)}
    ${edge("M508 93 C535 93 540 118 570 128", "selfAttention", "pm", markerId)}
    ${edge("M508 241 C535 225 540 128 570 128", "selfAttention", "pm", markerId)}
    ${edge("M508 163 C535 170 540 270 570 279", "residualFlow", "ba", markerId)}
    ${edge("M508 316 C535 316 540 279 570 279", "residualFlow", "ba", markerId)}
    ${edge("M340 382 C455 382 490 378 570 378", "selfAttention", "ba", markerId)}
    ${edge("M702 128 C730 128 730 190 754 196", "pmFlow", "pm", markerId)}
    ${edge("M702 279 C730 270 730 215 754 210", "residualFlow", "ba", markerId)}
    ${edge("M825 240 C825 275 825 294 825 330", "compose", "ba", markerId)}
    ${edge("M702 378 C724 378 730 362 754 362", "selfAttention", "ba", markerId)}

    ${edgeLabel(350, 84, "× (1−M)", "mask")}
    ${edgeLabel(351, 152, "× M", "mask")}
    ${edgeLabel(348, 296, "× Mref", "mask")}
    ${edgeLabel(711, 151, "× (1−M)", "pm")}
    ${edgeLabel(710, 258, "× M", "ba")}

    <line class="mechanism-divider" x1="24" y1="444" x2="896" y2="444"></line>
    <text class="mechanism-title" x="24" y="478">B · BranchedCrossAttnProcessor — split target/reference cross-attention at every attn2 site</text>
    <text class="mechanism-note" x="24" y="502">Face prompt → reference stream; identity reaches target through spatial SA. ${escapeHtml(caTrainingText)}</text>

    ${node(24, 528, 154, 58, "target", ["target_hidden"], "", "pm")}
    ${node(24, 606, 154, 58, "prompt", ["generation prompt"], "", "pm")}
    ${node(226, 522, 126, 54, "crossAttention", ["q_target"], "to_q(target)", "pm")}
    ${node(226, 606, 126, 58, "crossAttention", ["k_gen / v_gen"], "to_k · to_v", "pm")}
    ${node(430, 558, 150, 66, "crossAttention", ["target_hidden′"], "Attn(qtarget,kgen,vgen)", "pm")}

    ${node(24, 706, 154, 58, "memory", ["ref_hidden"], "", "ba")}
    ${node(226, 700, 126, 54, "crossAttention", ["q_ref"], "to_q(reference)", "ba")}
      ${node(430, 698, 150, 66, "facePrompt", ["ID-only", "face prompt"], "", "ba")}
    ${node(624, 700, 126, 58, "crossAttention", ["k_face / v_face"], "to_k · to_v", "ba")}
    ${node(770, 624, 126, 72, "crossAttention", ["ref_hidden′"], "Attn(qref,kface,vface)", "ba")}

    ${edge("M178 557 C195 557 206 549 226 549", "crossAttention", "pm", markerId)}
    ${edge("M178 635 C195 635 206 635 226 635", "crossAttention", "pm", markerId)}
    ${edge("M352 549 C385 549 394 579 430 587", "crossAttention", "pm", markerId)}
    ${edge("M352 635 C385 635 394 595 430 590", "crossAttention", "pm", markerId)}
    ${edge("M178 735 C195 735 206 727 226 727", "crossAttention", "ba", markerId)}
    ${edge("M580 735 C596 735 604 729 624 729", "facePrompt", "ba", markerId)}
    ${edge("M352 727 C510 810 710 800 770 672", "crossAttention", "ba", markerId)}
    ${edge("M750 729 C760 719 764 683 770 672", "crossAttention", "ba", markerId)}

    <rect class="site-chip" x="648" y="526" width="220" height="26" rx="13"></rect>
    <text class="site-chip-text" x="758" y="543">${escapeHtml(config.weightMode)}</text>
  </svg>`;
}

function renderResidualMechanismSvg(config, panelId) {
  const markerId = `mechanism-${panelId}`;
  const siteText =
    config.sites.count === 70 ? "repeated at all 70 CA sites" : "repeated at 16 selected CA sites";
  const routeDetail =
    config.sites.count === 70
      ? "all target-face attn2 sites · unit scalar gates"
      : "6 × 1.0 + 10 × 0.5 gates · 54 PM sites untouched";
  return `
  <svg class="mechanism-svg compact-mechanism" viewBox="0 0 920 560" aria-label="${escapeHtml(config.short)} detailed target-face residual attention">
    <defs>${markerDefs(markerId)}</defs>
    <text class="mechanism-title" x="24" y="30">BranchedCrossAttnProcessor (attn2) — target-face residual mode</text>
    <text class="mechanism-note" x="24" y="54">Both lanes use target-coordinate queries. Compact reference memory supplies only the identity K/V lane.</text>

    ${node(24, 214, 148, 70, "target", ["target_hidden"], "shared target Q source", "")}
    ${node(24, 76, 148, 66, "prompt", ["PhotoMaker", "context"], "normal PM K/V lane", "pm")}
    ${node(24, 398, 148, 70, "memory", ["Compact identity", "memory"], config.memory.label, "ba")}

    ${node(220, 88, 128, 54, "pmPass", ["k_PM / v_PM"], "attn.to_k · to_v", "pm")}
    ${node(220, 192, 128, 54, "pmPass", ["q_PM"], "attn.to_q(target)", "pm")}
    ${node(220, 306, 128, 54, "residual", ["q_ID"], "attn.to_q(target)", "ba")}
    ${node(220, 406, 128, 60, "residual", ["k_ID / v_ID"], "target_id_to_k · to_v", "ba")}

    ${node(414, 132, 148, 72, "pmPass", ["PM attention"], "Attn(qPM,kPM,vPM)", "pm")}
    ${node(414, 340, 148, 72, "residual", ["Identity attention"], "Attn(qID,kID,vID)", "ba")}
    ${node(612, 340, 132, 72, "residual", ["Zero-init delta"], "face_delta_out", "ba")}
    ${node(612, 446, 132, 62, "mask", ["gate × target M"], "localized strength", "mask")}
    ${node(790, 220, 108, 82, "compose", ["PM + Δface"], config.compositionShort, "ba")}

    ${edge("M172 109 C192 109 198 115 220 115", "pmFlow", "pm", markerId)}
    ${edge("M172 236 C194 214 198 219 220 219", "pmFlow", "pm", markerId)}
    ${edge("M172 258 C194 310 198 326 220 333", "residualFlow", "ba", markerId)}
    ${edge("M172 433 C194 433 198 436 220 436", "residualFlow", "ba", markerId)}
    ${edge("M348 115 C382 115 382 155 414 166", "pmFlow", "pm", markerId)}
    ${edge("M348 219 C382 219 382 176 414 170", "pmFlow", "pm", markerId)}
    ${edge("M348 333 C382 333 382 363 414 374", "residualFlow", "ba", markerId)}
    ${edge("M348 436 C382 436 382 387 414 380", "residualFlow", "ba", markerId)}
    ${edge("M562 376 C582 376 590 376 612 376", "residualFlow", "ba", markerId)}
    ${edge("M678 412 C678 426 678 432 678 446", "maskFlow", "mask", markerId)}
    ${edge("M562 166 C700 166 740 225 790 249", "pmFlow", "pm", markerId)}
    ${edge("M744 477 C780 450 766 305 790 277", "residualFlow", "ba", markerId)}

    ${edgeLabel(362, 106, "K / V", "pm")}
    ${edgeLabel(363, 211, "Q", "pm")}
    ${edgeLabel(363, 324, "Q", "ba")}
    ${edgeLabel(363, 428, "K / V", "ba")}
    ${edgeLabel(754, 422, "M · gate · Δ", "mask")}

    ${node(610, 72, 258, 66, "standardSelfAttention", ["attn1: original processor", "BranchedAttnProcessor absent"], "", "pm")}
    <rect class="site-chip" x="610" y="152" width="258" height="28" rx="14"></rect>
    <text class="site-chip-text" x="739" y="170">${escapeHtml(siteText)}</text>
    <text class="mechanism-note" x="612" y="198">${escapeHtml(routeDetail)}</text>
  </svg>`;
}

function renderCard(cardId, configId) {
  const config = CONFIGS[configId];
  const card = document.getElementById(cardId);
  const faceMae = Number.isFinite(config.faceMae)
    ? `${config.faceMae.toFixed(5)} @ ${config.metricStep}`
    : "not measured comparably";
  const idScore = Number.isFinite(config.idScore)
    ? `${config.idScore.toFixed(4)} @ ${config.metricStep}`
    : "not available";
  const siteMetric =
    config.topology === "legacy_spatial"
      ? config.sites.metricLabel || `${config.sites.count} SA + ${config.sites.count} CA`
      : `${config.sites.count} / 70 CA`;
  card.dataset.config = configId;
  card.querySelector(".card-header").innerHTML = `
    <div>
      <p class="eyebrow">${escapeHtml(config.family)} · ${escapeHtml(config.short)}</p>
      <h2>${escapeHtml(config.title)}</h2>
      <p>${escapeHtml(config.subtitle)}</p>
    </div>
    <span class="status ${escapeHtml(config.status)}">${escapeHtml(config.statusLabel)}</span>`;
  card.querySelector(".metrics-row").innerHTML = [
    ["Memory", config.memory.label, "memory"],
    ["BA attention sites", siteMetric, "sites"],
    ["Face MAE vs PM", faceMae, null],
    ["Mean ID sim", idScore, null],
  ]
    .map(
      ([label, value, key]) =>
        `<div class="metric"${key ? ` data-compare-key="${key}"` : ""}><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`,
    )
    .join("");
  card.querySelector(".diagram-mount").innerHTML = `
    ${renderSvg(config, cardId)}
    <section class="mechanism-panel" aria-label="${escapeHtml(config.short)} attention processor details">
      <div class="mechanism-heading" data-compare-keys="baPass selfAttention standardSelfAttention crossAttention residual sites">
        <div>
          <span>Processor internals</span>
          <strong>${
            config.topology === "legacy_spatial"
              ? "Full spatial branched self- and cross-attention"
              : "BranchedCrossAttnProcessor only (attn2); attn1 remains standard"
          }</strong>
        </div>
        <p>Q / K / V routes · click any block or arrow for code</p>
      </div>
      ${renderMechanismSvg(config, cardId)}
    </section>`;
}

function renderSummary(leftId, rightId) {
  const left = CONFIGS[leftId];
  const right = CONFIGS[rightId];
  const siteRatio = right.sites.effective / left.sites.effective;
  const comparableMae = Number.isFinite(left.faceMae) && Number.isFinite(right.faceMae);
  const maeRatio = comparableMae ? right.faceMae / left.faceMae : null;
  const pmSitesLeft = 70 - left.sites.count;
  const pmSitesRight = 70 - right.sites.count;
  const includesLegacy =
    left.topology === "legacy_spatial" || right.topology === "legacy_spatial";
  const bothLegacy =
    left.topology === "legacy_spatial" && right.topology === "legacy_spatial";
  const includesProposal = left.status === "proposed" || right.status === "proposed";
  const routeCopy = includesLegacy
    ? `${left.short}: ${
        left.topology === "legacy_spatial" ? "full reference grid" : "compact identity memory"
      } · ${right.short}: ${
        right.topology === "legacy_spatial" ? "full reference grid" : "compact identity memory"
      }`
    : `${left.short} ${left.sites.effective} → ${right.short} ${right.sites.effective}`;
  const routeExplanation = includesLegacy
    ? bothLegacy
      ? "Both retain the full reference-coordinate stream and the original doubled-latent branched-attention topology."
      : "The spatial run carries reference coordinates through a second U-Net stream; the compact run exposes only identity K/V memory."
    : `Right has ${(siteRatio * 100).toFixed(0)}% of the left configuration's unit-gate site equivalents.`;
  const pmAuthority = includesLegacy
    ? `${left.short}: ${
        left.topology === "legacy_spatial" ? "single absolute BA pass" : "protected PM baseline"
      } · ${right.short}: ${
        right.topology === "legacy_spatial" ? "single absolute BA pass" : "protected PM baseline"
      }`
    : `${pmSitesLeft} vs ${pmSitesRight}`;
  const pmExplanation = includesLegacy
    ? bothLegacy
      ? "Both return the target half of one doubled BA pass directly; neither protects an independent PhotoMaker epsilon baseline."
      : "The spatial target half is returned directly; the compact-residual run explicitly composes BA with an ordinary PhotoMaker prediction."
    : `${left.short} left / ${right.short} right. Untouched sites retain full PhotoMaker context.`;
  document.getElementById("comparison-summary").innerHTML = `
    <div class="comparison-fact">
      <span>${includesLegacy ? "Identity representation" : "Effective BA route"}</span>
      <strong>${escapeHtml(routeCopy)}</strong>
      <p>${escapeHtml(routeExplanation)}</p>
    </div>
    <div class="comparison-fact">
      <span>Observed face movement</span>
      <strong>${
        comparableMae
          ? `${escapeHtml(right.short)} is ${(maeRatio * 100).toFixed(0)}% of ${escapeHtml(left.short)}`
          : "No directly comparable fixed-set MAE"
      }</strong>
      <p>${
        comparableMae
          ? "Same-seed target-face RGB MAE relative to the PhotoMaker baseline."
          : includesProposal
            ? "A proposed configuration has no checkpoint or validation metric yet."
            : "Runs without the same fixed-set MAE protocol are not compared numerically."
      }</p>
    </div>
    <div class="comparison-fact">
      <span>${includesLegacy ? "PhotoMaker authority" : "Untouched PM CA sites"}</span>
      <strong>${escapeHtml(pmAuthority)}</strong>
      <p>${escapeHtml(pmExplanation)}</p>
    </div>
    <div class="comparison-fact">
      <span>Interpretation</span>
      <strong>${escapeHtml(right.architectureNote)}</strong>
      <p>${escapeHtml(left.architectureNote)}</p>
    </div>`;
}

function renderMatrix() {
  const body = document.querySelector("#config-matrix tbody");
  body.innerHTML = Object.entries(CONFIGS)
    .map(([id, config]) => {
      const faceMae = Number.isFinite(config.faceMae)
        ? `${config.faceMae.toFixed(5)} @ ${escapeHtml(config.metricStep)}`
        : "not comparable";
      const idScore = Number.isFinite(config.idScore)
        ? `${config.idScore.toFixed(4)} @ ${escapeHtml(config.metricStep)}`
        : "not available";
      return `
      <tr data-config="${id}" title="Use ${id} in the right comparison panel">
        <td class="run-cell"><strong>${id} · ${escapeHtml(config.title)}</strong><span>${escapeHtml(config.family)}</span></td>
        <td>${escapeHtml(config.memory.label)}</td>
        <td>${escapeHtml(config.sites.matrixLabel || `${config.sites.count} / 70`)}</td>
        <td>${escapeHtml(config.sites.effectiveLabel || config.sites.effective)}</td>
        <td>${escapeHtml(config.compositionShort)}</td>
        <td>${escapeHtml(config.objectiveShort)}</td>
        <td>${faceMae}</td>
        <td>${idScore}</td>
      </tr>`;
    })
    .join("");
}

const leftSelect = document.getElementById("left-config");
const rightSelect = document.getElementById("right-config");
const differenceToggle = document.getElementById("difference-mode");

function populateSelect(select) {
  select.innerHTML = Object.entries(CONFIGS)
    .map(
      ([id, config]) =>
        `<option value="${id}">${id} · ${escapeHtml(config.title)}</option>`,
    )
    .join("");
}

function updateUrl(left, right) {
  const url = new URL(window.location.href);
  url.searchParams.set("left", left);
  url.searchParams.set("right", right);
  if (differenceToggle.checked) {
    url.searchParams.set("diff", "1");
  } else {
    url.searchParams.delete("diff");
  }
  window.history.replaceState({}, "", url);
}

function applyDifferenceMode(leftId, rightId) {
  const rightCard = document.getElementById("right-card");
  const legend = document.getElementById("difference-legend");
  const enabled = differenceToggle.checked;
  rightCard.classList.toggle("difference-mode", enabled);
  legend.hidden = !enabled;

  if (!enabled) return;

  const differences = comparisonDifferences(CONFIGS[leftId], CONFIGS[rightId]);

  rightCard.querySelectorAll("[data-inspect]").forEach((element) => {
    element.classList.toggle("is-different", differences.keys.has(element.dataset.inspect));
  });
  rightCard.querySelectorAll("[data-compare-key], [data-compare-keys]").forEach((element) => {
    const keys = (element.dataset.compareKeys || element.dataset.compareKey).split(/\s+/);
    element.classList.toggle(
      "is-different",
      keys.some((key) => differences.keys.has(key)),
    );
  });

  const sitesDiffer = differences.keys.has("sites");
  rightCard.querySelectorAll(".site-chip, .site-chip-text").forEach((element) => {
    element.classList.toggle("is-different", sitesDiffer);
  });

  const title = document.getElementById("difference-legend-title");
  const detail = document.getElementById("difference-legend-detail");
  const count = differences.changedGroups.length;
  if (count === 0) {
    title.textContent = "No modeled differences";
    detail.textContent =
      "The selected left and right records have the same modeled architecture and effective configuration.";
  } else {
    title.textContent = `${count} changed ${count === 1 ? "category" : "categories"} on the right`;
    detail.textContent = differences.changedGroups.map((group) => group.label).join(" · ");
  }
}

function diffLines(leftText, rightText) {
  const left = String(leftText || "").split("\n");
  const right = String(rightText || "").split("\n");
  const lengths = Array.from({ length: left.length + 1 }, () =>
    Array(right.length + 1).fill(0),
  );

  for (let i = left.length - 1; i >= 0; i -= 1) {
    for (let j = right.length - 1; j >= 0; j -= 1) {
      lengths[i][j] =
        left[i] === right[j]
          ? lengths[i + 1][j + 1] + 1
          : Math.max(lengths[i + 1][j], lengths[i][j + 1]);
    }
  }

  const lines = [];
  let i = 0;
  let j = 0;
  while (i < left.length && j < right.length) {
    if (left[i] === right[j]) {
      lines.push({ type: "same", text: left[i] });
      i += 1;
      j += 1;
    } else if (lengths[i + 1][j] >= lengths[i][j + 1]) {
      lines.push({ type: "remove", text: left[i] });
      i += 1;
    } else {
      lines.push({ type: "add", text: right[j] });
      j += 1;
    }
  }
  while (i < left.length) {
    lines.push({ type: "remove", text: left[i] });
    i += 1;
  }
  while (j < right.length) {
    lines.push({ type: "add", text: right[j] });
    j += 1;
  }
  return lines;
}

function renderDiffLines(lines) {
  return lines
    .map((line) => {
      const prefix = line.type === "add" ? "+" : line.type === "remove" ? "−" : " ";
      return `<span class="code-diff-line ${line.type}"><b>${prefix}</b>${escapeHtml(line.text)}</span>`;
    })
    .join("");
}

function relevantSnippetDiffs(leftDetail, rightDetail) {
  const leftEntries = leftDetail.code || [];
  const rightEntries = rightDetail.code || [];
  const count = Math.max(leftEntries.length, rightEntries.length);
  const diffs = [];

  for (let index = 0; index < count; index += 1) {
    const left = leftEntries[index] || null;
    const right = rightEntries[index] || null;
    const leftSnippet = left?.snippet || "";
    const rightSnippet = right?.snippet || "";
    const sameSource =
      left?.path === right?.path &&
      left?.line === right?.line &&
      leftSnippet === rightSnippet;
    if (sameSource) continue;
    diffs.push({ left, right, lines: diffLines(leftSnippet, rightSnippet) });
  }
  return diffs;
}

function renderInspectorComparison(side, key) {
  const differencePanel = document.getElementById("inspector-difference");
  const codeDiffPanel = document.getElementById("inspector-code-diff");
  const isRightComparison = differenceToggle.checked && side === "right";
  differencePanel.hidden = !isRightComparison;
  codeDiffPanel.hidden = !isRightComparison;
  if (!isRightComparison) return;

  const leftId = leftSelect.value;
  const rightId = rightSelect.value;
  const left = CONFIGS[leftId];
  const right = CONFIGS[rightId];
  const differences = comparisonDifferences(left, right);
  const changes = differences.byKey.get(key) || [];
  const differenceLabel = document.getElementById("inspector-difference-label");
  const differenceBody = document.getElementById("inspector-difference-body");
  const codeDiffBody = document.getElementById("inspector-code-diff-body");

  differenceLabel.textContent = `Right compared with ${left.short}`;
  if (changes.length === 0) {
    differenceBody.innerHTML = `
      <p class="no-element-change">
        No modeled forward-path or effective-config change for this element.
      </p>`;
    codeDiffBody.innerHTML = `
      <p class="code-diff-note">No relevant code or effective-config diff for this element.</p>`;
    return;
  }

  differenceBody.innerHTML = changes
    .map(
      (change) => `
        <div class="element-change">
          <span>${escapeHtml(change.label)}</span>
          <del>${escapeHtml(change.left)}</del>
          <ins>${escapeHtml(change.right)}</ins>
        </div>`,
    )
    .join("");

  const effectiveLines = changes.flatMap((change) => [
    { type: "remove", text: `${change.codeName} = ${JSON.stringify(change.left)}` },
    { type: "add", text: `${change.codeName} = ${JSON.stringify(change.right)}` },
  ]);
  const leftDetail = detailFor(left, key);
  const rightDetail = detailFor(right, key);
  const snippetDiffs = relevantSnippetDiffs(leftDetail, rightDetail);
  const snippetHtml = snippetDiffs
    .map((diff) => {
      const leftSource = diff.left
        ? `${diff.left.path.replace("../../", "")}:${diff.left.line}`
        : "(no left snippet)";
      const rightSource = diff.right
        ? `${diff.right.path.replace("../../", "")}:${diff.right.line}`
        : "(no right snippet)";
      return `
        <article class="code-diff-card">
          <header>
            <span>${escapeHtml(left.short)} · ${escapeHtml(leftSource)}</span>
            <span>${escapeHtml(right.short)} · ${escapeHtml(rightSource)}</span>
          </header>
          <pre><code>${renderDiffLines(diff.lines)}</code></pre>
        </article>`;
    })
    .join("");

  codeDiffBody.innerHTML = `
    <article class="code-diff-card effective-config-diff">
      <header><span>Effective configuration</span></header>
      <pre><code>${renderDiffLines(effectiveLines)}</code></pre>
    </article>
    ${
      snippetHtml ||
      '<p class="code-diff-note">The linked forward snippet is unchanged; this element differs through effective configuration shown above.</p>'
    }`;
}

function renderAll() {
  const left = CONFIGS[leftSelect.value] ? leftSelect.value : "N3a";
  const right = CONFIGS[rightSelect.value] ? rightSelect.value : "NN1b";
  renderCard("left-card", left);
  renderCard("right-card", right);
  renderSummary(left, right);
  applyDifferenceMode(left, right);
  updateUrl(left, right);
}

function openInspector(configId, key, side = null) {
  const config = CONFIGS[configId];
  const detail = detailFor(config, key);
  const inspector = document.getElementById("inspector");
  document.getElementById("inspector-run").textContent = `${config.short} · ${config.title}`;
  document.getElementById("inspector-title").textContent = detail.title;
  document.getElementById("inspector-description").textContent = detail.description;
  renderInspectorComparison(side, key);
  document.getElementById("inspector-facts").innerHTML = Object.entries(detail.facts || {})
    .map(
      ([label, value]) =>
        `<div class="inspector-fact"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`,
    )
    .join("");
  document.getElementById("inspector-code").innerHTML = (detail.code || [])
    .map((entry) => {
      const belongsToExpBranch =
        config.topology !== "legacy_spatial" &&
        (entry.path.startsWith("../../src/") ||
          entry.path.startsWith("../../serv_new_runs/"));
      const sourcePath = belongsToExpBranch
        ? `${EXP_SOURCE_ROOT}/${entry.path.replace("../../", "")}`
        : entry.path;
      const displayPath = belongsToExpBranch
        ? `main_clean_exp:${entry.path.replace("../../", "")}`
        : entry.path.replace("../../", "");
      const href = `${sourcePath}#L${entry.line}`;
      return `
        <article class="code-card">
          <header>
            <span>${escapeHtml(displayPath)}:${entry.line}</span>
            <a href="${escapeHtml(href)}" target="_blank" rel="noreferrer">${escapeHtml(entry.label)} ↗</a>
          </header>
          <pre><code>${escapeHtml(entry.snippet)}</code></pre>
        </article>`;
    })
    .join("");
  inspector.classList.add("open");
  inspector.setAttribute("aria-hidden", "false");
  document.getElementById("inspector-scrim").classList.add("open");
}

function closeInspector() {
  const inspector = document.getElementById("inspector");
  inspector.classList.remove("open");
  inspector.setAttribute("aria-hidden", "true");
  document.getElementById("inspector-scrim").classList.remove("open");
}

populateSelect(leftSelect);
populateSelect(rightSelect);
const params = new URLSearchParams(window.location.search);
leftSelect.value = CONFIGS[params.get("left")] ? params.get("left") : "N3a";
rightSelect.value = CONFIGS[params.get("right")] ? params.get("right") : "NN1b";
differenceToggle.checked = params.get("diff") === "1";
renderMatrix();
renderAll();

leftSelect.addEventListener("change", renderAll);
rightSelect.addEventListener("change", renderAll);
differenceToggle.addEventListener("change", renderAll);
document.getElementById("swap-configs").addEventListener("click", () => {
  const left = leftSelect.value;
  leftSelect.value = rightSelect.value;
  rightSelect.value = left;
  renderAll();
});
document.getElementById("reset-view").addEventListener("click", () => {
  leftSelect.value = "N3a";
  rightSelect.value = "NN1b";
  differenceToggle.checked = false;
  renderAll();
});
document.getElementById("close-inspector").addEventListener("click", closeInspector);
document.getElementById("inspector-scrim").addEventListener("click", closeInspector);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closeInspector();
});

document.querySelector(".diagram-grid").addEventListener("click", (event) => {
  const target = event.target.closest("[data-inspect]");
  if (!target) return;
  const card = target.closest(".diagram-card");
  const side = card.id === "right-card" ? "right" : "left";
  openInspector(card.dataset.config, target.dataset.inspect, side);
});
document.querySelector(".diagram-grid").addEventListener("keydown", (event) => {
  if (event.key !== "Enter" && event.key !== " ") return;
  const target = event.target.closest("[data-inspect]");
  if (!target) return;
  event.preventDefault();
  const card = target.closest(".diagram-card");
  const side = card.id === "right-card" ? "right" : "left";
  openInspector(card.dataset.config, target.dataset.inspect, side);
});
document.querySelector("#config-matrix tbody").addEventListener("click", (event) => {
  const row = event.target.closest("[data-config]");
  if (!row) return;
  rightSelect.value = row.dataset.config;
  renderAll();
  document.getElementById("right-card").scrollIntoView({ behavior: "smooth", block: "start" });
});
