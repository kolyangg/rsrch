# Training execution flow and `photomaker_branched` code map

Date: 22 August 2026<br>
Code inspected: `clean_new` at `cf27bd8d0322594c9f096efe2d7c0c4c7f0001a3`<br>
Scope: the ten selected 24k E13-family recipes in `serv_run_packages/README.md`

Evidence labels in this document:

- **[code]** means the statement follows from the source/configuration at the commit above.
- **[measured]** means it was also checked by composing the configuration or running the repository's read-only CL23/CL27 validator.

No training job was launched for this analysis.

## 1. Short answer

CL27 is not a separate model implementation. It is a configuration leaf that selects behavior inside the shared E13-family model:

```text
CL27 YAML
  -> train.py
  -> PhotomakerBranchedLora                         lora2.py
       -> validate settings/trainables/checkpoint   e13_contract.py
       -> prepare identity, masks, reference latent lora2_helpers.py
       -> run doubled target/reference U-Net        branched_runtime.py
            -> temporal-frequency self-attention    hardcase_attn_processor.py
       -> collect CL27 auxiliary loss               e13_objectives.py
  -> MaskedDiffusionLoss
  -> backward / optimizer / validation / checkpoint PhotomakerLoraTrainer
```

The three easily confused files have distinct jobs:

- [`e13_contract.py`](../../src/model/photomaker_branched/e13_contract.py) is the **configuration, trainable-parameter, optimizer, and checkpoint guardrail** for every selected recipe. It does not implement the U-Net forward equation or a loss.
- [`e13_objectives.py`](../../src/model/photomaker_branched/e13_objectives.py) is the **optional training-objective and telemetry collector**. It is called after every model forward, but most recipes get a zero auxiliary loss.
- [`hardcase_attn_processor.py`](../../src/model/photomaker_branched/hardcase_attn_processor.py) implements **alternative self-attention routing equations**. It changes outputs only for CL19, CL23, CL27, and CL39. The historical name “hardcase” does not mean it is a data loader or loss file.

## 2. Which recipe activates which code

All ten recipes instantiate the same [`PhotomakerBranchedLora`](../../src/model/photomaker_branched/lora2.py#L34) class and use the same training loop. The leaf YAML changes data policy and/or guarded modules.

| Recipe | Data/validation difference | Self-attention processor that affects output | Additional objective or processor |
|---|---|---|---|
| E13 | `large_dataset`; original E13 validation path | `BranchedAttnProcessor` | None |
| BC_E13 | BigCelebs training data | `BranchedAttnProcessor` | None |
| CL14 | Cosmic data; target training mask feathered by two latent cells | `BranchedAttnProcessor` | None |
| CL14_CA | CL14 data; subject-v2 validation | `BranchedAttnProcessor` | `ResidualIdentityCrossAttnProcessorV3` in `up_blocks.0/1` cross-attention; CA telemetry |
| CL18 | Cosmic dual-reference samples; subject-v2 validation | `BranchedAttnProcessor` | Cross-view consistency objective |
| CL19 | Cosmic data; subject-v2 validation | `HardcaseBranchedAttnProcessor(mode="soft_router")` in all seven U-Net block groups | None |
| CL20 | Deterministic hard-case curriculum data; subject-v2 validation | `BranchedAttnProcessor` | None; this is a data-policy experiment |
| CL23 | Cosmic data; subject-v2 validation | `HardcaseBranchedAttnProcessor(mode="temporal_frequency")` in all seven groups | None |
| CL27 | CL23 plus deterministic semantic occlusion on 25% of training samples | Same temporal-frequency processor as CL23 | Frequency-surface loss in `up_blocks.0/1` |
| CL39 | CL27 data/objective | Same temporal-frequency processor | Detached null-key confidence in `up_blocks.0/1`, plus CL27 surface loss |

**[code]** “All seven groups” means `down_blocks.0`, `down_blocks.1`, `down_blocks.2`, `mid_block`, `up_blocks.0`, `up_blocks.1`, and `up_blocks.2`.

The corresponding config files are:

- [`E13_large_ds_joint_shadow_sa128_24k.yaml`](../../src/configs/E13_large_ds_joint_shadow_sa128_24k.yaml)
- [`BC_E13_big_celebs_joint_shadow_sa128_24k.yaml`](../../src/configs/BC_E13_big_celebs_joint_shadow_sa128_24k.yaml)
- [`CL14_cosmic_joint_shadow_sa128_softmask_24k.yaml`](../../src/configs/CL14_cosmic_joint_shadow_sa128_softmask_24k.yaml)
- [`CL14_CA_cosmic_residual_identity_ca_24k.yaml`](../../src/configs/CL14_CA_cosmic_residual_identity_ca_24k.yaml)
- [`CL18_cosmic_crossview_spatial_consistency_24k.yaml`](../../src/configs/CL18_cosmic_crossview_spatial_consistency_24k.yaml)
- [`CL19_cosmic_true_soft_fullquery_router_24k.yaml`](../../src/configs/CL19_cosmic_true_soft_fullquery_router_24k.yaml)
- [`CL20_cosmic_bigcelebs_hardcase_curriculum_24k.yaml`](../../src/configs/CL20_cosmic_bigcelebs_hardcase_curriculum_24k.yaml)
- [`CL23_cosmic_temporal_frequency_router_24k.yaml`](../../src/configs/CL23_cosmic_temporal_frequency_router_24k.yaml)
- [`CL27_cosmic_frequency_surface_energy_24k.yaml`](../../src/configs/CL27_cosmic_frequency_surface_energy_24k.yaml)
- [`CL39_cosmic_null_key_confidence_router_24k.yaml`](../../src/configs/CL39_cosmic_null_key_confidence_router_24k.yaml)

### 2.1 CL27's actual inheritance chain

Hydra composes CL27 through this chain:

```text
e13_family_24k
  -> E13_large_ds_joint_shadow_sa128_24k
  -> CL14_cosmic_joint_shadow_sa128_softmask_24k
  -> subject_v2_extension_24k
  -> CL19_cosmic_true_soft_fullquery_router_24k
  -> CL23_cosmic_temporal_frequency_router_24k
  -> CL27_cosmic_frequency_surface_energy_24k
```

Each layer contributes a small delta:

1. [`e13_family_24k.yaml`](../../src/configs/e13_family_24k.yaml) selects the common model/trainer/pipeline, 24k schedule, validation protocol, and default `e13_settings`.
2. E13 selects `large_dataset` and its fixed validation masks.
3. CL14 changes to Cosmic data and sets `ba_training_mask_feather: 2`.
4. [`subject_v2_extension_24k.yaml`](../../src/configs/subject_v2_extension_24k.yaml) selects the subject-v2 validation pipeline and identity metrics.
5. CL19 selects the full-query soft router.
6. CL23 changes that router to temporal-frequency mode and supplies its early/late low/high gains.
7. CL27 enables the frequency-surface objective in `up_blocks.0/1` and deterministic semantic occlusion with probability `0.25` and seed `150017`.

**[measured]** The composed CL27 run is 12 epochs × 2,000 optimizer steps = 24,000 steps, batch size 2, with the fixed 96-item validation panel. The repository validator reported 2,240 trainable tensors and 219,217,920 trainable parameters for this non-CA recipe.

## 3. Step-by-step: starting and running CL27

### 3.1 Launcher and preflight

The supported shell entry point is [`run_e13_family_24k_1gpu.sh`](../../launchers/active/run_e13_family_24k_1gpu.sh).

1. The launcher resolves the requested recipe and unique run name.
2. It rejects undeclared Hydra overrides so the sealed recipe is not silently changed.
3. It hashes source/config artifacts and performs dataset/cache preflights.
4. For CL23/CL27 it runs [`validate_cl23_cl27_config.py`](../../tools/validate_cl23_cl27_config.py), which composes the config and verifies the processor set, trainable ownership, schedule, and validation panel.
5. It verifies that InsightFace is using CUDA ONNX Runtime.
6. It invokes `accelerate launch train.py --config-name CL27_cosmic_frequency_surface_energy_24k`.
7. After startup it waits for `saved/<run_name>/comet_experiment.json`, preserving the immutable Comet experiment key.

### 3.2 Hydra composition and top-level construction

Execution enters [`main()` in `train.py`](../../train.py#L109).

1. Hydra composes the CL27 chain above.
2. `main()` sets the seed and creates the Accelerate runtime and Comet writer.
3. [`get_dataloaders()`](../../src/datasets/data_utils.py#L54) builds the configured Cosmic training loader and fixed manual-validation loader.
4. Hydra instantiates [`PhotomakerBranchedLora`](../../src/model/photomaker_branched/lora2.py#L34).
5. `model.prepare_for_training()` installs adapters and branched processors, then freezes every tensor not allowed by the E13 contract.
6. Hydra instantiates [`MaskedDiffusionLoss`](../../src/loss/diffusion_loss.py) and the configured metrics.
7. `model.get_trainable_params(config)` asks `e13_contract.optimizer_groups()` for exact optimizer groups. `assert_trainable_contract()` checks that the optimizer contains exactly those tensors.
8. Accelerate prepares the model, optimizer, loaders, and scheduler.
9. Hydra instantiates the configured subject-v2 validation pipeline and [`PhotomakerLoraTrainer`](../../src/trainer/sdxl_trainers.py#L227).
10. `trainer.train()` starts the epoch loop.

### 3.3 What the model constructor and `prepare_for_training()` do

[`PhotomakerBranchedLora.__init__()`](../../src/model/photomaker_branched/lora2.py#L37) builds the common model used by every recipe:

1. The inherited SDXL wrapper loads the VAE, U-Net, text encoders, tokenizer, and scheduler.
2. [`PhotoMakerIDEncoder_CLIPInsightfaceExtendtoken`](../../src/model/photomaker_branched/model_v2_NS.py#L130) is created for PhotoMaker identity-token conditioning.
3. [`create_face_analyzer()`](../../src/model/photomaker_branched/insightface_package.py#L25) creates the InsightFace detector/recognizer used to obtain the 512-D face embedding.
4. Trigger-token and fixed architecture invariants are set: no pose adaptation, no face CA mixing, rank-128 branched self-attention, and the selected identity prompt strategy.
5. [`initialise_e13_contract()`](../../src/model/photomaker_branched/e13_contract.py#L123) normalizes and validates CL27's settings.
6. The pretrained PhotoMaker default adapter is loaded.

Then [`prepare_for_training()`](../../src/model/photomaker_branched/lora2.py#L114):

1. Runs the inherited SDXL/PhotoMaker preparation.
2. Freezes the base model.
3. Adds the generic rank-32 LoRA adapter and activates it together with the PhotoMaker default adapter.
4. Calls [`install_branched_processors_for_training()`](../../src/model/photomaker_branched/lora2_helpers.py#L18).
5. That function calls [`patch_unet_attention_processors()`](../../src/model/photomaker_branched/branched_runtime.py#L15), which installs the CL27 temporal-frequency self-attention processors.
6. [`configure_trainables()`](../../src/model/photomaker_branched/e13_contract.py#L254) applies the exact trainable allowlist.

For CL27 the optimizer owns three roles:

- rank-128 `noise_to_*` and `ref_to_*` projections inside branched self-attention;
- effective rank-32 generic-adapter sites;
- effective rank-64 PhotoMaker-default-adapter sites.

The SDXL base, VAE, text encoders, PhotoMaker ID encoder, and InsightFace model remain frozen. CL14_CA alone adds the residual identity-CA rank-64 role.

### 3.4 Step-zero validation

Before the first optimizer update, [`BaseTrainer._train_epoch()`](../../src/trainer/base_trainer.py#L304) performs the required step-zero validation.

1. The configured alternate validation base, `SG161222/RealVisXL_V4.0`, is loaded into a temporary model/pipeline.
2. The trained architecture state is copied strictly into that model.
3. The validation model keeps the training run's branched and generic adapter state while restoring the pretrained PhotoMaker-default adapter (“shadow PhotoMaker default” protocol).
4. [`PhotomakerBranchedSubjectV2Pipeline`](../../src/pipelines/photomaker_branched_subject_v2.py#L118) chooses the reference face by overlap with the declared reference bounding box, rather than simply taking an arbitrary detected face.
5. Each of the fixed 96 validation items runs 50 DDIM denoising steps with CFG 5.
6. Images and the configured identity/prompt/quality metrics are logged.
7. The temporary validation model is released and the training model is restored.

For the sealed schedule:

| Denoising step | PhotoMaker identity conditioning | Branched attention |
|---|---:|---:|
| 0–9 | Off | Off |
| 10–14 | On | Off |
| 15–49 | On | On |

CL27's frequency-surface objective is training-only: processor collection requires both training mode and enabled gradients. Therefore CL27 validation/inference uses the same temporal-frequency routing equation as CL23; it does not add an inference loss pass.

### 3.5 Building one CL27 training batch

[`CosmicLargeAdaptedTrain.__getitem__()`](../../src/datasets/cosmic_large_adapted.py#L464) prepares a sample:

1. It selects and transforms the target image, prompt, target face box, one reference image, and reference face box.
2. For CL27, a deterministic RNG keyed by `semantic_occlusion_seed + dataset_index` selects 25% of samples for a semantic overlay/occlusion.
3. It emits `ba_occluder_mask`, identifying the synthetic covered region. Unselected samples receive an all-zero mask.
4. The collate function stacks image tensors and retains prompts, reference images, boxes, and auxiliary masks in batch fields.

This data change is what gives the CL27 surface objective a known “covered” face region. CL23 uses the same model route but has neither the overlay policy nor the surface objective.

### 3.6 One trainer step

[`PhotomakerLoraTrainer.process_batch()`](../../src/trainer/sdxl_trainers.py#L227) performs the optimizer step:

1. Calls `model(**batch)`.
2. Passes the returned noise prediction, diffusion-noise target, face mask, and `ba_aux_loss` to `MaskedDiffusionLoss`.
3. CL27 uses the face-masked diffusion loss every step (`masked_loss_step: 1`).
4. The criterion adds `ba_aux_loss` to the main diffusion loss.
5. Accelerate backpropagates, clips/logs as configured, steps the optimizer and scheduler, and clears gradients.
6. Detached CL27 telemetry is sent to the writer; it does not create an additional backward path.

### 3.7 Inside `PhotomakerBranchedLora.forward()`

[`forward()`](../../src/model/photomaker_branched/lora2.py#L154) is deliberately an orchestration method:

1. Encode target pixels with the VAE.
2. Sample target Gaussian noise and one diffusion timestep shared by the batch.
3. Add noise to target latents and build SDXL time IDs.
4. Call [`prepare_branched_training_inputs()`](../../src/model/photomaker_branched/lora2_helpers.py#L256).
5. Call [`prepare_frequency_surface_mask()`](../../src/model/photomaker_branched/e13_objectives.py#L20) to put the CL27 occluder mask on the model for the selected processors.
6. Call [`run_branched_forward_pass()`](../../src/model/photomaker_branched/lora2_helpers.py#L376), which delegates to the common two-branch runtime.
7. Call [`compute_e13_objectives()`](../../src/model/photomaker_branched/e13_objectives.py#L242).
8. Return target-lane noise prediction, sampled target noise, auxiliary loss, and telemetry.

`prepare_branched_training_inputs()` does the conditioning work:

1. [`_encode_prompts_with_trigger_word()`](../../src/model/photomaker_branched/lora2_helpers.py#L41) encodes the prompt and records the trigger-token positions.
2. The reference image is processed by CLIP vision and by InsightFace.
3. `PhotoMakerIDEncoder_CLIPInsightfaceExtendtoken.forward()` fuses those visual features into two 2048-D PhotoMaker identity tokens at the trigger positions.
4. [`encode_reference_latents()`](../../src/model/photomaker_branched/lora2_helpers.py#L207) VAE-encodes the reference image to the target latent resolution.
5. [`_bbox_to_target_mask()`](../../src/model/photomaker_branched/lora2_helpers.py#L169) makes the target face mask and applies CL27's inherited two-cell feather.
6. [`bbox_to_reference_mask()`](../../src/model/photomaker_branched/lora2_helpers.py#L128) makes the reference-face mask.

### 3.8 The doubled target/reference U-Net pass

[`two_branch_predict()`](../../src/model/photomaker_branched/branched_runtime.py#L251) is shared by training and validation:

1. It samples one reference-noise tensor for this pass/generation and noises the reference latent at the same scheduler timestep as the target lane.
2. It forms one doubled U-Net batch: `[target_noisy_latents, reference_noisy_latents]`.
3. It converts the actual scheduler timestep into normalized denoising progress. This matters because training always passes `step_idx=0`; the CL23/CL27 gain schedule must follow the real timestep instead.
4. `patch_unet_attention_processors()` supplies current target/reference masks, denoising progress, and CL27's ownership mask to the already-installed processors. The optimized path caches processor lookup rather than resolving the processor mapping once per layer.
5. With the selected `face_embed_strategy="id"`, the reference half receives only the scaled PhotoMaker identity-token positions from the already-fused prompt. No separate free-form “face” prompt is used.
6. It doubles text and SDXL added-conditioning tensors to match the latent batch.
7. It calls the U-Net once.
8. It returns only the first, merged target half. The reference half exists to provide features/K/V during attention.

### 3.9 What CL27 attention computes

[`create_hardcase_processor()`](../../src/model/photomaker_branched/hardcase_attn_processor.py#L439) sees CL27's `temporal_frequency` setting and installs `HardcaseBranchedAttnProcessor` in the declared self-attention groups.

For each selected self-attention site, [`_call_temporal_frequency()`](../../src/model/photomaker_branched/hardcase_attn_processor.py#L351):

1. Splits the doubled hidden states into target and reference halves.
2. Computes a full target-native self-attention message.
3. Computes a full reference-conditioned message using target queries and reference keys/values.
4. Takes `reference_message - native_message`.
5. Splits that delta spatially into fixed Gaussian low- and high-frequency components.
6. Interpolates low/high gains from the configured early values to late values using real denoising progress.
7. Multiplies the delta by the two-cell soft face router inherited from CL19.
8. Adds the routed low/high delta to the native target message.
9. Recombines the processed target half with the reference half so later U-Net layers still have both lanes.

CL23 performs exactly these nine operations. CL27 additionally asks processors in `up_blocks.0/1` to retain differentiable surface-loss terms.

The base [`BranchedAttnProcessor`](../../src/model/photomaker_branched/attn_processor_cleanest.py#L92), used directly by E13/BC_E13/CL14/CL14_CA/CL18/CL20, instead performs the older hard target-face replacement between target-native and target-query/reference-KV messages. The hardcase processor inherits its rank-128 branch projections from this base class.

#### 3.9.1 At-a-glance comparison

**[code]** `HardcaseBranchedAttnProcessor` is a subclass of `BranchedAttnProcessor`; it is not a separate U-Net architecture and it does not replace cross-attention. It reuses the base class's branch projections and mask preparation, but overrides the self-attention equation.

| Aspect | `BranchedAttnProcessor` | `HardcaseBranchedAttnProcessor` |
|---|---|---|
| Directly selected by | E13, BC_E13, CL14, CL14_CA, CL18, CL20 | CL19, CL23, CL27, CL39 |
| Config switch | `ba_hardcase_mode: off` | `soft_router` or `temporal_frequency` |
| U-Net sites | `attn1` self-attention sites | Declared `attn1` self-attention groups; all seven groups in the selected hardcase recipes |
| Class relationship | Base `nn.Module` implementation | Subclass of the base processor |
| Branch Q/K/V parameters | Independent rank-128 `noise_to_*` and `ref_to_*` projections | Exactly the same inherited projections |
| Target queries | Multiplied by background/face masks before the two target attention calls | Full target queries are used for both target messages |
| Native target message | Computed as a masked background branch | Computed as a full native target self-attention message |
| Reference-conditioned message | Computed as a masked face branch | Computed as a full target-Q/reference-KV message |
| Target merge | Binary hard replacement | CL19 cosine soft blend, or CL23+ routed frequency delta |
| Denoising-time dependence | None | CL23/27/39 interpolate low/high gains from real scheduler progress |
| Optional confidence | None | CL39 applies detached entropy-based reference confidence |
| Optional objective | None inside this processor | CL27/39 can retain differentiable surface terms in `up_blocks.0/1` |
| Processor-owned trainables | Rank-128 BA projections | The same inherited projections; no hardcase-only trainable parameters |
| Reference half | Full reference self-attention, propagated to the next U-Net layer | Same behavior |
| Cross-attention | Not changed by this class | Not changed by this class |
| Final caller-visible output | Runtime eventually returns only the target half | Same |

The name “hardcase” is architectural history, not a dataset selector. In particular, **CL20's `hardcase_curriculum` dataset does not use `HardcaseBranchedAttnProcessor`**: CL20 leaves `ba_hardcase_mode` off and therefore uses the base `BranchedAttnProcessor`.

#### 3.9.2 Shared foundation

Both processors receive hidden states in the fixed doubled order:

```text
[target/generation batch, matching reference batch]
```

For the notation below:

- `T` is the normalized target hidden state.
- `R` is the normalized reference hidden state.
- `M` is the target-face ownership mask.
- `Mr` is the reference-face support mask.
- `A(Q, K, V)` is PyTorch scaled-dot-product attention.
- `Qn/Kn/Vn` are the target/noise branch projections.
- `Qr/Kr/Vr` are the reference branch projections.
- `O` is the attention output linear projection (`attn.to_out[0]`).

**[code]** The common behavior inherited or reproduced by both classes is:

1. Apply Diffusers spatial normalization when configured.
2. Flatten `[B, C, H, W]` hidden states to `[B, H×W, C]` for attention.
3. Split target and reference halves and apply Diffusers group normalization when configured.
4. Use six independent branch projections: target/noise Q/K/V and reference Q/K/V.
5. Advance the reference half with full `Qr(R)`, `Kr(R)`, `Vr(R)` self-attention.
6. Rejoin target and reference halves, restore spatial shape, apply the original residual connection, and divide by Diffusers' `rescale_output_factor`.

The six branch projections are [`BranchLoRALinear`](../../src/model/photomaker_branched/attn_processor_cleanest.py#L11) modules. Each contains a frozen copy of the effective base + pretrained PhotoMaker projection and one trainable rank-128 LoRA delta. Both processor classes therefore start from and train the same branch parameterization.

#### 3.9.3 Exact base-processor route

[`BranchedAttnProcessor.__call__()`](../../src/model/photomaker_branched/attn_processor_cleanest.py#L161) creates three dense attention messages.

First it computes target queries once:

```text
q = Qn(T)
```

It then computes a target-native background message. The target query is zeroed inside the target face, while target K/V remain available over the full target grid:

```text
B = A(q * (1 - M), Kn(T), Vn(T))
```

It separately computes a reference-conditioned face message. The query is zeroed outside the target face, and the reference hidden state is zeroed outside the reference face before the reference K/V projections:

```text
F = A(q * M, Kr(R * Mr), Vr(R * Mr))
```

The target message is then hard-selected and sent through the output projection:

```text
T_base = O((1 - M) * B + scale * M * F)
```

The reference half advances independently:

```text
R_next = O(A(Qr(R), Kr(R), Vr(R)))
```

Important consequences:

- `M` is binary by the time attention uses it. `_prepare_mask()` resizes in two dimensions and thresholds at `> 0.5`; CL14's training feather can alter the resulting boundary geometry, but it does not remain a continuously weighted attention router here.
- The background and face queries are masked **before** their respective softmax operations and the resulting messages are masked again during the final hard merge.
- At an output token, the base processor selects either native-target context or reference-conditioned context; there is no intentional fractional blend between them.
- `self.scale` multiplies the selected face message. All selected clean recipes pass `scale=1.0`.

#### 3.9.4 CL19: full messages followed by one soft router

[`HardcaseBranchedAttnProcessor._full_target_lanes()`](../../src/model/photomaker_branched/hardcase_attn_processor.py#L250) does not mask target queries. It computes two meaningful, full-grid target messages:

```text
N = O(A(Qn(T), Kn(T), Vn(T)))
F = O(A(Qn(T), Kr(R * Mr), Vr(R * Mr)))
```

Here `N` is the full native target message and `F` is the full reference-conditioned message. Unlike the base implementation, `O` is applied to each complete message before routing.

CL19 then derives a router `S` from the binary target face mask. [`_soft_router_mask()`](../../src/model/photomaker_branched/hardcase_attn_processor.py#L143) erodes inward from the face boundary and assigns cosine weights to the transition rings. With the selected `hardcase_transition_cells=2`, a clean binary face region has:

```text
outside face = 0
outermost inside ring = 0.25
next inside ring = 0.75
remaining face interior = 1
```

[`_call_soft_router()`](../../src/model/photomaker_branched/hardcase_attn_processor.py#L332) blends the two full messages exactly once:

```text
T_CL19 = (1 - S) * N + S * F
       = N + S * (F - N)
```

This is the first major difference from the base processor:

- The base route makes two query-masked messages and then performs a binary ownership selection.
- CL19 makes two full-query messages and retains both near the boundary, allowing a controlled transition from native to reference-conditioned self-attention.

Away from the transition rings, CL19 still has the expected endpoints: `S=0` gives native target self-attention and `S=1` gives target-Q/reference-KV attention. However, it is not implemented as the base hard route: query masking, routing weights, and output-projection order differ.

The hardcase class inherits `self.scale`, but its CL19/23/27/39 routing equations do not multiply by that value. This is not an active selected-config discrepancy because the family runtime always passes `scale=1.0`.

#### 3.9.5 CL23: route frequency bands of the reference delta

CL23 uses `hardcase_mode: temporal_frequency`. It first computes the same `N`, `F`, and soft router `S` as CL19, then forms the reference-conditioned change relative to native target self-attention:

```text
D = F - N
```

[`_gaussian_split()`](../../src/model/photomaker_branched/hardcase_attn_processor.py#L161) reshapes `D` back to its square token grid and applies a fixed separable 5×5 Gaussian independently to each feature channel:

```text
L = Gaussian5x5(D)
H = D - L
```

`L` is the low-spatial-frequency part of the reference delta; `H` is the residual high-frequency part. These are frequency bands of the **attention output delta**, not Fourier bands of the input image or latent.

The runtime supplies real scheduler progress:

```text
p = 1 - timestep / (num_train_timesteps - 1)
```

Thus `p=0` is the noisy/early end and `p=1` is the clean/late end. The selected gains are linearly interpolated:

```text
g_low(p)  = 0.50 + p * (0.85 - 0.50)
g_high(p) = 0.75 + p * (1.25 - 0.75)
```

[`_call_temporal_frequency()`](../../src/model/photomaker_branched/hardcase_attn_processor.py#L351) adds the routed, gain-adjusted delta to the full native message:

```text
T_CL23 = N + S * (g_low(p) * L + g_high(p) * H)
```

This differs from CL19's direct blend. If both gains were exactly `1`, then `L + H = D` and the equation would reduce to `N + S*(F-N)`, the CL19 equation. The actual CL23 schedule deliberately weights low and high spatial detail differently and changes those weights over denoising time.

Training historically calls the shared runtime with `step_idx=0`, so these gains do not use loop index. [`two_branch_predict()`](../../src/model/photomaker_branched/branched_runtime.py) derives `p` from the actual scheduler timestep and refreshes every hardcase processor before the U-Net call.

#### 3.9.6 CL27 and CL39 additions

CL27 uses the same target-output equation as CL23. For identical weights and inputs, enabling CL27's surface collector does not directly modify `T_CL23`; it records differentiable terms from the already-computed low/high components in `up_blocks.0/1`.

That statement concerns one instantaneous forward at identical weights. During training, CL27's added gradient changes how the shared projections evolve, so a trained CL27 checkpoint is not expected to remain numerically identical to a trained CL23 checkpoint.

Inside [`_frequency_surface_loss()`](../../src/model/photomaker_branched/hardcase_attn_processor.py#L194):

1. `top = binary(occluder_mask) * binary(face_mask)`.
2. `visible = face_mask - top`.
3. Samples are eligible only when both regions contain tokens.
4. The top term measures routed high-band mean-square energy plus `0.25 ×` routed low-band mean-square energy under the synthetic occluder.
5. The floor term penalizes visible-face routed/native RMS ratios below `0.35`.
6. The processor stores those terms; `e13_objectives.py` later averages them across selected layers and applies weights `0.02` and `0.005` once.

The collection guard requires all three conditions: the processor is in a configured surface group, the module is in training mode, and gradients are enabled. Therefore CL27 inference executes the CL23 output equation without an auxiliary-loss pass.

CL39 keeps the CL27/CL23 frequency equation but adds a detached confidence `C` in `up_blocks.0/1`. [`_null_key_confidence()`](../../src/model/photomaker_branched/hardcase_attn_processor.py#L278):

1. Recomputes target-query/reference-key logits in bounded query chunks.
2. Converts each query's normalized attention entropy into `null_mass` with threshold `0.75` and temperature `0.08`.
3. Maps that to reference confidence, allowing at most `0.75` abstention and retaining at least a `0.25` reference fraction.
4. Detaches the complete confidence calculation from autograd.

CL39's output is therefore:

```text
T_CL39 = N + C * S * (g_low(p) * L + g_high(p) * H)
```

High-entropy/ambiguous reference matches move `C` toward `0.25`, retaining more native target attention. Confident matches keep `C` near `1`. Because `C` is detached and parameter-free, it changes forward routing and scales gradients flowing through the routed components, but no optimizer gradient trains the confidence calculation itself.

Unlike CL27's collector, CL39 confidence directly changes the attention output during both training and inference in its selected `up_blocks.0/1` sites.

#### 3.9.7 Mask semantics: what did not change

The hardcase variants change target routing, but they deliberately preserve the historical reference support behavior:

- Both classes inherit `_prepare_mask()`, which resizes masks and thresholds them to binary support.
- Both multiply the reference hidden state by `Mr` before `Kr/Vr` projection.
- Neither passes a true key-attention mask to scaled-dot-product attention.
- Consequently, all reference token positions remain in the softmax denominator. Positions outside `Mr` contribute K/V derived from a zeroed hidden state (or projection bias, if present). This is the documented historical “zero-sink” behavior, not strict removal of non-face keys.
- `attention_mask` and `encoder_hidden_states` are accepted by the processor signatures for Diffusers compatibility but are not used by these `attn1` self-attention equations.

CL27/39's ownership/occluder mask is a third mask with a different purpose: it identifies surface-loss regions and CL39 telemetry regions. It does not replace `M`, does not define reference support, and does not change CL27's forward message.

There is also a small batch-validation difference. The base processor uses the target-mask batch dimension to infer the split and then requires an equal reference half. The hardcase processor directly requires an even hidden-state batch and divides it in half. The selected runtime satisfies both contracts with the same `[target B, reference B]` layout.

#### 3.9.8 Trainable parameters and gradient routing

**[code]** The hardcase processor adds no trainable tensor of its own. It inherits the same six rank-128 branch projections initialized by `BranchedAttnProcessor.init_from_attention()`:

```text
noise_to_q, noise_to_k, noise_to_v
ref_to_q,   ref_to_k,   ref_to_v
```

The soft-router weights, temporal gains, Gaussian kernel, CL27 loss constants, and CL39 confidence constants are fixed configuration/runtime values. As a result:

- Base non-CA recipes and CL19/23/27/39 keep the same sealed optimizer ownership: `840` BA tensors / `127,795,200` BA parameters, and `2,240` total trainable tensors / `219,217,920` total trainable parameters.
- Checkpoint compatibility still distinguishes modes and groups through the architecture manifest even though parameter counts are equal.
- From the base processor's target-output merge, native-target gradients are selected outside `M`, while target-Q/reference-KV gradients are selected inside `M`; the separately propagated reference half also trains the reference projections.
- In CL19's transition rings, both full messages receive gradient because both have nonzero blend weights.
- In CL23/27/39, gradients flow through native and routed low/high delta components. CL27 adds its surface gradients only in selected up-block processors.
- CL39's detached `C` scales the routed-component gradients but receives no gradient itself.

The generic rank-32 and PhotoMaker-default rank-64 outer adapters are shared family trainables and are not introduced by either processor class. CL14_CA's additional rank-64 residual cross-attention parameters belong to a separate processor and are unrelated to the hardcase inheritance.

#### 3.9.9 Output projection, reference propagation, and compute

The processors also differ in where routing occurs relative to `attn.to_out[0]`:

- Base: merge raw background/face head outputs first, concatenate the target and reference halves, then apply the shared output projection once to that doubled result.
- Hardcase: apply the shared output projection separately to full `N` and `F`, route/blend in output-feature space, project the reference message separately, then concatenate the halves and apply only the output dropout stage.

For CL19's scalar convex blend and a linear output projection, these orders are closely related. In the implemented CL23 route, the hardcase order means the Gaussian split operates on the projected output-feature delta. It also means the output projection and its active adapters are evaluated for both candidate target messages.

Both processors perform three dense scaled-dot-product attention calls per selected self-attention site:

```text
base:     target background + target/reference face + reference self-attention
hardcase: full native target + full target/reference + reference self-attention
```

Masking queries in the base processor does not make its dense attention calls sparse. Hardcase overhead beyond those three calls is:

- one additional target-sized output-projection application because both candidate target messages are projected before routing;
- cosine-router mask construction for CL19+;
- a depthwise 5×5 Gaussian operation for CL23/27/39;
- CL27/39 surface reductions in selected training layers;
- CL39's extra chunked query/key logits and entropy calculation in selected layers.

Finally, both return a doubled hidden-state batch to the next U-Net layer so the reference representation continues to evolve. Only after the complete U-Net pass does `two_branch_predict()` discard the reference output half and return the merged target noise prediction to training or denoising.

### 3.10 What `e13_objectives.py` adds for CL27

Within the selected `up_blocks.0/1` processors, [`_frequency_surface_loss()`](../../src/model/photomaker_branched/hardcase_attn_processor.py#L194) partitions the face into:

- `top`: the synthetic occluder region intersected with the target face;
- `visible`: the rest of the target face.

It then creates two terms:

1. A top-region energy penalty on the routed high-frequency component plus `0.25 ×` the low-frequency component.
2. A visible-face floor penalty if the routed-delta RMS falls below `0.35 ×` native-message RMS.

[`_collect_frequency_surface_loss()`](../../src/model/photomaker_branched/e13_objectives.py#L85) averages live terms across selected processors and groups telemetry into `up0` and `up1`. [`compute_e13_objectives()`](../../src/model/photomaker_branched/e13_objectives.py#L242) applies CL27's weights:

```text
ba_aux_loss = 0.02 * top_loss + 0.005 * visible_floor_loss
```

Only samples containing both a non-empty covered-face region and a visible-face region are eligible. Samples without a CL27 overlay contribute zero to this auxiliary term.

For other recipes the same function behaves as follows:

| Recipe | `compute_e13_objectives()` result beyond common return fields |
|---|---|
| E13, BC_E13, CL14, CL19, CL20, CL23 | Zero auxiliary loss; no active objective telemetry |
| CL14_CA | Zero auxiliary loss; collects identity-CA telemetry |
| CL18 | Weighted stop-gradient cross-view consistency loss |
| CL27 | Frequency-surface auxiliary loss and telemetry |
| CL39 | CL27 surface loss plus null-key telemetry collected from the processors |

Thus `e13_objectives.py` is **called by all recipes**, but “called” does not mean “adds a loss.” Its settings guards determine that.

### 3.11 Epoch boundary, checkpoint, and later validation

After every 2,000 optimizer steps:

1. The trainer runs the same fixed 96-item validation protocol.
2. It saves a checkpoint through `PhotomakerBranchedLora.get_state_dict()`.
3. [`e13_contract.get_state_dict()`](../../src/model/photomaker_branched/e13_contract.py#L473) stores only the exact trainable U-Net tensors plus an architecture manifest.
4. [`e13_contract.load_state_dict()`](../../src/model/photomaker_branched/e13_contract.py#L502) refuses a checkpoint whose manifest, names, shapes, or architecture settings differ from the current recipe.
5. Training continues until epoch 12 / step 24,000.
6. The launcher waits for completion and then runs the deferred face-quality finalizer.

## 4. Every file in `src/model/photomaker_branched`

The status words below are intentional:

- **Active**: executes and affects the selected recipe's result.
- **Guarded**: called/imported, but its config guard makes it a no-op for that recipe.
- **Support**: executed transitively by another active module.
- **Not selected**: present for another mode or debugging, not part of these recipe outputs.

### 4.1 `__init__.py`

[`__init__.py`](../../src/model/photomaker_branched/__init__.py) is an empty package marker. It has no runtime function and selects no experiment.

### 4.2 `lora2.py` — common model entry point

[`lora2.py`](../../src/model/photomaker_branched/lora2.py) is **active for all ten recipes**.

Key API:

- `PhotomakerBranchedLora.__init__()` constructs the SDXL/PhotoMaker model and initializes the E13 settings contract.
- `prepare_for_training()` installs adapters/processors and applies trainable ownership.
- `forward()` orchestrates one training forward as described above.
- `get_trainable_params()`, `get_state_dict()`, `load_state_dict_()`, and `assert_trainable_contract()` delegate policy to `e13_contract.py`.
- `ensure_branched_after_eval()` restores runtime processor state after temporary validation.

This file answers “what is the model class?” It intentionally delegates detailed conditioning, routing, optional objectives, and checkpoint policy to smaller modules.

### 4.3 `lora2_helpers.py` — training-input and forward orchestration

[`lora2_helpers.py`](../../src/model/photomaker_branched/lora2_helpers.py) is **active for all ten recipes during training**.

Key functions:

- `install_branched_processors_for_training()` installs the selected processors, copies runtime settings, and configures trainables.
- `_encode_prompts_with_trigger_word()` performs SDXL prompt encoding and identifies PhotoMaker trigger tokens.
- `bbox_to_reference_mask()` turns the reference face box into a latent-resolution mask.
- `_bbox_to_target_mask()` turns the target face box into a latent mask; CL14-derived recipes use the two-cell feather setting here.
- `encode_reference_latents()` VAE-encodes the reference image.
- `prepare_branched_training_inputs()` combines prompt, CLIP image, InsightFace embedding, PhotoMaker identity-token fusion, masks, and reference latents.
- `run_branched_forward_pass()` calls `two_branch_predict()` with the selected training strategy.
- `ensure_branched_after_eval()` re-patches the training U-Net after validation.

### 4.4 `e13_contract.py` — family contract, not an objective

[`e13_contract.py`](../../src/model/photomaker_branched/e13_contract.py) is **active for all ten recipes at construction, optimizer setup, checkpoint save, and checkpoint load**.

Key functions:

- `normalise_e13_settings()` merges a leaf's values with shared defaults and normalizes types.
- `initialise_e13_contract()` validates legal combinations. For example, it rejects surface loss without temporal-frequency routing or residual CA outside declared groups.
- `copy_pipeline_runtime_settings()` puts only inference/runtime settings onto the validation pipeline/model.
- `trainable_role()` maps each U-Net tensor to one allowed optimizer role or rejects it.
- `expected_trainable_names()` and `configure_trainables()` produce/apply the allowlist.
- `assert_trainable_contract()` checks exact names, role counts, parameter counts, non-U-Net freezes, and optional optimizer membership.
- `optimizer_groups()` creates role-specific parameter groups and learning rates.
- `architecture_manifest()` serializes architecture-relevant settings and exact trainable metadata.
- `get_state_dict()` saves the manifest and trainable U-Net tensors.
- `_validate_compatible_manifest()` and `load_state_dict()` enforce strict compatibility on restore.

Why it exists: without this file, two similarly named runs could silently train a different tensor set or load an incompatible processor topology. It centralizes invariants rather than changing the attention computation.

Why it is named E13: E13 is the sealed base ownership/checkpoint profile inherited by all selected descendants. The file is not E13-only.

### 4.5 `e13_objectives.py` — conditional auxiliary objectives

[`e13_objectives.py`](../../src/model/photomaker_branched/e13_objectives.py) is **called for all ten recipes after the U-Net forward**, with behavior selected by settings.

Key functions:

- `prepare_frequency_surface_mask()` stores CL27/CL39's batch occluder mask for the processors. It clears/does nothing when the surface feature is disabled.
- `_collect_identity_ca_telemetry()` reads detached metrics from CL14_CA's residual CA processors; with no such processors it returns an empty dictionary.
- `_collect_frequency_surface_loss()` gathers differentiable CL27/CL39 terms and detached telemetry from selected up-block processors.
- `_crossview_consistency_loss()` performs CL18's optional second-reference teacher/student comparison. It returns zero when CL18 is not selected.
- `compute_e13_objectives()` is the public aggregator called by `lora2.forward()`. It returns `ba_aux_loss`, detached cross-view loss, and telemetry.

This file does not choose which attention processor is installed. It consumes results exposed by whichever optional processor/objective the config enabled.

### 4.6 `attn_processor_cleanest.py` — base branched self-attention

[`attn_processor_cleanest.py`](../../src/model/photomaker_branched/attn_processor_cleanest.py) is **active directly for E13, BC_E13, CL14, CL14_CA, CL18, and CL20**, and is the superclass/support code for CL19, CL23, CL27, and CL39.

Key API:

- `BranchLoRALinear` implements the low-rank branch projection.
- `_clone_effective_linear()` initializes a branch projection from the effective PhotoMaker/base linear projection without changing its initial function.
- `_branch_batch_sizes()` checks the target/reference doubled-batch layout.
- `BranchedAttnProcessor.init_from_attention()` initializes rank-128 target/reference Q/K/V branch projections.
- `set_masks()` receives current target and reference masks.
- `__call__()` computes the base hard replacement: target-native context outside the face and target-query/reference-KV context inside it, while retaining the reference lane.
- `_prepare_mask()` resizes/caches masks for each token resolution.

The “cleanest” name is historical. This is the common base implementation, not a config named Cleanest.

### 4.7 `hardcase_attn_processor.py` — CL19/23/27/39 self-attention variants

[`hardcase_attn_processor.py`](../../src/model/photomaker_branched/hardcase_attn_processor.py) is **output-active only for CL19, CL23, CL27, and CL39**.

`patch_unet_attention_processors()` calls its factory for every recipe, but [`create_hardcase_processor()`](../../src/model/photomaker_branched/hardcase_attn_processor.py#L439) returns `None` when `ba_hardcase_mode` is off or the U-Net block is not declared. The runtime then installs the base processor instead. This distinction explains why the file appears in the common call path without changing E13/CL14.

The equation-by-equation comparison with `BranchedAttnProcessor`, including masks, gradients, output projections, and runtime cost, is in Sections 3.9.1–3.9.9 above.

Key API:

- `set_denoise_progress()` receives the scheduler-derived progress used by CL23/27/39.
- `set_ownership_target_mask()` receives CL27/39's synthetic occluder mask.
- `frequency_surface_aux_loss()` and `latest_ba_telemetry()` expose per-processor results to `e13_objectives.py`.
- `_normalized_halves()` validates/splits target and reference hidden states.
- `_soft_router_mask()` makes CL19's two-cell cosine face router.
- `_gaussian_split()` splits a spatial message into low/high bands.
- `_full_target_lanes()` computes full native and reference-conditioned target messages.
- `_call_soft_router()` implements CL19.
- `_call_temporal_frequency()` implements CL23 and is also the main route for CL27/39.
- `_frequency_surface_loss()` computes CL27/39's optional training terms.
- `_null_key_confidence()` computes CL39's detached confidence from reference-key attention entropy.
- `__call__()` dispatches to the selected mode.

Activation summary:

```text
mode off                -> factory returns None -> BranchedAttnProcessor
mode soft_router        -> CL19
mode temporal_frequency -> CL23
                         -> CL27 + surface-loss collection
                         -> CL39 + surface-loss collection + null-key confidence
```

### 4.8 `branched_runtime.py` — processor installation and doubled U-Net runtime

[`branched_runtime.py`](../../src/model/photomaker_branched/branched_runtime.py) is **active for all ten recipes in training and BA-enabled validation**.

Key functions:

- `patch_unet_attention_processors()` installs or updates processors. It chooses the hardcase self-attention processor, base self-attention processor, and optional residual identity cross-attention processor according to settings; later calls update masks/progress without rebuilding the trainable objects.
- `encode_face_prompt()` creates the legacy literal-face prompt embedding. It is a fallback/alternative path; the selected family normally uses `face_embed_strategy="id"`.
- `two_branch_predict()` builds the doubled target/reference batch, patches current runtime state, calls the U-Net once, and returns the merged target prediction.

This is the bridge between high-level model/pipeline code and low-level attention processors.

### 4.9 `residual_identity_ca_processor_v3.py` — CL14_CA only

[`residual_identity_ca_processor_v3.py`](../../src/model/photomaker_branched/residual_identity_ca_processor_v3.py) is **constructed only for CL14_CA**, in declared `attn2` sites in `up_blocks.0/1`. The class is imported for common runtime type checks but does not affect the other nine recipes.

Key API:

- `_ResidualLoRALinear` is its rank-64 trainable low-rank projection.
- `init_from_attention()` initializes the processor from the native attention layer.
- `named_ba_trainables()` exposes its delta/output gate parameters to ownership checks.
- `set_masks()` and `set_class_tokens_mask()` receive target-face and PhotoMaker identity-token masks.
- `_project_attention()` executes an attention projection without altering the native CA route.
- `_prepare_spatial_mask()` resizes the target face mask.
- `_gather_identity_tokens()` selects only active PhotoMaker identity tokens.
- `__call__()` computes native cross-attention, then adds a face-local, RMS-normalized, zero-initialized, bounded-gate identity residual.
- `latest_ba_telemetry()` exposes detached gate/residual measurements.

CL14_CA still uses `BranchedAttnProcessor` for self-attention. This file adds a separate cross-attention residual; it does not replace BA.

### 4.10 `model_v2_NS.py` — PhotoMaker identity encoder

[`model_v2_NS.py`](../../src/model/photomaker_branched/model_v2_NS.py) is **active for all ten recipes**.

Key classes/functions:

- `MLP` projects the 512-D InsightFace embedding.
- `QFormerPerceiver` combines projected identity information with CLIP image hidden states using `FacePerceiverResampler`.
- `FuseModule.fuse_fn()` merges identity tokens with prompt tokens; `forward()` applies the merge at trigger-token positions.
- `PhotoMakerIDEncoder_CLIPInsightfaceExtendtoken.forward()` is the training/validation identity-conditioning route: CLIP reference pixels plus InsightFace embedding become two 2048-D identity tokens inserted into the prompt.
- `extract_id_features()` creates an optional reduced 2048-D feature used by the alternate `face_embed_strategy="id_embeds"` route.

Validation setup currently computes/caches `extract_id_features()` output, but all selected E13-family configs set `face_embed_strategy: id`. In that strategy `two_branch_predict()` does not pass the cached `id_embeds` feature to processors; it uses the already-fused identity-token prompt instead. The main selected result path is therefore `forward()`, not the optional `id_embeds` route.

### 4.11 `resampler.py` — identity-encoder support

[`resampler.py`](../../src/model/photomaker_branched/resampler.py) contains two resampler implementations.

Active/support path for all recipes:

- `FacePerceiverResampler.forward()` updates the identity latents from CLIP image features.
- `FeedForward()`, `reshape_tensor()`, and `PerceiverAttention.forward()` support that operation.

Not selected by the ten recipes:

- The separate generic `Resampler` class and its `masked_mean()` helper have no call site elsewhere in the inspected selected code. They are not instantiated by `model_v2_NS.py`.

### 4.12 `insightface_package.py` — face analysis and CUDA checks

[`insightface_package.py`](../../src/model/photomaker_branched/insightface_package.py) is **active for all ten recipes**.

Key API:

- `FaceAnalysis2.get()` extends InsightFace detection with an explicit detector size.
- `create_face_analyzer()` creates and prepares the detector/recognizer with the requested execution providers.
- `assert_cuda_face_analyzer()` fails preflight if the selected training analyzer is not actually using the required CUDA runtime/provider.
- `analyze_faces()` runs detection and retries with the configured detector size where needed.

Training uses the first detected reference face to obtain the 512-D embedding after dataset reference selection. Subject-v2 validation performs the additional declared-box overlap selection in the pipeline before identity conditioning/metrics.

### 4.13 `branch_helpers.py` — validation mask preparation

[`branch_helpers.py`](../../src/model/photomaker_branched/branch_helpers.py) is **active during branched validation/inference**, not in `lora2.forward()` training preparation.

Its public function `prepare_mask4()` reads the pipeline's generated/reference face mask, preserves soft masks, expands it to the current latent/CFG batch, and resizes it to latent resolution. The denoising helper then passes that mask to `two_branch_predict()`.

Training constructs equivalent target/reference masks in `lora2_helpers.py` because its inputs come directly from the dataset batch rather than pipeline state.

### 4.14 `debug_helpers.py` — optional validation image dumps

[`debug_helpers.py`](../../src/model/photomaker_branched/debug_helpers.py) is **not output-active for the selected ten configs**, because `e13_family_24k.yaml` sets `val_debug: false`.

Key functions:

- `log_debug_image()` controls debug-image log messages.
- `_val_debug_enabled()` reads the pipeline guard.
- `save_branch_previews()` decodes/saves per-step merged prediction previews.
- `debug_reference_latents_once()` writes one-time reference/mask diagnostics.
- `save_debug_ref_latents()` decodes cached reference latents.
- `save_debug_ref_mask_overlay()` writes reference-mask overlays.
- `save_debug_images()` is a generic tensor-image dump helper.

The validation helper imports and has guarded call sites for the first group, but they return before writing when `_val_debug` is false. No selected-code call site was found for `save_debug_images()` itself.

## 5. Common source outside this folder

The folder does not contain the entire training program. These external files are part of the CL27 path:

| File | Key role |
|---|---|
| [`train.py`](../../train.py) | Hydra/Accelerate construction and handoff to the trainer |
| [`data_utils.py`](../../src/datasets/data_utils.py) | Dataloader construction and collation |
| [`cosmic_large_adapted.py`](../../src/datasets/cosmic_large_adapted.py) | Cosmic sample selection and CL27 semantic-occlusion mask |
| [`base_trainer.py`](../../src/trainer/base_trainer.py) | Epoch loop, step-zero/periodic validation, temporary validation model, checkpoints |
| [`sdxl_trainers.py`](../../src/trainer/sdxl_trainers.py) | PhotoMaker batch processing, backward/optimizer step, generation and metric logging |
| [`diffusion_loss.py`](../../src/loss/diffusion_loss.py) | Face-masked diffusion MSE plus `ba_aux_loss` |
| [`photomaker_branched_clean.py`](../../src/pipelines/photomaker_branched_clean.py) | Main validation denoising pipeline |
| [`photomaker_branched_subject_v2.py`](../../src/pipelines/photomaker_branched_subject_v2.py) | Declared-box subject selection used by CL14_CA/CL18/CL19/CL20/CL23/CL27/CL39 |
| [`br_pipeline_helpers.py`](../../src/pipelines/br_pipeline_helpers.py) | Validation reference preparation, masks, ID features, denoising-step mode switches, and call into `two_branch_predict()` |

## 6. Practical navigation: where to look for a question

| Question | First file/function to inspect |
|---|---|
| Which model class does every recipe instantiate? | `lora2.py::PhotomakerBranchedLora` |
| Which leaf settings are legal? | `e13_contract.py::initialise_e13_contract` |
| Exactly which tensors train? | `e13_contract.py::trainable_role`, `assert_trainable_contract` |
| How are target/reference latents and prompts doubled? | `branched_runtime.py::two_branch_predict` |
| How are training prompts, ID tokens, boxes, and reference latents prepared? | `lora2_helpers.py::prepare_branched_training_inputs` |
| What is the base E13/CL14 BA equation? | `attn_processor_cleanest.py::BranchedAttnProcessor.__call__` |
| What changes in CL19/23/27/39? | `hardcase_attn_processor.py` |
| What exactly is CL27's extra loss? | `hardcase_attn_processor.py::_frequency_surface_loss`, then `e13_objectives.py::_collect_frequency_surface_loss` |
| What exactly is CL18's extra loss? | `e13_objectives.py::_crossview_consistency_loss` |
| What exactly is CL14_CA's extra path? | `residual_identity_ca_processor_v3.py::__call__` |
| How are CLIP and InsightFace fused into PhotoMaker prompt tokens? | `model_v2_NS.py::PhotoMakerIDEncoder_CLIPInsightfaceExtendtoken.forward` |
| How is a checkpoint protected from the wrong architecture? | `e13_contract.py::architecture_manifest`, `load_state_dict` |
| How does validation select the declared subject? | `photomaker_branched_subject_v2.py::_subject_v2_id_embeds` |
| Why are debug PNG helpers present but not writing files? | `debug_helpers.py::_val_debug_enabled` plus `e13_family_24k.yaml::val_debug` |

## 7. Imported or present does not always mean active

These are the main cases that can otherwise make the code map misleading:

1. `e13_objectives.compute_e13_objectives()` is called in every training forward, but returns zero optional loss for six of the ten recipes.
2. `hardcase_attn_processor.create_hardcase_processor()` is called while installing all recipe processors, but returns `None` when the mode/group does not select it.
3. `ResidualIdentityCrossAttnProcessorV3` is imported by the common runtime, but constructed only for CL14_CA.
4. Debug helper call sites exist in validation, but all selected configs disable them with `val_debug: false`.
5. `resampler.Resampler` is present but is not the `FacePerceiverResampler` used by the PhotoMaker ID encoder.
6. `model_v2_NS.extract_id_features()` is computed/cached in validation setup, but the selected `face_embed_strategy="id"` does not consume that alternate `id_embeds` route.
7. `branched_runtime.encode_face_prompt()` is a fallback/alternative strategy; the selected route uses PhotoMaker identity-token positions from the fused prompt.

For the detailed attention equations, trainable ownership counts, and architecture lineage, see [`2026-08-13_e13_family_architecture_reference.md`](2026-08-13_e13_family_architecture_reference.md) and [`2026-08-18_cl23_cl27_clean_extension.md`](2026-08-18_cl23_cl27_clean_extension.md).
