#!/usr/bin/env python3
"""Validate NN3a lab checkpoints at manifest-defined optimizer steps.

Each mode/checkpoint is a fresh batch-1 process with a console-only writer.
This keeps peak memory bounded and prevents the training framework from
creating validation experiments.  After rendering, images are uploaded
directly to the verified training Comet key with stream/checkpoint names.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
COSMIC_DATA = HERE / "data" / "id_00081_1017318003459"
ONE_ID_DATA = HERE / "data" / "one_id_nm0005092"
ONE_ID_ROOT = Path("/home/niko/rsrch/dataset_full/one_id")
ONE_ID_IMAGES = ONE_ID_ROOT / "nm0005092_adj"
ONE_ID_PROMPTS = Path("/home/niko/rsrch/dataset_full/val_dataset/prompts_10.txt")
ENV_BIN = Path("/home/niko/miniconda3/envs/photomaker_NS/bin")
PM_PATH = Path("/home/niko/models/PhotoMaker-V2/photomaker-v2.bin")
REALVIS = "SG161222/RealVisXL_V4.0"
MODES = {
    "canonical50": {
        "use_branched_attention": True,
        "automatic_bboxes": True,
        "ba_start": 15,
    },
    "earlyBA50": {
        "use_branched_attention": True,
        "automatic_bboxes": True,
        "ba_start": 12,
    },
    "pmControl50": {
        "use_branched_attention": False,
        "automatic_bboxes": False,
        "ba_start": 15,
    },
}
COMET_WORKSPACE = "nikolay-2104"
COMET_PROJECT = "rsrch-30oct"


def resolve_dataset_profile(training_manifest: dict) -> dict:
    profile = training_manifest.get("dataset_profile", "cosmic_large_id00081")
    if profile == "cosmic_large_id00081":
        return {
            "name": profile,
            "train_overrides": [
                "train_dataset_name=cosmic_large_neb",
                (
                    "datasets.train.cosmic_large_neb.data_json_pth="
                    f"{COSMIC_DATA / 'train_8refs.json'}"
                ),
                "datasets.train.cosmic_large_neb.num_refs=1",
            ],
            "images_dir": COSMIC_DATA / "validation_refs",
            "prompts_path": COSMIC_DATA / "validation_prompts_4.txt",
            "classes_path": COSMIC_DATA / "classes_ref.json",
            "ref_bbox_path": COSMIC_DATA / "ref_bboxes.json",
            "shared_pm_mask": (
                COSMIC_DATA / "pm_generated_bboxes_holdout_A_seed0.json"
            ),
        }
    if profile in {
        "one_id_nm0005092_subset8",
        "one_id_nm0005092_subset8_distinct",
        "one_id_nm0005092_full18_heldout_distinct",
    }:
        distinct_pairing = profile.endswith("_distinct")
        data_json = (
            ONE_ID_DATA / "full18_no_validation_train.json"
            if profile == "one_id_nm0005092_full18_heldout_distinct"
            else ONE_ID_DATA / "subset8_train.json"
        )
        return {
            "name": profile,
            "train_overrides": [
                "train_dataset_name=one_id",
                (
                    "datasets.train.one_id.cosmic_json_pth="
                    f"{data_json}"
                ),
                f"datasets.train.one_id.images_path={ONE_ID_IMAGES}",
                "datasets.train.one_id.num_refs=1",
                f"train_on_separate_image={'true' if distinct_pairing else 'false'}",
            ],
            "images_dir": ONE_ID_ROOT / "ref",
            "prompts_path": ONE_ID_PROMPTS,
            "classes_path": ONE_ID_ROOT / "one_id_classes_ref.json",
            "ref_bbox_path": ONE_ID_ROOT / "nm0005092_adj_test.json",
            "shared_pm_mask": (
                ONE_ID_DATA / "pm_generated_bboxes_ref51_seed0.json"
            ),
        }
    raise ValueError(f"Unsupported dataset profile: {profile}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--steps",
        default="0,200,400,600",
        help="Comma-separated subset of 0,200,400,600",
    )
    parser.add_argument(
        "--modes",
        default="canonical50,earlyBA50,pmControl50",
        help=f"Comma-separated subset of {','.join(MODES)}",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def ensure_comet_api_key() -> None:
    """Inherit the already-authorized key from the live production trainer."""
    if os.environ.get("COMET_API_KEY"):
        return
    marker = b"writer.run_name=ba_N3a_new1_1gpu"
    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit():
            continue
        try:
            command = (proc_dir / "cmdline").read_bytes()
            if marker not in command:
                continue
            for entry in (proc_dir / "environ").read_bytes().split(b"\0"):
                if entry.startswith(b"COMET_API_KEY="):
                    os.environ["COMET_API_KEY"] = entry.split(b"=", 1)[1].decode()
                    return
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    raise RuntimeError(
        "COMET_API_KEY unavailable and no authorized production trainer was found"
    )


def zero_lora_b(value):
    if isinstance(value, dict):
        return {
            key: (
                torch.zeros_like(item)
                if torch.is_tensor(item)
                and ("lora_B" in key or "lora.b" in key.lower())
                else zero_lora_b(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [zero_lora_b(item) for item in value]
    return value


def build_step0_checkpoint(source_checkpoint: Path, destination: Path) -> Path:
    checkpoint = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    if "state_dict" not in checkpoint:
        raise RuntimeError(f"Full checkpoint expected: {source_checkpoint}")
    derived = {
        "arch": checkpoint.get("arch"),
        "epoch": 0,
        "state_dict": zero_lora_b(copy.deepcopy(checkpoint["state_dict"])),
        "config": checkpoint.get("config"),
        "derived_from": str(source_checkpoint),
        "derivation": (
            "All LoRA-B tensors zeroed. NN3a_new1 BA projections are LoRA "
            "residuals, so this restores their exact zero-functional step-0 state."
        ),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(derived, destination)
    return destination


def checkpoint_for_step(
    run_dir: Path,
    run_name: str,
    step: int,
    checkpoint_every: int,
) -> Path:
    checkpoint_dir = run_dir / "checkpoints" / run_name
    epoch = step // checkpoint_every
    if step == 0:
        derived = run_dir / "validation" / "derived_step0.pth"
        if not derived.exists():
            source = checkpoint_dir / "checkpoint-epoch1.pth"
            if not source.exists():
                raise FileNotFoundError(source)
            build_step0_checkpoint(source, derived)
        return derived
    path = checkpoint_dir / f"checkpoint-epoch{epoch}.pth"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def resolve_comet_key(run_name: str) -> str:
    """Resolve the real Comet key; config.yaml contains only a local logger ID."""
    ensure_comet_api_key()
    from comet_ml import API

    api = API()
    for attempt in range(30):
        for page in range(10):
            experiments = api.get_experiments(
                COMET_WORKSPACE,
                COMET_PROJECT,
                page=page,
                page_size=100,
                sort_by="startTime",
                sort_order="desc",
            )
            for experiment in experiments:
                if experiment.name == run_name:
                    return str(experiment.key)
            if len(experiments) < 100:
                break
        if attempt < 29:
            time.sleep(10)
    raise RuntimeError(
        f"Could not resolve Comet experiment key for {run_name!r} after 5 minutes"
    )


def upload_images_to_comet(
    comet_key: str,
    run_name: str,
    mode_name: str,
    step: int,
    image_paths: list[Path],
) -> list[str]:
    """Upload rendered files without invoking the framework's Comet writer."""
    ensure_comet_api_key()
    from comet_ml import ExistingExperiment

    experiment = ExistingExperiment(previous_experiment=comet_key)
    if experiment.get_key() != comet_key:
        raise RuntimeError(
            f"Comet resume verification failed: {experiment.get_key()} != {comet_key}"
        )
    experiment.set_name(run_name)
    names = []
    for prompt_index, path in enumerate(sorted(image_paths)):
        name = (
            f"{mode_name}__step{step:04d}__p{prompt_index:02d}__{path.name}"
        )
        experiment.log_image(str(path), name=name, step=step)
        names.append(name)
    experiment.log_other(
        f"23Jul_{mode_name}_step{step:04d}_image_count", len(names)
    )
    experiment.end()
    return names


def command_for(
    run_dir: Path,
    training_manifest: dict,
    mode_name: str,
    step: int,
    checkpoint: Path,
):
    dataset = resolve_dataset_profile(training_manifest)
    architecture_id = training_manifest["architecture_id"]
    architecture = training_manifest["architecture"]
    checkpoint_every = int(
        training_manifest.get("protocol", {}).get("checkpoint_every", 200)
    )
    scope = architecture["lab_train_scope"]
    train_run_name = training_manifest["run_name"]
    comet_key = training_manifest["comet_experiment_key"]
    validation_name = f"{mode_name}__step{step:04d}"
    mode = MODES[mode_name]
    val_dir = run_dir / "validation" / mode_name / f"step_{step:04d}"
    run_fixed_bbox = run_dir / "validation" / "fixed_gen_bboxes.json"
    shared_pm_mask = dataset["shared_pm_mask"]
    use_shared_pm_mask = bool(
        mode["use_branched_attention"] and shared_pm_mask.exists()
    )
    use_run_fixed_bbox = bool(
        mode["use_branched_attention"]
        and not use_shared_pm_mask
        and run_fixed_bbox.exists()
    )
    use_fixed_bbox = use_shared_pm_mask or use_run_fixed_bbox
    if use_shared_pm_mask:
        bbox_seed = shared_pm_mask
    elif use_run_fixed_bbox:
        bbox_seed = run_fixed_bbox
    else:
        bbox_seed = val_dir / "gen_bboxes.json"
    automatic_bboxes = bool(mode["automatic_bboxes"] and not use_fixed_bbox)
    outputs = val_dir / "outputs"
    val_dir.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)
    if not bbox_seed.exists():
        bbox_seed.write_text("{}\n", encoding="utf-8")

    overrides = [
        "datasets=all_datasets",
        *dataset["train_overrides"],
        "val_datasets_names=[manual_val]",
        f"datasets.val.manual_val.images_dir={dataset['images_dir']}",
        f"datasets.val.manual_val.prompts_path={dataset['prompts_path']}",
        f"datasets.val.manual_val.classes_json_path={dataset['classes_path']}",
        f"datasets.val.manual_val.bbox_mask_ref={dataset['ref_bbox_path']}",
        f"datasets.val.manual_val.bbox_mask_gen={bbox_seed}",
        "datasets.val.manual_val.seeds=[0]",
        "datasets.val.manual_val.limit=4",
        "inference_metrics=[]",
        "dataloaders.manual_val.batch_size=1",
        "dataloaders.manual_val.num_workers=1",
        "dataloaders.train.batch_size=1",
        "dataloaders.train.batch_size_eff=1",
        "dataloaders.train.grad_accum_enabled=false",
        "dataloaders.train.num_workers=1",
        f"trainer.epoch_len={checkpoint_every}",
        "trainer.n_epochs=1",
        f"trainer.save_dir={outputs}",
        "trainer._target_=training_lab.nn3a_lab_trainer.NN3aLabTrainer",
        "model._target_=training_lab.nn3a_lab_model.NN3aTrainingLabModel",
        f"+model.lab_train_scope={scope}",
        f"+model.lab_optimizer_recipe={architecture.get('lab_optimizer_recipe', 'production')}",
        f"model.pretrained_model_name_or_path={REALVIS}",
        f"pipeline.pretrained_model_name_or_path={REALVIS}",
        "pretrained_model_for_validation_name_or_path=null",
        "model.rank=32",
        f"model.photomaker_path={PM_PATH}",
        "model.weight_dtype=bf16",
        "model.num_inference_steps=50",
        "model.photomaker_start_step=10",
        f"model.branched_attn_start_step={mode['ba_start']}",
        "pipeline.variant=null",
        "validation_args.num_images_per_prompt=1",
        "validation_args.num_inference_steps=50",
        "validation_args.guidance_scale=5",
        "validation_args.photomaker_start_step=10",
        "validation_args.merge_start_step=10",
        f"validation_args.branched_attn_start_step={mode['ba_start']}",
        f"validation_args.use_branched_attention={str(mode['use_branched_attention']).lower()}",
        "validation_args.use_bbox_mask_ref=true",
        f"validation_args.use_bbox_mask_gen={str(mode['use_branched_attention']).lower()}",
        f"automatic_bboxes={str(automatic_bboxes).lower()}",
        f"automatic_bboxes_every_val={str(automatic_bboxes).lower()}",
        f"force_log_first_auto_bbox={str(automatic_bboxes).lower()}",
        "validation_args.val_debug=false",
        f"validation_args.debug_dir={val_dir / 'debug'}",
        "val_debug=false",
        f"+lab_validation_stream={mode_name}",
        "+validation_only=true",
        f"saved_checkpoint={checkpoint}",
        "writer=console",
        f"writer.run_name={validation_name}",
    ]
    for field in (
        "lab_ref_kv_lr_scale",
        "lab_ref_v_lr_scale",
        "lab_ref_q_lr_scale",
        "lab_noise_lr_scale",
        "lab_up0_lr_scale",
    ):
        if field in architecture:
            overrides.append(f"+model.{field}={architecture[field]}")
    validation_port = (
        int(training_manifest.get("validation_port_base", 29740))
        + list(MODES).index(mode_name) * 20
        + step // checkpoint_every
    )
    command = [
        str(ENV_BIN / "accelerate"),
        "launch",
        "--config_file=src/configs/ddp/accelerate.yaml",
        "--main_process_ip=127.0.0.1",
        f"--main_process_port={validation_port}",
        "--num_processes=1",
        "train.py",
        "--config-name=one_id_ba_N3a_new1",
        *overrides,
    ]
    job_manifest = {
        "architecture_id": architecture_id,
        "dataset_profile": dataset["name"],
        "training_run_name": train_run_name,
        "comet_run_name": train_run_name,
        "comet_experiment_key": comet_key,
        "comet_stream": validation_name,
        "mode": mode_name,
        "mode_config": mode,
        "automatic_bboxes_this_job": automatic_bboxes,
        "fixed_bbox_reused": use_fixed_bbox,
        "generated_mask_source": str(bbox_seed),
        "generated_mask_kind": (
            "shared_photomaker_only_pass"
            if use_shared_pm_mask
            else "run_fixed"
            if use_run_fixed_bbox
            else "generated_this_job"
        ),
        "step": step,
        "checkpoint": str(checkpoint),
        "command": command,
        "status": "prepared",
    }
    (val_dir / "validation_manifest.json").write_text(
        json.dumps(job_manifest, indent=2) + "\n", encoding="utf-8"
    )
    (val_dir / "resolved_command.sh").write_text(
        "#!/usr/bin/env bash\n" + shlex.join(command) + "\n", encoding="utf-8"
    )
    return val_dir, job_manifest, command


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    training_manifest_path = run_dir / "run_manifest.json"
    training_manifest = json.loads(training_manifest_path.read_text(encoding="utf-8"))
    allowed_statuses = {"running", "completed", "resuming"}
    if (
        training_manifest.get("status") not in allowed_statuses
        and not args.dry_run
    ):
        raise SystemExit(
            f"Training run is not validation-ready: "
            f"{training_manifest.get('status')}"
        )
    if not training_manifest.get("comet_experiment_key"):
        if args.dry_run:
            training_manifest["comet_experiment_key"] = "DRY_RUN_COMET_KEY"
        else:
            training_manifest["comet_experiment_key"] = resolve_comet_key(
                training_manifest["run_name"]
            )
            latest_manifest = json.loads(
                training_manifest_path.read_text(encoding="utf-8")
            )
            latest_manifest["comet_experiment_key"] = training_manifest[
                "comet_experiment_key"
            ]
            training_manifest = latest_manifest
            training_manifest_path.write_text(
                json.dumps(training_manifest, indent=2) + "\n", encoding="utf-8"
            )
    validation_snapshot = run_dir / "validation" / "source_snapshot"
    validation_snapshot.mkdir(parents=True, exist_ok=True)
    for source in (HERE / "launch_validation.py", HERE / "summarize_run.py"):
        shutil.copy2(source, validation_snapshot / source.name)
    steps = [int(value) for value in args.steps.split(",") if value]
    modes = [value for value in args.modes.split(",") if value]
    protocol = training_manifest.get("protocol", {})
    valid_steps = {
        int(value)
        for value in protocol.get("validation_steps", [0, 200, 400, 600])
    }
    checkpoint_every = int(protocol.get("checkpoint_every", 200))
    if not set(steps) <= valid_steps:
        raise SystemExit(f"Unsupported steps: {steps}")
    unknown_modes = set(modes) - set(MODES)
    if unknown_modes:
        raise SystemExit(f"Unsupported modes: {sorted(unknown_modes)}")

    environment = os.environ.copy()
    environment["PATH"] = f"{ENV_BIN}:{environment.get('PATH', '')}"
    environment["PYTHONPATH"] = f"{HERE}:{PROJECT}:{environment.get('PYTHONPATH', '')}"
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    environment["HYDRA_FULL_ERROR"] = "1"
    environment["FACEANALYSIS_CPU"] = "1"
    environment["COMET_DISABLE_AUTO_LOGGING"] = "1"
    environment["COMET_LOGGING_CONSOLE"] = "ERROR"
    environment["ACCELERATE_LOG_LEVEL"] = "error"
    environment["TRANSFORMERS_VERBOSITY"] = "error"
    environment["DIFFUSERS_VERBOSITY"] = "error"

    jobs = []
    # PM control is checkpoint-independent; save time by generating it once.
    requested = [
        (mode, step)
        for mode in modes
        for step in steps
        if mode != "pmControl50" or step == min(steps)
    ]
    for mode_name, step in requested:
        checkpoint = checkpoint_for_step(
            run_dir,
            training_manifest["run_name"],
            step,
            checkpoint_every,
        )
        val_dir, job_manifest, command = command_for(
            run_dir, training_manifest, mode_name, step, checkpoint
        )
        jobs.append(str(val_dir))
        print(shlex.join(command), flush=True)
        if args.dry_run:
            continue
        log_path = val_dir / "validation.log"
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=PROJECT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        job_manifest["status"] = (
            "completed" if completed.returncode == 0 else "failed"
        )
        job_manifest["returncode"] = completed.returncode
        (val_dir / "validation_manifest.json").write_text(
            json.dumps(job_manifest, indent=2) + "\n", encoding="utf-8"
        )
        if completed.returncode:
            print(f"Validation failed: {val_dir}; see {log_path}", file=sys.stderr)
            return completed.returncode
        rendered_images = sorted((val_dir / "outputs").rglob("*.png"))
        if not rendered_images:
            raise FileNotFoundError(
                f"Validation completed without local images under: {val_dir / 'outputs'}"
            )
        uploaded_names = upload_images_to_comet(
            training_manifest["comet_experiment_key"],
            training_manifest["run_name"],
            mode_name,
            step,
            rendered_images,
        )
        job_manifest["comet_upload_status"] = "completed"
        job_manifest["comet_uploaded_names"] = uploaded_names
        (val_dir / "validation_manifest.json").write_text(
            json.dumps(job_manifest, indent=2) + "\n", encoding="utf-8"
        )
        if (
            job_manifest["mode_config"]["use_branched_attention"]
            and job_manifest["automatic_bboxes_this_job"]
        ):
            generated_bbox = val_dir / "gen_bboxes_auto.json"
            if not generated_bbox.exists():
                raise FileNotFoundError(
                    f"Automatic bbox job did not create {generated_bbox}"
                )
            fixed_bbox = run_dir / "validation" / "fixed_gen_bboxes.json"
            shutil.copy2(generated_bbox, fixed_bbox)
            print(f"Locked recurring generation bboxes: {fixed_bbox}", flush=True)

    (run_dir / "validation" / "suite_manifest.json").write_text(
        json.dumps({"jobs": jobs, "requested_steps": steps, "requested_modes": modes}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
