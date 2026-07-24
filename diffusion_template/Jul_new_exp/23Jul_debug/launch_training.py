#!/usr/bin/env python3
"""Resolve and launch one reproducible NN3a training arm.

Training and validation are deliberately decoupled while the long production
run shares the GPU.  This command saves weights at steps 200/400/600 without
running the memory-heavy native validation.  ``validate_checkpoints.py`` then
evaluates step 0/200/400/600 and logs collision-proof image names.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
REGISTRY = HERE / "architecture_registry.json"
DATA = HERE / "data" / "id_00081_1017318003459"
ONE_ID_DATA = HERE / "data" / "one_id_nm0005092"
ONE_ID_IMAGES = Path("/home/niko/rsrch/dataset_full/one_id/nm0005092_adj")
ONE_ID_FULL_JSON = ONE_ID_DATA / "full18_no_validation_train.json"
EXPERIMENTS = HERE / "experiments"
EXPERIMENTS_4K = HERE / "experiments_4k"
PM_PATH = Path("/home/niko/models/PhotoMaker-V2/photomaker-v2.bin")
ENV_BIN = Path("/home/niko/miniconda3/envs/photomaker_NS/bin")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("architecture")
    parser.add_argument(
        "--run-id",
        help=(
            "Effective experiment ID. The positional architecture remains the "
            "base registry recipe copied into the run manifest."
        ),
    )
    parser.add_argument(
        "--protocol-id",
        choices=("short", "4k"),
        help="Override the registry recipe's default protocol.",
    )
    parser.add_argument(
        "--dataset-profile",
        choices=(
            "cosmic_large_id00081",
            "one_id_nm0005092_subset8_distinct",
            "one_id_nm0005092_full18_heldout_distinct",
        ),
        help="Run a base architecture under one audited dataset profile.",
    )
    parser.add_argument(
        "--port-slot",
        type=int,
        choices=(0, 1),
        help="Training/validation port lane for paired execution.",
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--resume-run-dir",
        type=Path,
        help="Resume the latest full checkpoint in an interrupted local run.",
    )
    parser.add_argument(
        "--smoke-steps",
        type=int,
        help=(
            "Run a short console-only training preflight without checkpoints "
            "or Comet; intended for new forward/loss code."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.run_dir is not None and args.resume_run_dir is not None:
        parser.error("--run-dir and --resume-run-dir are mutually exclusive")
    if args.dry_run and args.resume_run_dir is not None:
        parser.error("--dry-run cannot be combined with --resume-run-dir")
    if args.smoke_steps is not None and args.smoke_steps < 1:
        parser.error("--smoke-steps must be positive")
    if args.smoke_steps is not None and args.resume_run_dir is not None:
        parser.error("--smoke-steps cannot be combined with --resume-run-dir")
    if args.run_id and not re.fullmatch(r"[A-Za-z0-9_]+", args.run_id):
        parser.error("--run-id may contain only letters, digits, and underscores")
    return args


def checkpoint_epoch(path: Path) -> int:
    return int(path.stem.removeprefix("checkpoint-epoch"))


def main() -> int:
    args = parse_args()
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    if args.architecture not in registry["architectures"]:
        choices = ", ".join(registry["architectures"])
        raise SystemExit(f"Unknown architecture {args.architecture!r}; choose: {choices}")
    base_architecture_id = args.architecture
    architecture_id = args.run_id or base_architecture_id
    spec = dict(registry["architectures"][base_architecture_id])
    protocol_id = args.protocol_id or spec.get("protocol_id", "short")
    protocol = (
        registry["protocol_4k"] if protocol_id == "4k" else registry["protocol"]
    )
    optimizer_steps = int(protocol["optimizer_steps"])
    checkpoint_every = int(protocol["checkpoint_every"])
    if optimizer_steps % checkpoint_every:
        raise SystemExit(
            f"optimizer_steps={optimizer_steps} is not divisible by "
            f"checkpoint_every={checkpoint_every}"
        )
    dataset_profile = (
        args.dataset_profile
        or spec.get("dataset_profile", "cosmic_large_id00081")
    )
    port_slot = (
        int(args.port_slot)
        if args.port_slot is not None
        else int(spec.get("port_slot", 0))
    )
    distinct_pairing = False

    if dataset_profile == "cosmic_large_id00081":
        selected_data = DATA / "train_8refs.json"
        selected_data_manifest = DATA / "split_manifest.json"
        identity_tag = "id00081"
        dataset_overrides = [
            "train_dataset_name=cosmic_large_neb",
            f"datasets.train.cosmic_large_neb.data_json_pth={selected_data}",
            "datasets.train.cosmic_large_neb.num_refs=1",
            "+datasets.train.cosmic_large_neb.ref_crop_margin_min=0.2",
            "+datasets.train.cosmic_large_neb.ref_crop_margin_max=0.6",
            "+datasets.train.cosmic_large_neb.ref_downscale_jitter=0.5",
        ]
    elif dataset_profile in {
        "one_id_nm0005092_subset8",
        "one_id_nm0005092_subset8_distinct",
        "one_id_nm0005092_full18_heldout_distinct",
    }:
        distinct_pairing = dataset_profile.endswith("_distinct")
        full_dataset = (
            dataset_profile == "one_id_nm0005092_full18_heldout_distinct"
        )
        selected_data = (
            ONE_ID_FULL_JSON
            if full_dataset
            else ONE_ID_DATA / "subset8_train.json"
        )
        selected_data_manifest = (
            ONE_ID_DATA / "full18_heldout_manifest.json"
            if full_dataset
            else ONE_ID_DATA / "subset_manifest.json"
        )
        if full_dataset:
            identity_tag = "nm0005092_oneid18_distinct_heldout51"
        else:
            identity_tag = (
                "nm0005092_oneid8_distinct"
                if distinct_pairing
                else "nm0005092_oneid8_sameimage"
            )
        dataset_overrides = [
            "train_dataset_name=one_id",
            f"datasets.train.one_id.cosmic_json_pth={selected_data}",
            f"datasets.train.one_id.images_path={ONE_ID_IMAGES}",
            "datasets.train.one_id.num_refs=1",
            f"train_on_separate_image={'true' if distinct_pairing else 'false'}",
        ]
    else:
        raise SystemExit(f"Unsupported dataset profile: {dataset_profile}")

    if not selected_data.exists() or not selected_data_manifest.exists():
        raise SystemExit(
            f"Missing data bundle for {dataset_profile}: "
            f"{selected_data} / {selected_data_manifest}"
        )
    if not PM_PATH.exists():
        raise SystemExit(f"PhotoMaker checkpoint missing: {PM_PATH}")
    if not (ENV_BIN / "python").exists():
        raise SystemExit(f"photomaker_NS missing: {ENV_BIN}")
    is_smoke = args.smoke_steps is not None
    if not is_smoke and not os.environ.get("COMET_API_KEY"):
        raise SystemExit("COMET_API_KEY is not available to the launcher")

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    is_resume = args.resume_run_dir is not None
    if is_resume:
        run_dir = args.resume_run_dir.resolve()
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.exists():
            raise SystemExit(f"Resume manifest missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("base_architecture_id", manifest.get("architecture_id")) != (
            base_architecture_id
        ):
            raise SystemExit(
                "Resume architecture mismatch: "
                f"{manifest.get('base_architecture_id', manifest.get('architecture_id'))!r} "
                f"!= {base_architecture_id!r}"
            )
        if manifest.get("architecture_id") != architecture_id:
            raise SystemExit(
                "Resume run-ID mismatch: "
                f"{manifest.get('architecture_id')!r} != {architecture_id!r}"
            )
        run_name = manifest["run_name"]
    else:
        prefix = (
            "SMOKE_23Jul"
            if is_smoke
            else "23Jul4k"
            if protocol_id == "4k"
            else "23Jul"
        )
        suffix = (
            f"smoke{args.smoke_steps}"
            if is_smoke
            else f"s0_{optimizer_steps}"
        )
        run_name = f"{prefix}_{architecture_id}_{identity_tag}_{suffix}__{stamp}"
        if args.dry_run:
            default_root = HERE / "dry_runs"
        elif is_smoke:
            default_root = HERE / "smoke_tests"
        else:
            default_root = EXPERIMENTS_4K if protocol_id == "4k" else EXPERIMENTS
        run_dir = (args.run_dir or (default_root / f"{stamp}__{run_name}")).resolve()
        if run_dir.exists() and any(run_dir.iterdir()):
            raise SystemExit(f"Refusing to reuse non-empty run directory: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = run_dir / "checkpoints"
    debug_dir = run_dir / "debug"
    checkpoints.mkdir(exist_ok=True)
    debug_dir.mkdir(exist_ok=True)

    model_target = "training_lab.nn3a_lab_model.NN3aTrainingLabModel"
    pairing_audit = None
    if distinct_pairing or dataset_profile == "cosmic_large_id00081":
        audit_env = os.environ.copy()
        audit_env["PYTHONPATH"] = os.pathsep.join(
            [
                str(PROJECT),
                str(HERE),
                audit_env.get("PYTHONPATH", ""),
            ]
        )
        audit_command = [
            str(ENV_BIN / "python"),
            str(
                HERE
                / (
                    "audit_one_id_pairing.py"
                    if distinct_pairing
                    else "audit_cosmic_pairing.py"
                )
            ),
        ]
        if distinct_pairing:
            audit_command.extend(
                [
                    "--data-json",
                    str(selected_data),
                    "--seeds-per-target",
                    "8",
                ]
            )
        audit_result = subprocess.run(
            audit_command,
            cwd=HERE,
            env=audit_env,
            text=True,
            capture_output=True,
        )
        if audit_result.returncode != 0:
            raise SystemExit(
                "Target/reference pairing audit failed before launch:\n"
                f"{audit_result.stdout}\n{audit_result.stderr}"
            )
        pairing_audit = json.loads(audit_result.stdout)
    epoch_len = int(args.smoke_steps) if is_smoke else checkpoint_every
    n_epochs = 1 if is_smoke else optimizer_steps // checkpoint_every
    log_step = 1 if is_smoke else int(protocol.get("log_step", 25))
    overrides = [
        "datasets=all_datasets",
        *dataset_overrides,
        "val_datasets_names=[]",
        "inference_metrics=[]",
        f"trainer.epoch_len={epoch_len}",
        f"trainer.n_epochs={n_epochs}",
        "trainer.seed=0",
        f"trainer.log_step={log_step}",
        f"trainer.save_dir={checkpoints}",
        "dataloaders.train.batch_size=1",
        "dataloaders.train.grad_accum_enabled=false",
        "dataloaders.train.batch_size_eff=1",
        "dataloaders.train.num_workers=2",
        f"model._target_={model_target}",
        f"+model.lab_train_scope={spec['lab_train_scope']}",
        f"+model.lab_optimizer_recipe={spec.get('lab_optimizer_recipe', 'production')}",
        "model.rank=32",
        f"model.photomaker_path={PM_PATH}",
        "model.weight_dtype=bf16",
        "pipeline.variant=null",
        f"lr_for_lora={spec['lr_for_lora']}",
        f"ba_noise_lr_scale={spec['ba_noise_lr_scale']}",
        f"loss_kind={spec['loss_kind']}",
        f"lambda_face={spec['lambda_face']}",
        f"optimizer.weight_decay={spec['optimizer_weight_decay']}",
        "trainer.max_grad_norm=1.0",
        "trainer.masked_loss_step=2",
        f"lr_scheduler.warmup_steps={protocol['warmup_steps']}",
        "automatic_bboxes=false",
        "automatic_bboxes_every_val=false",
        "force_log_first_auto_bbox=false",
        "val_debug=false",
        (
            "pretrained_model_for_validation_name_or_path=null"
            if is_smoke
            else "pretrained_model_for_validation_name_or_path=SG161222/RealVisXL_V4.0"
        ),
        f"trainer.save_period={999 if is_smoke else 1}",
        f"weights_only_save_period={0 if is_smoke else 1}",
        f"writer={'console' if is_smoke else 'cometml'}",
        f"writer.run_name={run_name}",
    ]
    for field in (
        "lab_ref_kv_lr_scale",
        "lab_ref_v_lr_scale",
        "lab_ref_q_lr_scale",
        "lab_noise_lr_scale",
        "lab_up0_lr_scale",
        "lab_pm_teacher_weight",
    ):
        if field in spec:
            overrides.append(f"+model.{field}={spec[field]}")
    if "ba_train_timestep_mode" in spec:
        overrides.append(
            f"model.ba_train_timestep_mode={spec['ba_train_timestep_mode']}"
        )
    if "lab_staged_up0_start_step" in spec:
        overrides.extend(
            [
                "trainer._target_=training_lab.nn3a_lab_trainer.NN3aLabTrainer",
                f"+lab_staged_up0_start_step={spec['lab_staged_up0_start_step']}",
            ]
        )
    resume_checkpoint = None
    if is_resume:
        checkpoint_candidates = sorted(
            checkpoints.rglob("checkpoint-epoch*.pth"), key=checkpoint_epoch
        )
        if not checkpoint_candidates:
            raise SystemExit(f"No resumable full checkpoint under: {checkpoints}")
        resume_checkpoint = checkpoint_candidates[-1]
        resume_epoch = checkpoint_epoch(resume_checkpoint)
        if resume_epoch >= optimizer_steps // checkpoint_every:
            raise SystemExit(
                f"Run already reached its final checkpoint: {resume_checkpoint}"
            )
        comet_experiment_key = manifest.get("comet_experiment_key")
        if not comet_experiment_key:
            raise SystemExit(
                "Refusing to resume without the original Comet experiment key"
            )
        overrides.extend(
            [
                "continue_run=true",
                f"saved_checkpoint={resume_checkpoint}",
                f"trainer.resume_from={resume_checkpoint.name}",
                f"cometml_id={comet_experiment_key}",
            ]
        )
    command = [
        str(ENV_BIN / "accelerate"),
        "launch",
        "--config_file=src/configs/ddp/accelerate.yaml",
        "--main_process_ip=127.0.0.1",
        f"--main_process_port={29731 + port_slot}",
        "--num_processes=1",
        "train.py",
        "--config-name=one_id_ba_N3a_new1",
        *overrides,
    ]
    if is_resume:
        attempt = {
            "created_utc": stamp,
            "status": "prepared",
            "checkpoint": str(resume_checkpoint),
            "command": command,
        }
        manifest.setdefault("resume_attempts", []).append(attempt)
        manifest["status"] = "resuming"
    else:
        manifest = {
            "created_utc": stamp,
            "status": "prepared",
            "run_name": run_name,
            "architecture_id": architecture_id,
            "base_architecture_id": base_architecture_id,
            "architecture": spec,
            "dataset_profile": dataset_profile,
            "protocol": protocol,
            "protocol_id": protocol_id,
            "port_slot": port_slot,
            "validation_port_base": 29800 + 100 * port_slot,
            "smoke_steps": args.smoke_steps,
            "selected_id_data": str(selected_data_manifest),
            "pairing_audit": pairing_audit,
            "production_config": "src/configs/one_id_ba_N3a_new1.yaml",
            "model_target": model_target,
            "command": command,
            "validation_note": (
                "Console-only forward/backward preflight; no checkpoints or "
                "validation are expected."
                if is_smoke
                else (
                    "Training-native validation disabled. Checkpoints are "
                    "evaluated by the console-writer validation watcher and "
                    "uploaded into this training experiment."
                    + (
                        " The full OneID training JSON was filtered to exclude "
                        "51.jpg, preserving the sole validation reference as "
                        "held out."
                        if dataset_profile
                        == "one_id_nm0005092_full18_heldout_distinct"
                        else ""
                    )
                )
            ),
        }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "resolved_command.sh").write_text(
        "#!/usr/bin/env bash\n" + shlex.join(command) + "\n", encoding="utf-8"
    )
    (run_dir / "STATUS.md").write_text(
        f"# {run_name}\n\nStatus: prepared\n\n"
        f"Architecture: `{architecture_id}` (base recipe: "
        f"`{base_architecture_id}`)\n\n"
        + (
            f"Smoke preflight steps: {args.smoke_steps}; no checkpoints expected.\n"
            if is_smoke
            else (
                "Expected checkpoints: "
                + ", ".join(
                    str(step)
                    for step in protocol["validation_steps"]
                    if int(step) > 0
                )
                + ".\n"
            )
        ),
        encoding="utf-8",
    )
    if not is_resume:
        source_snapshot = run_dir / "source_snapshot"
        source_snapshot.mkdir()
        for source in (
            REGISTRY,
            HERE / "launch_training.py",
            HERE / "launch_validation.py",
            HERE / "summarize_run.py",
            HERE / "checkpoint_diagnostics.py",
            HERE / "log_validation_step_metrics.py",
            HERE / "audit_comet_unity.py",
            HERE
            / (
                "audit_one_id_pairing.py"
                if distinct_pairing
                else "audit_cosmic_pairing.py"
            ),
            HERE / "training_lab" / "nn3a_lab_model.py",
            selected_data,
            selected_data_manifest,
        ):
            shutil.copy2(source, source_snapshot / source.name)

    print(run_dir)
    print(shlex.join(command))
    if args.dry_run:
        return 0

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

    log_path = run_dir / "training.log"
    manifest["status"] = "running"
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    with log_path.open("a" if is_resume else "w", encoding="utf-8") as log:
        if is_resume:
            log.write(
                f"\n[23Jul resume {stamp}] checkpoint={resume_checkpoint}\n"
            )
            log.flush()
        try:
            completed = subprocess.run(
                command,
                cwd=PROJECT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
            returncode = completed.returncode
        except KeyboardInterrupt:
            returncode = 130
            log.write("\n[23Jul launcher] interrupted after checkpoint boundary.\n")
            log.flush()
    latest_manifest = json.loads(
        (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    latest_manifest["status"] = "completed" if returncode == 0 else "interrupted"
    latest_manifest["returncode"] = returncode
    if is_resume:
        attempt["status"] = latest_manifest["status"]
        attempt["returncode"] = returncode
        latest_manifest["resume_attempts"] = manifest["resume_attempts"]
    manifest = latest_manifest
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "STATUS.md").write_text(
        f"# {run_name}\n\nStatus: {manifest['status']}\n\n"
        f"Return code: `{returncode}`\n\n"
        f"Log: `{log_path}`\n",
        encoding="utf-8",
    )
    return returncode


if __name__ == "__main__":
    sys.exit(main())
