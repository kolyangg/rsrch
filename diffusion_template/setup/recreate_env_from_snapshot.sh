#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <snapshot_dir> [new_env_name]"
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is not available in PATH."
  exit 1
fi

SNAPSHOT_DIR_INPUT="$1"
if [[ ! -d "${SNAPSHOT_DIR_INPUT}" ]]; then
  echo "ERROR: snapshot_dir does not exist: ${SNAPSHOT_DIR_INPUT}"
  exit 1
fi
# Canonicalize to an absolute path so generated helper scripts/hooks
# never rely on caller CWD.
SNAPSHOT_DIR="$(cd "${SNAPSHOT_DIR_INPUT}" && pwd -P)"

EXPLICIT_FILE="${SNAPSHOT_DIR}/conda_explicit.txt"
NOBUILDS_FILE="${SNAPSHOT_DIR}/environment_nobuilds.yml"
PIP_FILE="${SNAPSHOT_DIR}/pip_freeze.txt"

if [[ -n "${2:-}" ]]; then
  TARGET_ENV="$2"
else
  TARGET_ENV=""
  if [[ -f "${SNAPSHOT_DIR}/snapshot_meta.txt" ]]; then
    TARGET_ENV="$(grep -E '^env_name=' "${SNAPSHOT_DIR}/snapshot_meta.txt" | head -n1 | cut -d'=' -f2- || true)"
  fi
  if [[ -z "${TARGET_ENV}" && -f "${NOBUILDS_FILE}" ]]; then
    TARGET_ENV="$(grep -E '^name:' "${NOBUILDS_FILE}" | head -n1 | cut -d':' -f2- | xargs || true)"
  fi
  if [[ -z "${TARGET_ENV}" ]]; then
    TARGET_ENV="env_restored"
  fi
fi

echo "Restoring environment:"
echo "  snapshot: ${SNAPSHOT_DIR}"
echo "  target_env: ${TARGET_ENV}"

# Keep conda operations isolated from active runtime overlays (e.g. _gcc_runtime),
# otherwise libmamba can fail to load due to incompatible libstdc++/CXXABI.
_clean_ld_library_path() {
  local input="${1:-}"
  local out=""
  local part=""
  IFS=':' read -r -a _parts <<< "${input}"
  for part in "${_parts[@]}"; do
    [[ -z "${part}" ]] && continue
    case "${part}" in
      */_gcc_runtime/lib* ) continue ;;
    esac
    if [[ -z "${out}" ]]; then
      out="${part}"
    else
      out="${out}:${part}"
    fi
  done
  printf '%s' "${out}"
}

CONDA_CLEAN_LD_LIBRARY_PATH="$(_clean_ld_library_path "${LD_LIBRARY_PATH:-}")"
run_conda() {
  CONDA_SOLVER=classic LD_PRELOAD= LD_LIBRARY_PATH="${CONDA_CLEAN_LD_LIBRARY_PATH}" conda "$@"
}

if run_conda env list | awk '{print $1}' | grep -Fxq "${TARGET_ENV}"; then
  echo "ERROR: conda env '${TARGET_ENV}' already exists. Choose another name."
  exit 1
fi

set +e
if [[ -f "${EXPLICIT_FILE}" ]]; then
  echo "[1/3] Trying exact conda restore from ${EXPLICIT_FILE}"
  run_conda create -y -n "${TARGET_ENV}" --file "${EXPLICIT_FILE}"
  STATUS=$?
else
  STATUS=1
fi
set -e

if [[ ${STATUS} -ne 0 ]]; then
  if [[ ! -f "${NOBUILDS_FILE}" ]]; then
    echo "ERROR: exact conda restore failed and ${NOBUILDS_FILE} is missing."
    exit 1
  fi
  echo "[1/3] Exact restore failed (likely OS/arch/build mismatch)."
  echo "[2/3] Falling back to portable conda spec ${NOBUILDS_FILE}"
  run_conda env create -n "${TARGET_ENV}" -f "${NOBUILDS_FILE}"
fi

# Enter activated env with cleaned loader variables.
unset LD_PRELOAD
if [[ -n "${CONDA_CLEAN_LD_LIBRARY_PATH}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_CLEAN_LD_LIBRARY_PATH}"
else
  unset LD_LIBRARY_PATH
fi

eval "$(run_conda shell.bash hook)"
conda activate "${TARGET_ENV}"

if [[ -f "${PIP_FILE}" ]]; then
  echo "[3/8] Installing pip packages from ${PIP_FILE}"
  python -m pip install --upgrade pip
  CLIP_SOURCE="${CLIP_SOURCE:-git+https://github.com/openai/CLIP.git}"
  TMP_PIP_FILE="$(mktemp)"
  FAILED_REQS_FILE="${SNAPSHOT_DIR}/pip_failed_requirements.txt"
  : > "${FAILED_REQS_FILE}"

  # Normalize freeze file for cross-machine restore:
  # - remove clip requirements (installed separately with --no-deps below)
  # - drop comments/empty lines
  # - map local file refs ("pkg @ file://...") to bare package name
  # - deduplicate while preserving first occurrence
  awk '
    {
      gsub(/\r$/, "", $0)
      if ($0 ~ /^[[:space:]]*$/) next
      if ($0 ~ /^[[:space:]]*#/) next
      if ($0 ~ /^clip([[:space:]]*@|==|>=|<=|~=|>|<|$)/) next
      if ($0 ~ /@ file:\/\//) {
        n = split($0, parts, " @ ")
        if (n >= 1 && parts[1] != "") {
          $0 = parts[1]
        } else {
          next
        }
      }
      if (!seen[$0]++) print $0
    }
  ' "${PIP_FILE}" > "${TMP_PIP_FILE}"

  if ! command -v uv >/dev/null 2>&1; then
    echo "[3/8] Installing uv for faster pip restore"
    python -m pip install --upgrade uv
  fi

  PY_BIN="$(command -v python)"
  echo "[3/8] Installing with uv (bulk pass)"
  set +e
  uv pip install --python "${PY_BIN}" -r "${TMP_PIP_FILE}"
  UV_STATUS=$?
  set -e

  if [[ ${UV_STATUS} -ne 0 ]]; then
    echo "[3/8] Bulk uv install failed. Retrying package-by-package to maximize coverage."
    while IFS= read -r req; do
      [[ -z "${req}" ]] && continue
      set +e
      uv pip install --python "${PY_BIN}" "${req}"
      ONE_STATUS=$?
      set -e
      if [[ ${ONE_STATUS} -ne 0 ]]; then
        echo "${req}" >> "${FAILED_REQS_FILE}"
      fi
    done < "${TMP_PIP_FILE}"
  fi

  rm -f "${TMP_PIP_FILE}"

  if [[ -s "${FAILED_REQS_FILE}" ]]; then
    echo "WARNING: Some pip requirements could not be installed."
    echo "See: ${FAILED_REQS_FILE}"
  else
    rm -f "${FAILED_REQS_FILE}"
  fi
else
  echo "[3/8] pip_freeze.txt not found; skipping pip install."
fi

# Pin huggingface-hub for consistent model loading behavior across machines.
# Default is read from snapshot; can be overridden via HF_HUB_VERSION.
HF_HUB_VERSION="${HF_HUB_VERSION:-}"
if [[ -z "${HF_HUB_VERSION}" && -f "${PIP_FILE}" ]]; then
  HF_HUB_VERSION="$(awk -F'==' '/^huggingface-hub==/ {print $2; exit}' "${PIP_FILE}" | xargs || true)"
fi
if [[ -n "${HF_HUB_VERSION}" ]]; then
  echo "[hub] Pinning huggingface-hub==${HF_HUB_VERSION}"
  PY_BIN="$(command -v python)"
  if ! command -v uv >/dev/null 2>&1; then
    python -m pip install --upgrade uv
  fi
  uv pip install --python "${PY_BIN}" --no-deps --force-reinstall "huggingface-hub==${HF_HUB_VERSION}"
  "${PY_BIN}" - <<'PY'
import importlib.metadata as md
print("huggingface-hub", md.version("huggingface-hub"))
PY
else
  echo "[hub] No huggingface-hub pin found in snapshot; skipping."
fi

# Pin ORT GPU to a CUDA-12 compatible build to avoid provider load errors
# like "libcublasLt.so.11 not found" when torch stack is CUDA 12.
SKIP_ORT_PIN="${SKIP_ORT_PIN:-0}"
if [[ "${SKIP_ORT_PIN}" != "1" ]]; then
  echo "[4/8] Pinning ONNX Runtime GPU to CUDA-12 compatible version"
  ORT_GPU_VERSION="${ORT_GPU_VERSION:-1.20.1}"
  PY_BIN="$(command -v python)"
  if ! command -v uv >/dev/null 2>&1; then
    python -m pip install --upgrade uv
  fi
  set +e
  uv pip uninstall --python "${PY_BIN}" onnxruntime onnxruntime-gpu >/dev/null 2>&1
  set -e
  uv pip install --python "${PY_BIN}" "onnxruntime-gpu==${ORT_GPU_VERSION}"
else
  echo "[4/8] Skipping ORT pin (SKIP_ORT_PIN=1)."
fi

# Optional but recommended for insightface/onnx/other compiled wheels:
# create an isolated GCC runtime overlay and export it for the shell.
SKIP_RUNTIME_OVERLAY="${SKIP_RUNTIME_OVERLAY:-0}"
RUNTIME_EXPORT_FILE="${SNAPSHOT_DIR}/activate_runtime_${TARGET_ENV}.sh"
if [[ "${SKIP_RUNTIME_OVERLAY}" != "1" ]]; then
  echo "[5/8] Preparing C++ runtime overlay (libstdc++)"
  RUNTIME_PREFIX="${RUNTIME_PREFIX:-${SNAPSHOT_DIR}/_gcc_runtime}"
  RUNTIME_CHANNEL="${RUNTIME_CHANNEL:-conda-forge}"
  RUNTIME_LIBSTDCPP_VERSION="${RUNTIME_LIBSTDCPP_VERSION:-13}"
  RUNTIME_LIBGCC_VERSION="${RUNTIME_LIBGCC_VERSION:-13}"

  if [[ ! -f "${RUNTIME_PREFIX}/lib/libstdc++.so.6" ]]; then
    echo "[5/8] Creating runtime overlay at ${RUNTIME_PREFIX}"
    run_conda create -y -p "${RUNTIME_PREFIX}" \
      -c "${RUNTIME_CHANNEL}" \
      --override-channels \
      "libstdcxx-ng=${RUNTIME_LIBSTDCPP_VERSION}" \
      "libgcc-ng=${RUNTIME_LIBGCC_VERSION}"
  else
    echo "[5/8] Reusing existing runtime overlay at ${RUNTIME_PREFIX}"
  fi

  cat > "${RUNTIME_EXPORT_FILE}" <<EOF
#!/usr/bin/env bash
# Auto-generated by recreate_env_from_snapshot.sh for env: ${TARGET_ENV}
# Populate NVIDIA pip CUDA runtime lib dirs (site-packages/nvidia/*/lib) if present.
if command -v python >/dev/null 2>&1; then
  _nvidia_pip_libs="\$(python - <<'PY'
import glob
import os
import site

paths = []
for base in site.getsitepackages():
    for p in glob.glob(os.path.join(base, "nvidia", "*", "lib")):
        if os.path.isdir(p):
            paths.append(p)
seen = set()
ordered = []
for p in paths:
    if p not in seen:
        seen.add(p)
        ordered.append(p)
print(":".join(ordered))
PY
)"
  if [[ -n "\${_nvidia_pip_libs}" ]]; then
    export NVIDIA_PIP_LIBS="\${_nvidia_pip_libs}"
  fi
fi

if [[ -n "\${NVIDIA_PIP_LIBS:-}" ]]; then
  export LD_LIBRARY_PATH="${RUNTIME_PREFIX}/lib:\${NVIDIA_PIP_LIBS}:\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH:-}"
else
  export LD_LIBRARY_PATH="${RUNTIME_PREFIX}/lib:\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH:-}"
fi
export LD_PRELOAD="${RUNTIME_PREFIX}/lib/libstdc++.so.6:\${LD_PRELOAD:-}"
EOF
  chmod +x "${RUNTIME_EXPORT_FILE}"

  if [[ -f "${RUNTIME_PREFIX}/lib/libstdc++.so.6" ]]; then
    if strings "${RUNTIME_PREFIX}/lib/libstdc++.so.6" | grep -q "GLIBCXX_3.4.32"; then
      echo "[5/8] Runtime overlay OK (GLIBCXX_3.4.32 found)."
    else
      echo "WARNING: ${RUNTIME_PREFIX}/lib/libstdc++.so.6 does not expose GLIBCXX_3.4.32"
    fi
  fi
else
  echo "[5/8] Skipping runtime overlay (SKIP_RUNTIME_OVERLAY=1)."
fi

# Prefetch InsightFace model to avoid first-run network failures during training.
SKIP_INSIGHTFACE_PREFETCH="${SKIP_INSIGHTFACE_PREFETCH:-0}"
if [[ "${SKIP_INSIGHTFACE_PREFETCH}" != "1" ]]; then
  echo "[6/8] Prefetching InsightFace model"
  INSIGHTFACE_ROOT="${INSIGHTFACE_ROOT:-${HOME}/.insightface}"
  INSIGHTFACE_MODEL_NAME="${INSIGHTFACE_MODEL_NAME:-buffalo_l}"
  INSIGHTFACE_MODELS_DIR="${INSIGHTFACE_ROOT}/models"
  INSIGHTFACE_MODEL_DIR="${INSIGHTFACE_MODELS_DIR}/${INSIGHTFACE_MODEL_NAME}"
  INSIGHTFACE_ZIP="${INSIGHTFACE_MODELS_DIR}/${INSIGHTFACE_MODEL_NAME}.zip"
  INSIGHTFACE_URL="${INSIGHTFACE_URL:-https://github.com/deepinsight/insightface/releases/download/v0.7/${INSIGHTFACE_MODEL_NAME}.zip}"

  mkdir -p "${INSIGHTFACE_MODELS_DIR}"
  if [[ -d "${INSIGHTFACE_MODEL_DIR}" ]]; then
    echo "[6/8] InsightFace model already present: ${INSIGHTFACE_MODEL_DIR}"
  else
    echo "[6/8] Downloading ${INSIGHTFACE_URL}"
    if command -v wget >/dev/null 2>&1; then
      wget -c --tries=20 --timeout=30 -O "${INSIGHTFACE_ZIP}" "${INSIGHTFACE_URL}"
    elif command -v curl >/dev/null 2>&1; then
      curl -L --retry 20 --retry-all-errors --connect-timeout 30 -o "${INSIGHTFACE_ZIP}" "${INSIGHTFACE_URL}"
    else
      python - "${INSIGHTFACE_URL}" "${INSIGHTFACE_ZIP}" <<'PY'
import pathlib
import sys
import urllib.request

url, out_path = sys.argv[1], sys.argv[2]
pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
urllib.request.urlretrieve(url, out_path)
PY
    fi

    python - "${INSIGHTFACE_ZIP}" "${INSIGHTFACE_MODELS_DIR}" <<'PY'
import pathlib
import sys
import zipfile

zip_path, out_dir = sys.argv[1], sys.argv[2]
pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(out_dir)
print(f"Extracted {zip_path} -> {out_dir}")
PY
  fi
else
  echo "[6/8] Skipping InsightFace prefetch (SKIP_INSIGHTFACE_PREFETCH=1)."
fi

# Prefetch PhotoMaker-V2 weights and create compatibility symlink for configs
# that hardcode a specific HF snapshot hash.
SKIP_PHOTOMAKER_PREFETCH="${SKIP_PHOTOMAKER_PREFETCH:-0}"
PHOTOMAKER_EXPORT_FILE="${SNAPSHOT_DIR}/activate_photomaker_${TARGET_ENV}.sh"
if [[ "${SKIP_PHOTOMAKER_PREFETCH}" != "1" ]]; then
  echo "[7/8] Prefetching PhotoMaker-V2 weights"
  PHOTOMAKER_REPO_ID="${PHOTOMAKER_REPO_ID:-TencentARC/PhotoMaker-V2}"
  PHOTOMAKER_FILENAME="${PHOTOMAKER_FILENAME:-photomaker-v2.bin}"
  PHOTOMAKER_EXPECTED_SNAPSHOT="${PHOTOMAKER_EXPECTED_SNAPSHOT:-f5a1e5155dc02166253fa7e29d13519f5ba22eac}"
  PHOTOMAKER_CACHE_ROOT="${PHOTOMAKER_CACHE_ROOT:-${HOME}/.cache/huggingface/hub}"

  PHOTOMAKER_REAL_PATH="$(PHOTOMAKER_REPO_ID="${PHOTOMAKER_REPO_ID}" PHOTOMAKER_FILENAME="${PHOTOMAKER_FILENAME}" python - <<'PY'
import os
import time
from huggingface_hub import hf_hub_download

repo_id = os.environ["PHOTOMAKER_REPO_ID"]
filename = os.environ["PHOTOMAKER_FILENAME"]

last_err = None
for _ in range(5):
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            resume_download=True,
        )
        print(path)
        raise SystemExit(0)
    except Exception as exc:
        last_err = exc
        time.sleep(3)
raise RuntimeError(f"Failed to download {repo_id}/{filename}: {last_err}")
PY
)"

  PHOTOMAKER_EXPECTED_DIR="${PHOTOMAKER_CACHE_ROOT}/models--TencentARC--PhotoMaker-V2/snapshots/${PHOTOMAKER_EXPECTED_SNAPSHOT}"
  PHOTOMAKER_EXPECTED_PATH="${PHOTOMAKER_EXPECTED_DIR}/${PHOTOMAKER_FILENAME}"
  mkdir -p "${PHOTOMAKER_EXPECTED_DIR}"
  ln -sf "${PHOTOMAKER_REAL_PATH}" "${PHOTOMAKER_EXPECTED_PATH}"

  cat > "${PHOTOMAKER_EXPORT_FILE}" <<EOF
#!/usr/bin/env bash
# Auto-generated by recreate_env_from_snapshot.sh for env: ${TARGET_ENV}
export PHOTOMAKER_PATH="${PHOTOMAKER_REAL_PATH}"
export PM_PATH="${PHOTOMAKER_REAL_PATH}"
EOF
  chmod +x "${PHOTOMAKER_EXPORT_FILE}"
  echo "[7/8] PhotoMaker path: ${PHOTOMAKER_REAL_PATH}"
  echo "[7/8] Compatibility symlink: ${PHOTOMAKER_EXPECTED_PATH}"
else
  echo "[7/8] Skipping PhotoMaker prefetch (SKIP_PHOTOMAKER_PREFETCH=1)."
fi

# Prefetch RealVisXL fp16 snapshot to avoid runtime
# "variant=fp16 ... no such modeling files" errors on fresh machines.
SKIP_REALVIS_PREFETCH="${SKIP_REALVIS_PREFETCH:-0}"
if [[ "${SKIP_REALVIS_PREFETCH}" != "1" ]]; then
  echo "[rv] Prefetching RealVisXL fp16 snapshot"
  REALVIS_REPO_ID="${REALVIS_REPO_ID:-SG161222/RealVisXL_V4.0}"
  REALVIS_CACHE_ROOT="${REALVIS_CACHE_ROOT:-${HF_HOME:-${HOME}/.cache/huggingface}}"
  REALVIS_CACHE_ROOT="${REALVIS_CACHE_ROOT/#\~/${HOME}}"
  REALVIS_MODEL_DIR="${REALVIS_CACHE_ROOT}/hub/models--SG161222--RealVisXL_V4.0"
  REALVIS_FORCE_CLEAN="${REALVIS_FORCE_CLEAN:-1}"
  if [[ "${REALVIS_FORCE_CLEAN}" == "1" ]]; then
    rm -rf "${REALVIS_MODEL_DIR}"
  fi

  PY_BIN="$(command -v python)"
  REALVIS_REPO_ID="${REALVIS_REPO_ID}" REALVIS_CACHE_ROOT="${REALVIS_CACHE_ROOT}" "${PY_BIN}" - <<'PY'
import os
from huggingface_hub import snapshot_download

repo_id = os.environ["REALVIS_REPO_ID"]
cache_dir = os.environ["REALVIS_CACHE_ROOT"]
snapshot_path = snapshot_download(
    repo_id=repo_id,
    cache_dir=cache_dir,
    resume_download=True,
    allow_patterns=[
        "model_index.json",
        "scheduler/*",
        "tokenizer/*",
        "tokenizer_2/*",
        "text_encoder/model.fp16.safetensors",
        "text_encoder_2/model.fp16.safetensors",
        "unet/diffusion_pytorch_model.fp16.safetensors",
        "vae/diffusion_pytorch_model.fp16.safetensors",
    ],
)
print(f"RealVis fp16 snapshot: {snapshot_path}")
PY
else
  echo "[rv] Skipping RealVis prefetch (SKIP_REALVIS_PREFETCH=1)."
fi

# Prefetch CLIP model weights used by text_sim metric to avoid
# multi-rank first-download race that can trigger SHA256 mismatch.
SKIP_CLIP_PREFETCH="${SKIP_CLIP_PREFETCH:-0}"
if [[ "${SKIP_CLIP_PREFETCH}" != "1" ]]; then
  echo "[8/8] Prefetching OpenAI CLIP model"
  CLIP_MODEL_NAME="${CLIP_MODEL_NAME:-ViT-L/14@336px}"
  CLIP_SOURCE="${CLIP_SOURCE:-git+https://github.com/openai/CLIP.git}"
  CLIP_CACHE_DIR="${CLIP_CACHE_DIR:-${HOME}/.cache/clip}"
  CLIP_CACHE_DIR="${CLIP_CACHE_DIR/#\~/${HOME}}"
  CLIP_PREFETCH_RETRIES="${CLIP_PREFETCH_RETRIES:-10}"
  mkdir -p "${CLIP_CACHE_DIR}"
  PY_BIN="$(command -v python)"

  if ! "${PY_BIN}" - <<'PY' >/dev/null 2>&1
import clip
required = ("load", "tokenize")
ok = all(hasattr(clip, name) for name in required)
raise SystemExit(0 if ok else 1)
PY
  then
    echo "[8/8] Detected non-OpenAI clip package or missing API"
    echo "[8/8] Installing CLIP from ${CLIP_SOURCE}"
    "${PY_BIN}" -m pip uninstall -y clip >/dev/null 2>&1 || true
    if ! command -v uv >/dev/null 2>&1; then
      "${PY_BIN}" -m pip install --upgrade uv
    fi
    uv pip install --python "${PY_BIN}" --no-deps "${CLIP_SOURCE}"
  fi
  CLIP_MODEL_NAME="${CLIP_MODEL_NAME}" CLIP_CACHE_DIR="${CLIP_CACHE_DIR}" CLIP_PREFETCH_RETRIES="${CLIP_PREFETCH_RETRIES}" "${PY_BIN}" - <<'PY'
import os
import time
import clip

model_name = os.environ["CLIP_MODEL_NAME"]
cache_dir = os.path.expanduser(os.environ["CLIP_CACHE_DIR"])
retries = int(os.environ.get("CLIP_PREFETCH_RETRIES", "10"))

if hasattr(clip, "available_models"):
    available = clip.available_models()
    if isinstance(available, (list, tuple)) and available and model_name not in available:
        raise RuntimeError(f"Unknown CLIP model: {model_name}. Available: {available}")

last_error = None
for attempt in range(1, retries + 1):
    try:
        clip.load(model_name, device="cpu", download_root=cache_dir)
        print(f"CLIP prefetch OK: {model_name} -> {cache_dir}")
        raise SystemExit(0)
    except RuntimeError as exc:
        last_error = exc
        text = str(exc).lower()
        if "sha256 checksum" in text or "checksum does not not match" in text:
            for name in os.listdir(cache_dir):
                if name.endswith(".pt"):
                    try:
                        os.remove(os.path.join(cache_dir, name))
                    except OSError:
                        pass
        time.sleep(2)

raise RuntimeError(f"Failed to prefetch CLIP model {model_name} after {retries} attempts: {last_error}")
PY
else
  echo "[8/8] Skipping CLIP prefetch (SKIP_CLIP_PREFETCH=1)."
fi

# Install conda activation hooks so runtime overlay / PhotoMaker path are
# automatically applied on every `conda activate <env>`.
AUTO_CONDA_HOOKS="${AUTO_CONDA_HOOKS:-1}"
if [[ "${AUTO_CONDA_HOOKS}" == "1" && -n "${CONDA_PREFIX:-}" ]]; then
  HOOK_ACT_DIR="${CONDA_PREFIX}/etc/conda/activate.d"
  HOOK_DEACT_DIR="${CONDA_PREFIX}/etc/conda/deactivate.d"
  HOOK_ACT_FILE="${HOOK_ACT_DIR}/zz_photomaker_runtime.sh"
  HOOK_DEACT_FILE="${HOOK_DEACT_DIR}/zz_photomaker_runtime.sh"
  mkdir -p "${HOOK_ACT_DIR}" "${HOOK_DEACT_DIR}"

  cat > "${HOOK_ACT_FILE}" <<EOF
#!/usr/bin/env bash
if [[ "\${PM_RUNTIME_HOOK_APPLIED:-0}" == "1" ]]; then
  return 0
fi
export _PM_OLD_LD_LIBRARY_PATH="\${LD_LIBRARY_PATH-__PM_UNSET__}"
export _PM_OLD_LD_PRELOAD="\${LD_PRELOAD-__PM_UNSET__}"
export _PM_OLD_PHOTOMAKER_PATH="\${PHOTOMAKER_PATH-__PM_UNSET__}"
export _PM_OLD_PM_PATH="\${PM_PATH-__PM_UNSET__}"
if [[ -f "${RUNTIME_EXPORT_FILE}" ]]; then
  source "${RUNTIME_EXPORT_FILE}"
fi
if [[ -f "${PHOTOMAKER_EXPORT_FILE}" ]]; then
  source "${PHOTOMAKER_EXPORT_FILE}"
fi
export PM_RUNTIME_HOOK_APPLIED=1
EOF
  chmod +x "${HOOK_ACT_FILE}"

  cat > "${HOOK_DEACT_FILE}" <<'EOF'
#!/usr/bin/env bash
if [[ "${PM_RUNTIME_HOOK_APPLIED:-0}" != "1" ]]; then
  return 0
fi

if [[ "${_PM_OLD_LD_LIBRARY_PATH-__PM_UNSET__}" == "__PM_UNSET__" ]]; then
  unset LD_LIBRARY_PATH
else
  export LD_LIBRARY_PATH="${_PM_OLD_LD_LIBRARY_PATH}"
fi

if [[ "${_PM_OLD_LD_PRELOAD-__PM_UNSET__}" == "__PM_UNSET__" ]]; then
  unset LD_PRELOAD
else
  export LD_PRELOAD="${_PM_OLD_LD_PRELOAD}"
fi

if [[ "${_PM_OLD_PHOTOMAKER_PATH-__PM_UNSET__}" == "__PM_UNSET__" ]]; then
  unset PHOTOMAKER_PATH
else
  export PHOTOMAKER_PATH="${_PM_OLD_PHOTOMAKER_PATH}"
fi

if [[ "${_PM_OLD_PM_PATH-__PM_UNSET__}" == "__PM_UNSET__" ]]; then
  unset PM_PATH
else
  export PM_PATH="${_PM_OLD_PM_PATH}"
fi

unset _PM_OLD_LD_LIBRARY_PATH _PM_OLD_LD_PRELOAD _PM_OLD_PHOTOMAKER_PATH _PM_OLD_PM_PATH PM_RUNTIME_HOOK_APPLIED
EOF
  chmod +x "${HOOK_DEACT_FILE}"
  echo "[hook] Installed conda hooks:"
  echo "       ${HOOK_ACT_FILE}"
  echo "       ${HOOK_DEACT_FILE}"
fi

# Hard guard: ensure critical torch stack versions still match the snapshot.
ENFORCE_TORCH_GUARD="${ENFORCE_TORCH_GUARD:-1}"
if [[ "${ENFORCE_TORCH_GUARD}" == "1" && -f "${PIP_FILE}" ]]; then
  echo "[guard] Verifying torch stack versions"
  TORCH_GUARD_FILE="$(mktemp)"
  awk '/^(torch|torchvision|torchaudio|triton)==/ {print}' "${PIP_FILE}" | sort -u > "${TORCH_GUARD_FILE}"
  if [[ -s "${TORCH_GUARD_FILE}" ]]; then
    TORCH_GUARD_FILE="${TORCH_GUARD_FILE}" "${PY_BIN}" - <<'PY'
import os
import sys
from importlib import metadata

req_file = os.environ["TORCH_GUARD_FILE"]
required = {}
with open(req_file, "r", encoding="utf-8") as handle:
    for raw in handle:
        line = raw.strip()
        if not line or "==" not in line:
            continue
        name, expected = line.split("==", 1)
        required[name.strip().lower()] = expected.strip()

missing = []
mismatch = []
for package, expected in required.items():
    try:
        installed = metadata.version(package)
    except metadata.PackageNotFoundError:
        missing.append((package, expected))
        continue
    if installed != expected:
        mismatch.append((package, expected, installed))

if missing or mismatch:
    print("ERROR: Torch stack guard failed. Installed versions differ from snapshot.")
    for package, expected in missing:
        print(f"  MISSING  {package}=={expected}")
    for package, expected, installed in mismatch:
        print(f"  MISMATCH {package}: expected {expected}, got {installed}")
    print("Set ENFORCE_TORCH_GUARD=0 to bypass (not recommended).")
    raise SystemExit(1)

print("Torch stack guard OK.")
PY
  fi
  rm -f "${TORCH_GUARD_FILE}"
fi

echo "Done. Restored env: ${TARGET_ENV}"
echo "Verify:"
echo "  conda activate ${TARGET_ENV}"
if [[ "${AUTO_CONDA_HOOKS:-1}" == "1" ]]; then
  echo "  # runtime/PhotoMaker exports are auto-applied by conda activate hooks"
else
  if [[ "${SKIP_RUNTIME_OVERLAY}" != "1" ]]; then
    echo "  source ${RUNTIME_EXPORT_FILE}"
  fi
  if [[ "${SKIP_PHOTOMAKER_PREFETCH}" != "1" ]]; then
    echo "  source ${PHOTOMAKER_EXPORT_FILE}"
  fi
fi
if [[ "${SKIP_INSIGHTFACE_PREFETCH}" == "1" ]]; then
  echo "  # optional: export INSIGHTFACE_HOME and pre-download buffalo_l manually before training"
fi
echo "  python --version"
echo "  python -m pip freeze | head"
echo "  # then run your accelerate launch command in this same shell"
