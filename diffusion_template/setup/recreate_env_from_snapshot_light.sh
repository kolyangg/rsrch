#!/usr/bin/env bash
set -euo pipefail

DEFAULT_ENV_PREFIX="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/conda_env/photomaker_NS"

usage() {
  cat <<EOF
Usage: $0 <snapshot_dir> [--prefix <env_prefix>]

Create a lighter, prefix-based restore of a saved conda snapshot.

Defaults:
  --prefix ${DEFAULT_ENV_PREFIX}

Differences vs recreate_env_from_snapshot.sh:
  - restores into a dedicated conda prefix instead of a named env
  - uses the portable conda spec by default
  - does not force CUDA-specific ORT pins, GCC runtime overlays, or torch nightly upgrades
  - keeps PhotoMaker/InsightFace/HF/CLIP caches under the created env prefix

Optional env toggles:
  SKIP_INSIGHTFACE_PREFETCH=1
  SKIP_PHOTOMAKER_PREFETCH=1
  SKIP_REALVIS_PREFETCH=1
  SKIP_CLIP_PREFETCH=1
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

ENV_PREFIX="${DEFAULT_ENV_PREFIX}"
POSITIONAL=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--prefix)
      [[ $# -ge 2 ]] || die "--prefix requires a value"
      ENV_PREFIX="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        POSITIONAL+=("$1")
        shift
      done
      ;;
    -*)
      die "Unknown option: $1"
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

[[ ${#POSITIONAL[@]} -eq 1 ]] || {
  usage >&2
  exit 1
}

if ! command -v conda >/dev/null 2>&1; then
  die "conda is not available in PATH."
fi

SNAPSHOT_DIR_INPUT="${POSITIONAL[0]}"
[[ -d "${SNAPSHOT_DIR_INPUT}" ]] || die "snapshot_dir does not exist: ${SNAPSHOT_DIR_INPUT}"
SNAPSHOT_DIR="$(cd "${SNAPSHOT_DIR_INPUT}" && pwd -P)"

ENV_PREFIX_PARENT="$(dirname "${ENV_PREFIX}")"
mkdir -p "${ENV_PREFIX_PARENT}"
ENV_PREFIX_PARENT="$(cd "${ENV_PREFIX_PARENT}" && pwd -P)"
ENV_PREFIX="${ENV_PREFIX_PARENT}/$(basename "${ENV_PREFIX}")"
TARGET_ENV_LABEL="$(basename "${ENV_PREFIX}")"

EXPLICIT_FILE="${SNAPSHOT_DIR}/conda_explicit.txt"
NOBUILDS_FILE="${SNAPSHOT_DIR}/environment_nobuilds.yml"
PIP_FILE="${SNAPSHOT_DIR}/pip_freeze.txt"

[[ -f "${NOBUILDS_FILE}" ]] || die "portable conda spec is missing: ${NOBUILDS_FILE}"

if [[ -e "${ENV_PREFIX}" ]]; then
  die "target prefix already exists: ${ENV_PREFIX}"
fi

echo "Restoring light environment:"
echo "  snapshot:      ${SNAPSHOT_DIR}"
echo "  target_prefix: ${ENV_PREFIX}"
echo "  env_label:     ${TARGET_ENV_LABEL}"

# Keep conda operations isolated from active runtime overlays.
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

TMP_FILES=()
CONDA_TEMP_PKGS_DIR=""
cleanup() {
  local f=""
  for f in "${TMP_FILES[@]}"; do
    [[ -n "${f}" ]] && rm -f "${f}"
  done
  if [[ -n "${CONDA_TEMP_PKGS_DIR}" && -d "${CONDA_TEMP_PKGS_DIR}" ]]; then
    rm -rf "${CONDA_TEMP_PKGS_DIR}"
  fi
}
trap cleanup EXIT

TMP_ENV_FILE="$(mktemp)"
TMP_FILES+=("${TMP_ENV_FILE}")
awk '
  /^[[:space:]]*name:[[:space:]]*/ { next }
  /^[[:space:]]*prefix:[[:space:]]*/ { next }
  { print }
' "${NOBUILDS_FILE}" > "${TMP_ENV_FILE}"

CONDA_TEMP_PKGS_DIR="$(mktemp -d)"
export CONDA_PKGS_DIRS="${CONDA_TEMP_PKGS_DIR}"

echo "[conda] Creating env from ${NOBUILDS_FILE}"
run_conda env create --prefix "${ENV_PREFIX}" -f "${TMP_ENV_FILE}"

unset LD_PRELOAD
if [[ -n "${CONDA_CLEAN_LD_LIBRARY_PATH}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_CLEAN_LD_LIBRARY_PATH}"
else
  unset LD_LIBRARY_PATH
fi

eval "$(run_conda shell.bash hook)"
conda activate "${ENV_PREFIX}"

PY_BIN="$(command -v python)"

ENV_STATE_DIR="${CONDA_PREFIX}/var/photomaker_ns"
ENV_CACHE_DIR="${ENV_STATE_DIR}/cache"
ENV_LOG_DIR="${ENV_STATE_DIR}/logs"
HF_HOME_DIR="${ENV_CACHE_DIR}/huggingface"
HF_HUB_CACHE_DIR="${HF_HOME_DIR}/hub"
TRANSFORMERS_CACHE_DIR="${ENV_CACHE_DIR}/transformers"
HF_DATASETS_CACHE_DIR="${ENV_CACHE_DIR}/datasets"
XDG_CACHE_HOME_DIR="${ENV_CACHE_DIR}/xdg"
CLIP_CACHE_DIR_LOCAL="${ENV_CACHE_DIR}/clip"
INSIGHTFACE_HOME_DIR="${ENV_STATE_DIR}/insightface"
PIP_CACHE_DIR_LOCAL="${ENV_CACHE_DIR}/pip"
UV_CACHE_DIR_LOCAL="${ENV_CACHE_DIR}/uv"
mkdir -p \
  "${ENV_LOG_DIR}" \
  "${HF_HUB_CACHE_DIR}" \
  "${TRANSFORMERS_CACHE_DIR}" \
  "${HF_DATASETS_CACHE_DIR}" \
  "${XDG_CACHE_HOME_DIR}" \
  "${CLIP_CACHE_DIR_LOCAL}" \
  "${INSIGHTFACE_HOME_DIR}" \
  "${PIP_CACHE_DIR_LOCAL}" \
  "${UV_CACHE_DIR_LOCAL}"

export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_CACHE_DIR="${PIP_CACHE_DIR_LOCAL}"
export UV_CACHE_DIR="${UV_CACHE_DIR_LOCAL}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME_DIR}"
export HF_HOME="${HF_HOME_DIR}"
export HF_HUB_CACHE="${HF_HUB_CACHE_DIR}"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE_DIR}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE_DIR}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE_DIR}"
export CLIP_CACHE_DIR="${CLIP_CACHE_DIR_LOCAL}"
export INSIGHTFACE_HOME="${INSIGHTFACE_HOME_DIR}"
export INSIGHTFACE_ROOT="${INSIGHTFACE_HOME_DIR}"

normalize_pip_requirements() {
  local in_file="$1"
  local out_file="$2"

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
  ' "${in_file}" > "${out_file}"
}

if [[ -f "${PIP_FILE}" ]]; then
  echo "[pip] Reconciling pip packages from ${PIP_FILE}"
  TMP_PIP_FILE="$(mktemp)"
  TMP_FILES+=("${TMP_PIP_FILE}")
  TMP_PIP_DELTA_FILE="$(mktemp)"
  TMP_FILES+=("${TMP_PIP_DELTA_FILE}")
  FAILED_REQS_FILE="${ENV_LOG_DIR}/pip_failed_requirements.txt"
  : > "${FAILED_REQS_FILE}"

  normalize_pip_requirements "${PIP_FILE}" "${TMP_PIP_FILE}"

  "${PY_BIN}" - "${TMP_PIP_FILE}" "${TMP_PIP_DELTA_FILE}" <<'PY'
import sys
from importlib import metadata

from packaging.requirements import Requirement

in_file, out_file = sys.argv[1:3]

with open(in_file, "r", encoding="utf-8") as handle:
    raw_requirements = [line.strip() for line in handle if line.strip()]

to_install = []
for raw in raw_requirements:
    try:
        req = Requirement(raw)
    except Exception:
        to_install.append(raw)
        continue
    if req.marker and not req.marker.evaluate():
        continue

    try:
        installed_version = metadata.version(req.name)
    except metadata.PackageNotFoundError:
        to_install.append(raw)
        continue

    if req.specifier and installed_version not in req.specifier:
        to_install.append(raw)

with open(out_file, "w", encoding="utf-8") as handle:
    for item in to_install:
        handle.write(f"{item}\n")
PY

  if [[ -s "${TMP_PIP_DELTA_FILE}" ]]; then
    if ! command -v uv >/dev/null 2>&1; then
      echo "[pip] Installing uv inside ${CONDA_PREFIX}"
      "${PY_BIN}" -m pip install uv
    fi

    echo "[pip] Installing only missing/mismatched pip packages"
    set +e
    uv pip install --python "${PY_BIN}" -r "${TMP_PIP_DELTA_FILE}"
    UV_STATUS=$?
    set -e

    if [[ ${UV_STATUS} -ne 0 ]]; then
      echo "[pip] Bulk install failed. Retrying requirement-by-requirement."
      while IFS= read -r req; do
        [[ -z "${req}" ]] && continue
        set +e
        uv pip install --python "${PY_BIN}" "${req}"
        ONE_STATUS=$?
        set -e
        if [[ ${ONE_STATUS} -ne 0 ]]; then
          echo "${req}" >> "${FAILED_REQS_FILE}"
        fi
      done < "${TMP_PIP_DELTA_FILE}"
    fi

    if [[ -s "${FAILED_REQS_FILE}" ]]; then
      echo "WARNING: Some pip requirements could not be installed."
      echo "See: ${FAILED_REQS_FILE}"
    else
      rm -f "${FAILED_REQS_FILE}"
    fi
  else
    rm -f "${FAILED_REQS_FILE}"
    echo "[pip] Snapshot pip requirements are already satisfied after conda restore."
  fi
else
  echo "[pip] pip_freeze.txt not found; skipping additional pip restore."
fi

CLIP_SOURCE="${CLIP_SOURCE:-git+https://github.com/openai/CLIP.git}"
echo "[clip] Verifying OpenAI CLIP package"
if ! "${PY_BIN}" - <<'PY' >/dev/null 2>&1
import clip

required = ("load", "tokenize")
raise SystemExit(0 if all(hasattr(clip, name) for name in required) else 1)
PY
then
  if ! command -v uv >/dev/null 2>&1; then
    "${PY_BIN}" -m pip install uv
  fi
  CLIP_SETUPTOOLS_SPEC="${CLIP_SETUPTOOLS_SPEC:-setuptools<81}"
  uv pip install --python "${PY_BIN}" --upgrade "${CLIP_SETUPTOOLS_SPEC}" wheel

  set +e
  uv pip install --python "${PY_BIN}" --no-deps "${CLIP_SOURCE}"
  CLIP_INSTALL_STATUS=$?
  set -e

  if [[ ${CLIP_INSTALL_STATUS} -ne 0 ]]; then
    echo "[clip] uv install failed. Retrying with pip --no-build-isolation."
    "${PY_BIN}" -m pip install --upgrade "${CLIP_SETUPTOOLS_SPEC}" wheel
    "${PY_BIN}" -m pip install --no-deps --no-build-isolation "${CLIP_SOURCE}"
  fi
else
  echo "[clip] OpenAI CLIP API already present."
fi

PHOTOMAKER_REAL_PATH=""

SKIP_INSIGHTFACE_PREFETCH="${SKIP_INSIGHTFACE_PREFETCH:-0}"
if [[ "${SKIP_INSIGHTFACE_PREFETCH}" != "1" ]]; then
  echo "[insightface] Prefetching buffalo_l into ${INSIGHTFACE_HOME_DIR}"
  INSIGHTFACE_MODEL_NAME="${INSIGHTFACE_MODEL_NAME:-buffalo_l}"
  INSIGHTFACE_MODELS_DIR="${INSIGHTFACE_HOME_DIR}/models"
  INSIGHTFACE_MODEL_DIR="${INSIGHTFACE_MODELS_DIR}/${INSIGHTFACE_MODEL_NAME}"
  INSIGHTFACE_ZIP="${INSIGHTFACE_MODELS_DIR}/${INSIGHTFACE_MODEL_NAME}.zip"
  INSIGHTFACE_URL="${INSIGHTFACE_URL:-https://github.com/deepinsight/insightface/releases/download/v0.7/${INSIGHTFACE_MODEL_NAME}.zip}"

  mkdir -p "${INSIGHTFACE_MODELS_DIR}"
  if [[ -d "${INSIGHTFACE_MODEL_DIR}" ]]; then
    echo "[insightface] Reusing existing model dir: ${INSIGHTFACE_MODEL_DIR}"
  else
    if command -v wget >/dev/null 2>&1; then
      wget -c --tries=20 --timeout=30 -O "${INSIGHTFACE_ZIP}" "${INSIGHTFACE_URL}"
    elif command -v curl >/dev/null 2>&1; then
      curl -L --retry 20 --retry-all-errors --connect-timeout 30 -o "${INSIGHTFACE_ZIP}" "${INSIGHTFACE_URL}"
    else
      "${PY_BIN}" - "${INSIGHTFACE_URL}" "${INSIGHTFACE_ZIP}" <<'PY'
import pathlib
import sys
import urllib.request

url, out_path = sys.argv[1], sys.argv[2]
pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
urllib.request.urlretrieve(url, out_path)
PY
    fi

    "${PY_BIN}" - "${INSIGHTFACE_ZIP}" "${INSIGHTFACE_MODELS_DIR}" <<'PY'
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
  echo "[insightface] Skipping prefetch (SKIP_INSIGHTFACE_PREFETCH=1)."
fi

SKIP_PHOTOMAKER_PREFETCH="${SKIP_PHOTOMAKER_PREFETCH:-0}"
if [[ "${SKIP_PHOTOMAKER_PREFETCH}" != "1" ]]; then
  echo "[photomaker] Prefetching PhotoMaker-V2 into ${HF_HUB_CACHE_DIR}"
  PHOTOMAKER_REPO_ID="${PHOTOMAKER_REPO_ID:-TencentARC/PhotoMaker-V2}"
  PHOTOMAKER_FILENAME="${PHOTOMAKER_FILENAME:-photomaker-v2.bin}"
  PHOTOMAKER_EXPECTED_SNAPSHOT="${PHOTOMAKER_EXPECTED_SNAPSHOT:-f5a1e5155dc02166253fa7e29d13519f5ba22eac}"
  PHOTOMAKER_CACHE_ROOT="${PHOTOMAKER_CACHE_ROOT:-${HF_HUB_CACHE_DIR}}"

  PHOTOMAKER_REAL_PATH="$(PHOTOMAKER_REPO_ID="${PHOTOMAKER_REPO_ID}" PHOTOMAKER_FILENAME="${PHOTOMAKER_FILENAME}" "${PY_BIN}" - <<'PY'
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

  echo "[photomaker] Path: ${PHOTOMAKER_REAL_PATH}"
  echo "[photomaker] Compatibility symlink: ${PHOTOMAKER_EXPECTED_PATH}"
else
  echo "[photomaker] Skipping prefetch (SKIP_PHOTOMAKER_PREFETCH=1)."
fi

SKIP_REALVIS_PREFETCH="${SKIP_REALVIS_PREFETCH:-0}"
if [[ "${SKIP_REALVIS_PREFETCH}" != "1" ]]; then
  echo "[realvis] Prefetching RealVisXL fp16 into ${HF_HOME_DIR}"
  REALVIS_REPO_ID="${REALVIS_REPO_ID:-SG161222/RealVisXL_V4.0}"
  REALVIS_CACHE_ROOT="${REALVIS_CACHE_ROOT:-${HF_HOME_DIR}}"
  REALVIS_MODEL_DIR="${REALVIS_CACHE_ROOT}/hub/models--SG161222--RealVisXL_V4.0"
  REALVIS_FORCE_CLEAN="${REALVIS_FORCE_CLEAN:-0}"
  if [[ "${REALVIS_FORCE_CLEAN}" == "1" ]]; then
    rm -rf "${REALVIS_MODEL_DIR}"
  fi

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
  echo "[realvis] Skipping prefetch (SKIP_REALVIS_PREFETCH=1)."
fi

SKIP_CLIP_PREFETCH="${SKIP_CLIP_PREFETCH:-0}"
if [[ "${SKIP_CLIP_PREFETCH}" != "1" ]]; then
  echo "[clip] Prefetching CLIP weights into ${CLIP_CACHE_DIR_LOCAL}"
  CLIP_MODEL_NAME="${CLIP_MODEL_NAME:-ViT-L/14@336px}"
  CLIP_PREFETCH_RETRIES="${CLIP_PREFETCH_RETRIES:-10}"
  CLIP_MODEL_NAME="${CLIP_MODEL_NAME}" CLIP_CACHE_DIR="${CLIP_CACHE_DIR_LOCAL}" CLIP_PREFETCH_RETRIES="${CLIP_PREFETCH_RETRIES}" "${PY_BIN}" - <<'PY'
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
for _ in range(retries):
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
  echo "[clip] Skipping prefetch (SKIP_CLIP_PREFETCH=1)."
fi

EXPORT_FILE="${ENV_STATE_DIR}/activate_photomaker_light.sh"
HOOK_ACT_DIR="${CONDA_PREFIX}/etc/conda/activate.d"
HOOK_DEACT_DIR="${CONDA_PREFIX}/etc/conda/deactivate.d"
HOOK_ACT_FILE="${HOOK_ACT_DIR}/zz_photomaker_light.sh"
HOOK_DEACT_FILE="${HOOK_DEACT_DIR}/zz_photomaker_light.sh"
mkdir -p "${HOOK_ACT_DIR}" "${HOOK_DEACT_DIR}"

echo "[hooks] Writing env-local activation exports"
cat > "${EXPORT_FILE}" <<EOF
#!/usr/bin/env bash
export PM_ENV_ROOT="${CONDA_PREFIX}"
export PM_ENV_DATA_ROOT="${ENV_STATE_DIR}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME_DIR}"
export HF_HOME="${HF_HOME_DIR}"
export HF_HUB_CACHE="${HF_HUB_CACHE_DIR}"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE_DIR}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE_DIR}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE_DIR}"
export CLIP_CACHE_DIR="${CLIP_CACHE_DIR_LOCAL}"
export INSIGHTFACE_HOME="${INSIGHTFACE_HOME_DIR}"
export INSIGHTFACE_ROOT="${INSIGHTFACE_HOME_DIR}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR_LOCAL}"
export UV_CACHE_DIR="${UV_CACHE_DIR_LOCAL}"
if [[ -d "\${CONDA_PREFIX}/lib" ]]; then
  export LD_LIBRARY_PATH="\${CONDA_PREFIX}/lib\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"
fi
EOF

if [[ -n "${PHOTOMAKER_REAL_PATH}" ]]; then
  cat >> "${EXPORT_FILE}" <<EOF
if [[ -f "${PHOTOMAKER_REAL_PATH}" ]]; then
  export PHOTOMAKER_PATH="${PHOTOMAKER_REAL_PATH}"
  export PM_PATH="${PHOTOMAKER_REAL_PATH}"
fi
EOF
fi
chmod +x "${EXPORT_FILE}"

cat > "${HOOK_ACT_FILE}" <<EOF
#!/usr/bin/env bash
if [[ "\${PM_LIGHT_HOOK_APPLIED:-0}" == "1" ]]; then
  return 0
fi
export _PM_LIGHT_OLD_PM_ENV_ROOT="\${PM_ENV_ROOT-__PM_UNSET__}"
export _PM_LIGHT_OLD_PM_ENV_DATA_ROOT="\${PM_ENV_DATA_ROOT-__PM_UNSET__}"
export _PM_LIGHT_OLD_XDG_CACHE_HOME="\${XDG_CACHE_HOME-__PM_UNSET__}"
export _PM_LIGHT_OLD_HF_HOME="\${HF_HOME-__PM_UNSET__}"
export _PM_LIGHT_OLD_HF_HUB_CACHE="\${HF_HUB_CACHE-__PM_UNSET__}"
export _PM_LIGHT_OLD_HUGGINGFACE_HUB_CACHE="\${HUGGINGFACE_HUB_CACHE-__PM_UNSET__}"
export _PM_LIGHT_OLD_TRANSFORMERS_CACHE="\${TRANSFORMERS_CACHE-__PM_UNSET__}"
export _PM_LIGHT_OLD_HF_DATASETS_CACHE="\${HF_DATASETS_CACHE-__PM_UNSET__}"
export _PM_LIGHT_OLD_CLIP_CACHE_DIR="\${CLIP_CACHE_DIR-__PM_UNSET__}"
export _PM_LIGHT_OLD_INSIGHTFACE_HOME="\${INSIGHTFACE_HOME-__PM_UNSET__}"
export _PM_LIGHT_OLD_INSIGHTFACE_ROOT="\${INSIGHTFACE_ROOT-__PM_UNSET__}"
export _PM_LIGHT_OLD_PIP_CACHE_DIR="\${PIP_CACHE_DIR-__PM_UNSET__}"
export _PM_LIGHT_OLD_UV_CACHE_DIR="\${UV_CACHE_DIR-__PM_UNSET__}"
export _PM_LIGHT_OLD_LD_LIBRARY_PATH="\${LD_LIBRARY_PATH-__PM_UNSET__}"
export _PM_LIGHT_OLD_PHOTOMAKER_PATH="\${PHOTOMAKER_PATH-__PM_UNSET__}"
export _PM_LIGHT_OLD_PM_PATH="\${PM_PATH-__PM_UNSET__}"
source "${EXPORT_FILE}"
export PM_LIGHT_HOOK_APPLIED=1
EOF
chmod +x "${HOOK_ACT_FILE}"

cat > "${HOOK_DEACT_FILE}" <<'EOF'
#!/usr/bin/env bash
if [[ "${PM_LIGHT_HOOK_APPLIED:-0}" != "1" ]]; then
  return 0
fi

for name in \
  PM_ENV_ROOT \
  PM_ENV_DATA_ROOT \
  XDG_CACHE_HOME \
  HF_HOME \
  HF_HUB_CACHE \
  HUGGINGFACE_HUB_CACHE \
  TRANSFORMERS_CACHE \
  HF_DATASETS_CACHE \
  CLIP_CACHE_DIR \
  INSIGHTFACE_HOME \
  INSIGHTFACE_ROOT \
  PIP_CACHE_DIR \
  UV_CACHE_DIR \
  LD_LIBRARY_PATH \
  PHOTOMAKER_PATH \
  PM_PATH
do
  old_var="_PM_LIGHT_OLD_${name}"
  old_value="${!old_var-__PM_UNSET__}"
  if [[ "${old_value}" == "__PM_UNSET__" ]]; then
    unset "${name}"
  else
    export "${name}=${old_value}"
  fi
done

unset \
  _PM_LIGHT_OLD_PM_ENV_ROOT \
  _PM_LIGHT_OLD_PM_ENV_DATA_ROOT \
  _PM_LIGHT_OLD_XDG_CACHE_HOME \
  _PM_LIGHT_OLD_HF_HOME \
  _PM_LIGHT_OLD_HF_HUB_CACHE \
  _PM_LIGHT_OLD_HUGGINGFACE_HUB_CACHE \
  _PM_LIGHT_OLD_TRANSFORMERS_CACHE \
  _PM_LIGHT_OLD_HF_DATASETS_CACHE \
  _PM_LIGHT_OLD_CLIP_CACHE_DIR \
  _PM_LIGHT_OLD_INSIGHTFACE_HOME \
  _PM_LIGHT_OLD_INSIGHTFACE_ROOT \
  _PM_LIGHT_OLD_PIP_CACHE_DIR \
  _PM_LIGHT_OLD_UV_CACHE_DIR \
  _PM_LIGHT_OLD_LD_LIBRARY_PATH \
  _PM_LIGHT_OLD_PHOTOMAKER_PATH \
  _PM_LIGHT_OLD_PM_PATH \
  PM_LIGHT_HOOK_APPLIED
EOF
chmod +x "${HOOK_DEACT_FILE}"

echo
echo "Done. Restored light env at: ${CONDA_PREFIX}"
echo "Conda snapshot source used: ${NOBUILDS_FILE}"
if [[ -f "${EXPLICIT_FILE}" ]]; then
  echo "Exact explicit spec intentionally skipped in light mode: ${EXPLICIT_FILE}"
fi
echo "Activate with:"
echo "  conda activate ${CONDA_PREFIX}"
echo "Verification:"
echo "  python --version"
echo "  python -m pip freeze | head"
echo "  python - <<'PY'"
echo "import os"
echo "print('PM_PATH=', os.environ.get('PM_PATH'))"
echo "print('HF_HOME=', os.environ.get('HF_HOME'))"
echo "print('INSIGHTFACE_HOME=', os.environ.get('INSIGHTFACE_HOME'))"
echo "print('CLIP_CACHE_DIR=', os.environ.get('CLIP_CACHE_DIR'))"
echo "PY"
