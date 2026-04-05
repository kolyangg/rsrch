#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  download_cosmic_dataset_archive.sh [--dataset_full_adj] GOOGLE_DRIVE_FILE_ID_OR_URL [extract_parent]

Arguments:
  GOOGLE_DRIVE_FILE_ID_OR_URL
                         Google Drive file ID or sharing URL for the archive
  extract_parent         Optional directory where dataset_full/ should be created.
                         Defaults to the parent of the repo root.

Flags:
  --dataset_full_adj     Treat the archive as large_dataset_adj.tar containing
                         large_dataset_adj/... and extract it into
                         <extract_parent>/dataset_full/ so the final path is:
                         <extract_parent>/dataset_full/large_dataset_adj

This script:
  1. downloads the archive with gdown
  2. extracts the archive into the correct location for training
  3. removes the downloaded archive
EOF
}

DATASET_FULL_ADJ_MODE=0
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset_full_adj|--dataset-full-adj)
            DATASET_FULL_ADJ_MODE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ ${#POSITIONAL_ARGS[@]} -lt 1 || ${#POSITIONAL_ARGS[@]} -gt 2 ]]; then
    usage >&2
    exit 1
fi

set -- "${POSITIONAL_ARGS[@]}"

if ! command -v gdown >/dev/null 2>&1; then
    echo "gdown is required but not installed. Install it with: pip install gdown" >&2
    exit 1
fi

FILE_ID="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EXTRACT_PARENT="${2:-$(cd "$REPO_ROOT/.." && pwd)}"
if [[ "$DATASET_FULL_ADJ_MODE" -eq 1 ]]; then
    ARCHIVE_BASENAME="large_dataset_adj.tar"
else
    ARCHIVE_BASENAME="cosmic_dataset_images.tar"
fi
ARCHIVE_PATH="$EXTRACT_PARENT/$ARCHIVE_BASENAME"

DOWNLOAD_INPUT="$FILE_ID"
if [[ "$DOWNLOAD_INPUT" =~ ^https?:// ]]; then
    if [[ "$DOWNLOAD_INPUT" =~ /file/d/([^/?]+) ]]; then
        FILE_ID="${BASH_REMATCH[1]}"
        DOWNLOAD_URL="https://drive.google.com/uc?id=${FILE_ID}"
    elif [[ "$DOWNLOAD_INPUT" =~ [\?\&]id=([^&]+) ]]; then
        FILE_ID="${BASH_REMATCH[1]}"
        DOWNLOAD_URL="https://drive.google.com/uc?id=${FILE_ID}"
    else
        DOWNLOAD_URL="$DOWNLOAD_INPUT"
    fi
else
    DOWNLOAD_URL="https://drive.google.com/uc?id=${DOWNLOAD_INPUT}"
fi

mkdir -p "$EXTRACT_PARENT"

echo "Downloading archive to: $ARCHIVE_PATH"
gdown "$DOWNLOAD_URL" -O "$ARCHIVE_PATH"

if [[ "$DATASET_FULL_ADJ_MODE" -eq 1 ]]; then
    TARGET_DIR="$EXTRACT_PARENT/dataset_full"
    mkdir -p "$TARGET_DIR"
    echo "Extracting large_dataset_adj archive into: $TARGET_DIR"
    tar -xf "$ARCHIVE_PATH" -C "$TARGET_DIR"
else
    TARGET_DIR="$EXTRACT_PARENT"
    echo "Extracting archive into: $TARGET_DIR"
    tar -xf "$ARCHIVE_PATH" -C "$TARGET_DIR"
fi

echo "Removing archive: $ARCHIVE_PATH"
rm -f "$ARCHIVE_PATH"

echo "Done."
if [[ "$DATASET_FULL_ADJ_MODE" -eq 1 ]]; then
    echo "Expected dataset root: $EXTRACT_PARENT/dataset_full/large_dataset_adj"
else
    echo "Expected dataset root: $EXTRACT_PARENT/dataset_full"
fi
