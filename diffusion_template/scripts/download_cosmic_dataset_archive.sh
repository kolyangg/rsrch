#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  download_cosmic_dataset_archive.sh GOOGLE_DRIVE_FILE_ID [extract_parent]

Arguments:
  GOOGLE_DRIVE_FILE_ID   Google Drive file ID for cosmic_dataset_images.tar
  extract_parent         Optional directory where dataset_full/ should be created.
                         Defaults to the parent of the repo root.

This script:
  1. downloads the archive with gdown
  2. extracts dataset_full/... into extract_parent
  3. removes the downloaded archive
EOF
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage >&2
    exit 1
fi

if ! command -v gdown >/dev/null 2>&1; then
    echo "gdown is required but not installed. Install it with: pip install gdown" >&2
    exit 1
fi

FILE_ID="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EXTRACT_PARENT="${2:-$(cd "$REPO_ROOT/.." && pwd)}"
ARCHIVE_PATH="$EXTRACT_PARENT/cosmic_dataset_images.tar"
DOWNLOAD_URL="https://drive.google.com/uc?id=${FILE_ID}"

mkdir -p "$EXTRACT_PARENT"

echo "Downloading archive to: $ARCHIVE_PATH"
gdown "$DOWNLOAD_URL" -O "$ARCHIVE_PATH"

echo "Extracting archive into: $EXTRACT_PARENT"
tar -xf "$ARCHIVE_PATH" -C "$EXTRACT_PARENT"

echo "Removing archive: $ARCHIVE_PATH"
rm -f "$ARCHIVE_PATH"

echo "Done."
echo "Expected dataset root: $EXTRACT_PARENT/dataset_full"
