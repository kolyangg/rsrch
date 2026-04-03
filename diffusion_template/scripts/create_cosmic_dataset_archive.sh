#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${1:-/home/kolyangg/rsrch/dataset_full}"
OUTPUT_PATH="${2:-/home/kolyangg/rsrch/diffusion_template/cosmic_dataset_images.tar}"

require_path() {
    local path="$1"
    if [[ ! -e "$path" ]]; then
        echo "Missing required path: $path" >&2
        exit 1
    fi
}

require_path "$DATASET_ROOT"
require_path "$DATASET_ROOT/LAION-5B"
require_path "$DATASET_ROOT/LAION-5B-Filtered/laion1B-nolang"
require_path "$DATASET_ROOT/LAION-5B-Filtered-Large/laion1B-nolang"

mkdir -p "$(dirname "$OUTPUT_PATH")"

DATASET_PARENT="$(dirname "$DATASET_ROOT")"
DATASET_BASENAME="$(basename "$DATASET_ROOT")"

tar -cf "$OUTPUT_PATH" \
    -C "$DATASET_PARENT" \
    "$DATASET_BASENAME/LAION-5B" \
    "$DATASET_BASENAME/LAION-5B-Filtered/laion1B-nolang" \
    "$DATASET_BASENAME/LAION-5B-Filtered-Large/laion1B-nolang"

echo "Created archive: $OUTPUT_PATH"
