#!/usr/bin/env bash
set -euo pipefail

# 13 Aug 2026 - AICODE-NOTE: Storage cleanup must fail closed before unlinking
# anything; only sealed regular .pth files below the Nasilaev owner root qualify.
OWNER_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev"

execute=0
expected_count=""
expected_bytes=""
expected_sha256=""
confirmed_sha256=""

usage() {
    cat >&2 <<'EOF'
Usage: guarded_delete_pth_list.sh [--execute] \
  --expected-count N --expected-bytes N --expected-sha256 HEX \
  [--confirm-delete-list-sha256 HEX]

Reads canonical deletion candidates from stdin:
  <size_in_bytes><TAB><absolute_path>

Without --execute, validates the complete list and performs no deletion.
Execution additionally requires --confirm-delete-list-sha256 to exactly match
--expected-sha256.
EOF
}

fail() {
    printf 'GUARD_ERROR: %s\n' "$*" >&2
    exit 64
}

while (($#)); do
    case "$1" in
        --execute)
            execute=1
            shift
            ;;
        --expected-count)
            (($# >= 2)) || fail "missing value for --expected-count"
            expected_count="$2"
            shift 2
            ;;
        --expected-bytes)
            (($# >= 2)) || fail "missing value for --expected-bytes"
            expected_bytes="$2"
            shift 2
            ;;
        --expected-sha256)
            (($# >= 2)) || fail "missing value for --expected-sha256"
            expected_sha256="$2"
            shift 2
            ;;
        --confirm-delete-list-sha256)
            (($# >= 2)) || fail "missing value for --confirm-delete-list-sha256"
            confirmed_sha256="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            fail "unknown argument: $1"
            ;;
    esac
done

[[ "$expected_count" =~ ^[1-9][0-9]*$ ]] || fail "expected count must be a positive integer"
[[ "$expected_bytes" =~ ^[1-9][0-9]*$ ]] || fail "expected bytes must be a positive integer"
[[ "$expected_sha256" =~ ^[0-9a-f]{64}$ ]] || fail "expected SHA-256 must be 64 lowercase hex characters"
[[ -d "$OWNER_ROOT" ]] || fail "owner root is unavailable: $OWNER_ROOT"
owner_real="$(readlink -f -- "$OWNER_ROOT")" || fail "cannot resolve owner root"
[[ -n "$owner_real" ]] || fail "resolved owner root is empty"

declare -a candidate_sizes=()
declare -a candidate_paths=()
declare -a canonical_lines=()
declare -A seen_paths=()
count=0
total_bytes=0

validate_target() {
    local expected_size="$1"
    local candidate="$2"
    local ordinal="$3"
    local resolved actual_size

    [[ "$expected_size" =~ ^[1-9][0-9]*$ ]] || fail "entry $ordinal has invalid size: $expected_size"
    [[ -n "$candidate" ]] || fail "entry $ordinal has an empty path"
    [[ "$candidate" == "$OWNER_ROOT/"* ]] || fail "entry $ordinal is outside owner root: $candidate"
    [[ "$candidate" == *.pth ]] || fail "entry $ordinal is not a .pth file: $candidate"
    [[ ! -L "$candidate" ]] || fail "entry $ordinal is a symlink: $candidate"
    [[ -f "$candidate" ]] || fail "entry $ordinal is not a regular existing file: $candidate"

    resolved="$(readlink -f -- "$candidate")" || fail "entry $ordinal cannot be resolved: $candidate"
    [[ "$resolved" == "$owner_real/"* ]] || fail "entry $ordinal resolves outside owner root: $candidate -> $resolved"
    [[ "$resolved" == *.pth ]] || fail "entry $ordinal resolves to a non-.pth target: $candidate -> $resolved"

    actual_size="$(stat -c '%s' -- "$candidate")" || fail "entry $ordinal cannot be stat'ed: $candidate"
    [[ "$actual_size" == "$expected_size" ]] || fail "entry $ordinal size changed: expected $expected_size, found $actual_size: $candidate"
}

while IFS=$'\t' read -r size path extra; do
    ((count += 1))
    [[ -z "${extra:-}" ]] || fail "entry $count has more than two tab-separated fields"
    validate_target "$size" "$path" "$count"
    [[ -z "${seen_paths[$path]:-}" ]] || fail "entry $count duplicates a path: $path"
    seen_paths["$path"]=1
    candidate_sizes+=("$size")
    candidate_paths+=("$path")
    canonical_lines+=("${size}"$'\t'"${path}")
    ((total_bytes += size))
done

[[ "$count" == "$expected_count" ]] || fail "candidate count mismatch: expected $expected_count, found $count"
[[ "$total_bytes" == "$expected_bytes" ]] || fail "candidate byte total mismatch: expected $expected_bytes, found $total_bytes"

actual_sha256="$(printf '%s\n' "${canonical_lines[@]}" | sha256sum | awk '{print $1}')"
[[ "$actual_sha256" == "$expected_sha256" ]] || fail "candidate SHA-256 mismatch: expected $expected_sha256, found $actual_sha256"

printf 'GUARD_VALIDATED count=%s bytes=%s sha256=%s root=%s\n' \
    "$count" "$total_bytes" "$actual_sha256" "$owner_real"

if ((execute == 0)); then
    printf 'GUARD_DRY_RUN_OK no files deleted\n'
    exit 0
fi

[[ "$confirmed_sha256" == "$expected_sha256" ]] || fail "execution confirmation SHA-256 is absent or does not match"

# Revalidate the complete sealed set once more before the first unlink.
for ((i = 0; i < count; i++)); do
    validate_target "${candidate_sizes[$i]}" "${candidate_paths[$i]}" "$((i + 1))"
done

deleted_count=0
deleted_bytes=0
for ((i = 0; i < count; i++)); do
    size="${candidate_sizes[$i]}"
    path="${candidate_paths[$i]}"
    validate_target "$size" "$path" "$((i + 1))"
    rm -- "$path"
    [[ ! -e "$path" && ! -L "$path" ]] || fail "entry $((i + 1)) still exists after rm: $path"
    ((deleted_count += 1))
    ((deleted_bytes += size))
    if ((deleted_count % 100 == 0)); then
        printf 'GUARD_PROGRESS deleted=%s/%s bytes=%s\n' "$deleted_count" "$count" "$deleted_bytes"
    fi
done

[[ "$deleted_count" == "$expected_count" ]] || fail "post-delete count mismatch"
[[ "$deleted_bytes" == "$expected_bytes" ]] || fail "post-delete byte total mismatch"
printf 'GUARD_DELETE_OK count=%s bytes=%s sha256=%s\n' \
    "$deleted_count" "$deleted_bytes" "$actual_sha256"
