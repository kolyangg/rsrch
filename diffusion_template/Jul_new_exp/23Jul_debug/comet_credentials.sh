#!/usr/bin/env bash
# Source the already-configured Comet credential without copying it into this
# experiment folder or printing it into logs.

if [[ -z "${COMET_API_KEY:-}" ]]; then
    CREDENTIAL_SOURCE="/home/niko/rsrch/diffusion_template/serv_new_runs/start_ba_nr_alt_vast_N3a.sh"
    if [[ ! -r "${CREDENTIAL_SOURCE}" ]]; then
        echo "Comet credential source is unavailable: ${CREDENTIAL_SOURCE}" >&2
        return 2 2>/dev/null || exit 2
    fi
    COMET_ASSIGNMENT="$(
        rg -m1 '^[[:space:]]*(export[[:space:]]+)?COMET_API_KEY=' \
            "${CREDENTIAL_SOURCE}"
    )"
    if [[ -z "${COMET_ASSIGNMENT}" ]]; then
        echo "No Comet credential assignment found in configured source." >&2
        return 2 2>/dev/null || exit 2
    fi
    eval "${COMET_ASSIGNMENT}"
    export COMET_API_KEY
fi

if [[ -z "${COMET_API_KEY:-}" ]]; then
    echo "COMET_API_KEY remains unavailable." >&2
    return 2 2>/dev/null || exit 2
fi
