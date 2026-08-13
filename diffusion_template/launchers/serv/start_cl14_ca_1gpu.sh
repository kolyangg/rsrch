#!/usr/bin/env bash
# 13 Aug 2026 - Dedicated Serv entry point fixes the CL14_CA config identity;
# environment activation and all scientific gates remain in the shared script.
set -euo pipefail

export CONFIG_NAME=CL14_CA_cosmic_residual_identity_ca_24k
exec "$(dirname "${BASH_SOURCE[0]}")/start_e13_family_1gpu.sh"
