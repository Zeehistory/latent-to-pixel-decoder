#!/bin/bash
set -euo pipefail
REPO_ROOT="${BASE_DIR:-${SLURM_SUBMIT_DIR:-}}"
source "$REPO_ROOT/scripts/restitution/_restitution_env.sh"
source "$REPO_ROOT/scripts/ablation/_activate_env.sh"
cd "$BASE_DIR"
mkdir -p logs
