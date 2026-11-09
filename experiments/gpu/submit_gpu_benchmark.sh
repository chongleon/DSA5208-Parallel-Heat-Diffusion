#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
command -v qsub >/dev/null || { echo 'qsub not found: run on Vanda login node'; exit 1; }
qsub gpu_benchmark.pbs
