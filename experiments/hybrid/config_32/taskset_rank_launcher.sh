#!/bin/bash
set -euo pipefail

RANK="${OMPI_COMM_WORLD_RANK:?Missing MPI rank}"
THREADS="${OMP_NUM_THREADS:?Missing OMP_NUM_THREADS}"

mapfile -t CPUS < <(
  python3 - <<'PY'
import os
for cpu in sorted(os.sched_getaffinity(0)):
    print(cpu)
PY
)

if [[ "${#CPUS[@]}" -ne 32 ]]; then
  echo "ERROR: Expected 32 allowed CPUs, found ${#CPUS[@]}" >&2
  exit 1
fi

START=$((RANK * THREADS))
CPU_LIST=$(IFS=,; echo "${CPUS[*]:START:THREADS}")
if [[ -z "$CPU_LIST" ]]; then
  echo "ERROR: Empty CPU assignment for Rank=$RANK" >&2
  exit 1
fi

echo "Rank=$RANK Threads=$THREADS AssignedCPUs=$CPU_LIST" >&2

exec taskset -c "$CPU_LIST" "$@"
