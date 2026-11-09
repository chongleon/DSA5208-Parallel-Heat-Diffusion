
#!/bin/bash
set -euo pipefail

PROJECT="personal-leyaozhang-nus"
MODULE="OpenMPI/4.1.4-GCC-11.3.0"

N=4096
STEPS=1000
REPEATS=3
TOTAL_CORES=32

WALLTIME="01:00:00"
MEM="32gb"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$SCRIPT_DIR"

MODE="${1:---dry-run}"

if [[ "$MODE" != "--dry-run" && "$MODE" != "--submit" ]]; then
  echo "Usage: bash submit_hybrid32.sh [--dry-run|--submit]"
  exit 1
fi

if [[ ! -f "$PROJECT_DIR/hybrid.cpp" ]]; then
  echo "ERROR: hybrid.cpp not found at $PROJECT_DIR"
  exit 1
fi

RANKS=(1 2 4 8 16 32)

mkdir -p jobs bin results logs

if [[ "$MODE" == "--submit" ]]; then
  if compgen -G "results/hybrid_P*.csv" > /dev/null; then
    echo "ERROR: Existing Hybrid results found."
    exit 1
  fi

  if ! mkdir "$SCRIPT_DIR/.hybrid32_submission_lock" 2>/dev/null; then
    echo "ERROR: Hybrid submission already started."
    exit 1
  fi
fi

echo "=== Hybrid 32-Core Experiment ==="
echo "Project: $PROJECT_DIR"
echo "Experiment: $SCRIPT_DIR"
echo "N=$N Steps=$STEPS Repeats=$REPEATS"
echo "Total CPU cores=$TOTAL_CORES"
echo "Mode=$MODE"

# Generate a launcher that dynamically assigns CPU cores
# to each MPI rank using the CPU affinity allowed by PBS.

cat > "$SCRIPT_DIR/taskset_rank_launcher.sh" <<'LAUNCHER'
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
LAUNCHER

chmod +x "$SCRIPT_DIR/taskset_rank_launcher.sh"

# Generate PBS scripts for each configuration
for P in "${RANKS[@]}"; do

  THREADS=$((TOTAL_CORES / P))

  JOB="$SCRIPT_DIR/jobs/hybrid_P${P}.pbs"

  cat > "$JOB" <<JOBEOF
#!/bin/bash
#PBS -P ${PROJECT}
#PBS -N hybrid32_P${P}
#PBS -l select=1:ncpus=32:mpiprocs=${P}:ompthreads=${THREADS}:mem=${MEM}
#PBS -l walltime=${WALLTIME}
#PBS -j oe
#PBS -o ${SCRIPT_DIR}/logs/

set -euo pipefail

cd "${SCRIPT_DIR}" || exit 1

module load ${MODULE} || exit 1

export OMP_NUM_THREADS=${THREADS}
export OMP_DYNAMIC=FALSE
export OMP_PLACES=cores
export OMP_PROC_BIND=close

echo "=== Environment ==="
echo "JobID=\$PBS_JOBID"
echo "Node=\$(hostname)"
echo "Ranks=${P}"
echo "Threads=${THREADS}"
echo "TotalCores=${TOTAL_CORES}"
echo "N=${N}"
echo "Steps=${STEPS}"
echo "MPI_Module=${MODULE}"
echo "CompilerFlags=-O3 -fopenmp -std=c++17"

echo "Allocated nodes:"
sort "\$PBS_NODEFILE" | uniq -c

echo "CPU affinity:"
taskset -pc \$\$

echo "=== Compile Hybrid ==="

mpicxx -O3 -fopenmp -std=c++17 \
  "${PROJECT_DIR}/hybrid.cpp" \
  -o "${SCRIPT_DIR}/bin/hybrid_P${P}"

echo "=== Hybrid P=${P} T=${THREADS} ==="

for REP in \$(seq 1 ${REPEATS}); do

  echo "Starting Repeat=\$REP at \$(date -Is)"

  OUT=\$(timeout 300s mpirun -np ${P} --bind-to none \\
    "${SCRIPT_DIR}/taskset_rank_launcher.sh" \\
    "${SCRIPT_DIR}/bin/hybrid_P${P}" ${N} ${STEPS}) || exit 1

  printf '%s\n' "\$OUT" > "${SCRIPT_DIR}/results/hybrid_P${P}_run\${REP}.txt"

  RUNTIME=\$(printf '%s\n' "\$OUT" | awk -F= '/^Runtime=/{print \$2}')
  COMM=\$(printf '%s\n' "\$OUT" | awk -F= '/^CommunicationTime=/{print \$2}')
  COMP=\$(printf '%s\n' "\$OUT" | awk -F= '/^ComputationTime=/{print \$2}')

  if [[ -z "\$RUNTIME" || -z "\$COMM" || -z "\$COMP" ]]; then
    echo "ERROR: Missing timing fields P=${P} Repeat=\$REP" >&2
    exit 1
  fi

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \\
    '${P}' '${THREADS}' '${TOTAL_CORES}' '${N}' '${STEPS}' \\
    "\$REP" "\$RUNTIME" "\$COMM" "\$COMP" \\
    >> "${SCRIPT_DIR}/results/hybrid_P${P}.csv"

  echo "Completed P=${P} Repeat=\$REP Runtime=\$RUNTIME"

done

echo "=== Finished ==="
JOBEOF

  bash -n "$JOB"

  echo "P=$P Threads=$THREADS TotalCores=$TOTAL_CORES"

done

if [[ "$MODE" == "--dry-run" ]]; then
  echo "=== DRY RUN COMPLETE ==="
  echo "No PBS jobs submitted."
  exit 0
fi

# Initialize CSV files
for P in "${RANKS[@]}"; do
  printf 'Ranks,Threads,TotalCores,N,Steps,Repeat,Runtime,CommunicationTime,ComputationTime\n' \
    > "$SCRIPT_DIR/results/hybrid_P${P}.csv"
done

printf 'Ranks,Threads,JobID,JobScript\n' > "$SCRIPT_DIR/submitted_jobs.csv"

for P in "${RANKS[@]}"; do

  THREADS=$((TOTAL_CORES / P))
  JOB="$SCRIPT_DIR/jobs/hybrid_P${P}.pbs"

  JOB_ID=$(qsub "$JOB") || {
    echo "ERROR: Submission failed at P=$P."
    echo "Check submitted_jobs.csv before retrying."
    exit 1
  }

  printf '%s,%s,%s,%s\n' "$P" "$THREADS" "$JOB_ID" "$JOB" \
    >> "$SCRIPT_DIR/submitted_jobs.csv"

  echo "Submitted P=$P Threads=$THREADS JobID=$JOB_ID"

done

echo "=== ALL 6 JOBS SUBMITTED ==="
echo "Results: $SCRIPT_DIR/results/"
echo "Logs: $SCRIPT_DIR/logs/"
echo "Job IDs: $SCRIPT_DIR/submitted_jobs.csv"
