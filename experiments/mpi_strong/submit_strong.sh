
#!/bin/bash
set -euo pipefail

PROJECT="personal-leyaozhang-nus"
MODULE="OpenMPI/4.1.4-GCC-11.3.0"

N=4096
STEPS=1000
REPEATS=3

CORES_PER_NODE=${CORES_PER_NODE:-32}
MEM_PER_NODE=${MEM_PER_NODE:-32gb}
WALLTIME=${WALLTIME:-01:00:00}

# Locate directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$SCRIPT_DIR"

# Usage:
# bash submit_strong.sh --dry-run
# bash submit_strong.sh --submit

MODE="${1:---dry-run}"

if [[ "$MODE" != "--dry-run" && "$MODE" != "--submit" ]]; then
  echo "Usage: bash submit_strong.sh [--dry-run|--submit]"
  exit 1
fi

# Check source file
if [[ ! -f "$PROJECT_DIR/mpi.cpp" ]]; then
  echo "ERROR: mpi.cpp not found at $PROJECT_DIR"
  exit 1
fi

# Create experiment directories
mkdir -p jobs bin results logs

# Prevent overwriting existing results
if compgen -G "results/strong_P*.csv" > /dev/null; then
  echo "ERROR: Existing Strong Scaling results found."
  echo "Do not overwrite existing experiment data."
  exit 1
fi

RANKS=(1 2 4 8 16 32 64 128 256)

echo "=== MPI Strong Scaling ==="
echo "Project: $PROJECT_DIR"
echo "Experiment: $SCRIPT_DIR"
echo "N=$N Steps=$STEPS Repeats=$REPEATS"
echo "Cores per node=$CORES_PER_NODE"
echo "Mode=$MODE"

# Generate PBS scripts
for P in "${RANKS[@]}"; do

  NODES=$(( (P + CORES_PER_NODE - 1) / CORES_PER_NODE ))
  RPN=$(( (P + NODES - 1) / NODES ))

  if [[ $((NODES * RPN)) -ne "$P" ]]; then
    echo "ERROR: Cannot evenly distribute P=$P"
    exit 1
  fi

  JOB="$SCRIPT_DIR/jobs/strong_${P}.pbs"

  cat > "$JOB" <<JOBEOF
#!/bin/bash
#PBS -P ${PROJECT}
#PBS -N mpi_s${P}
#PBS -l select=${NODES}:ncpus=${RPN}:mpiprocs=${RPN}:ompthreads=1:mem=${MEM_PER_NODE}
#PBS -l walltime=${WALLTIME}
#PBS -j oe
#PBS -o ${SCRIPT_DIR}/logs/

set -euo pipefail

cd "${SCRIPT_DIR}" || exit 1

module load ${MODULE} || exit 1
export OMP_NUM_THREADS=1

echo "=== Environment ==="
echo "JobID=\$PBS_JOBID"
echo "Host=\$(hostname)"
echo "MPI_Module=${MODULE}"
echo "CompilerFlags=-O3 -std=c++17"
echo "Allocated nodes:"
sort "\$PBS_NODEFILE" | uniq -c

echo "=== Compile MPI ==="
mpicxx -O3 -std=c++17 "${PROJECT_DIR}/mpi.cpp" -o "${SCRIPT_DIR}/bin/mpi_strong_P${P}" || exit 1

echo "=== Strong Scaling P=${P} ==="

for REP in \$(seq 1 ${REPEATS}); do

  echo "Starting P=${P} Repeat=\$REP at \$(date -Is)"

  OUT=\$(mpirun -np ${P} "${SCRIPT_DIR}/bin/mpi_strong_P${P}" ${N} ${STEPS}) || exit 1

  printf '%s\n' "\$OUT" > "${SCRIPT_DIR}/results/strong_P${P}_run\${REP}.txt"

  RUNTIME=\$(printf '%s\n' "\$OUT" | awk -F= '/^Runtime=/{print \$2}')
  COMM=\$(printf '%s\n' "\$OUT" | awk -F= '/^CommunicationTime=/{print \$2}')
  COMP=\$(printf '%s\n' "\$OUT" | awk -F= '/^ComputationTime=/{print \$2}')

  if [[ -z "\$RUNTIME" || -z "\$COMM" || -z "\$COMP" ]]; then
    echo "ERROR: Missing timing fields P=${P} Repeat=\$REP" >&2
    exit 1
  fi

  printf '%s,%s,%s,%s,%s,%s,%s\n' \
    '${P}' '${N}' '${STEPS}' \
    "\$REP" "\$RUNTIME" "\$COMM" "\$COMP" \
    >> "${SCRIPT_DIR}/results/strong_P${P}.csv"

  echo "Completed P=${P} Repeat=\$REP Runtime=\$RUNTIME"

done

echo "=== Finished ==="
JOBEOF

  bash -n "$JOB"

  echo "P=$P Nodes=$NODES RanksPerNode=$RPN"

done

if [[ "$MODE" == "--dry-run" ]]; then
  echo "=== DRY RUN COMPLETE ==="
  echo "No PBS jobs submitted."
  exit 0
fi

# Create a submission lock to prevent duplicate submissions
if ! mkdir "$SCRIPT_DIR/.strong_submission_lock" 2>/dev/null; then
  echo "ERROR: A submission has already been started."
  exit 1
fi

# Initialize result files before submitting jobs
for P in "${RANKS[@]}"; do
  printf 'Ranks,N,Steps,Repeat,Runtime,CommunicationTime,ComputationTime\n' \
    > "$SCRIPT_DIR/results/strong_P${P}.csv"
done

# Submit jobs and record their IDs
printf 'Ranks,JobID,JobScript\n' > "$SCRIPT_DIR/submitted_jobs.csv"

for P in "${RANKS[@]}"; do

  JOB="$SCRIPT_DIR/jobs/strong_${P}.pbs"

  JOB_ID=$(qsub "$JOB") || {
    echo "ERROR: Submission failed at P=$P."
    echo "Check submitted_jobs.csv before retrying."
    exit 1
  }

  printf '%s,%s,%s\n' "$P" "$JOB_ID" "$JOB" \
    >> "$SCRIPT_DIR/submitted_jobs.csv"

  echo "Submitted P=$P JobID=$JOB_ID"

done

echo "=== ALL 9 JOBS SUBMITTED ==="
echo "Results: $SCRIPT_DIR/results/"
echo "Logs: $SCRIPT_DIR/logs/"
echo "Job IDs: $SCRIPT_DIR/submitted_jobs.csv"
