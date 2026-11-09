# DSA5208 Project 3 — Parallel 2D Heat Diffusion

**National University of Singapore · DSA5208**  
**Authors:** Ye Tiantian (A0354470E), Zhang Leyao (A0356596J)

## Overview

This project implements a two-dimensional heat-diffusion simulation using a five-point finite-difference stencil in C++17. Four implementations are evaluated:

1. **Serial CPU** — sequential updates with double buffering.
2. **Pure MPI** — one-dimensional row decomposition with `MPI_Sendrecv` halo exchanges.
3. **Hybrid MPI + OpenMP** — MPI decomposition and halo exchanges with OpenMP parallel updates within each rank.
4. **MPI + OpenMP GPU offloading** — MPI communication on the host and OpenMP `target teams distribute parallel for` updates on an NVIDIA GPU.

The grid is `N × N`, with a central hot region initially at 100, zero-valued outer boundaries, and update coefficient `lambda = 0.2`. Each executable accepts `N` and `Steps`, optionally followed by `--validate` to write its final temperature grid as CSV.

## Source files and layout

```text
DSA5208_project3/
├── README.md
├── serial.cpp
├── mpi.cpp
├── hybrid.cpp
├── validation/                     # Serial and MPI reference grids (local archive)
└── experiments/
    ├── serial/                     # serial_scaling.pbs and measurements
    ├── mpi_strong/                 # submit_strong.sh, jobs/, results/, logs/
    ├── mpi_weak/                   # submit_weak.sh, jobs/, results/, logs/
    ├── hybrid/
    │   ├── validation/            # Hybrid correctness evidence
    │   └── config_32/             # submit_hybrid32.sh, taskset launcher, results/
    │       └── rerun/             # Supplementary P=2,4,8 experiments
    └── gpu/
        ├── hybrid_gpu.cpp
        ├── build_gpu.sh
        ├── gpu_test.pbs
        ├── gpu_benchmark.pbs
        ├── submit_gpu_benchmark.sh
        ├── validate_gpu.py
        ├── collect_gpu_results.py
        └── summarize_gpu_results.py
```

This archive includes measured results, PBS logs, and some historical diagnostic files.The maintained GPU scripts are directly under `experiments/gpu/`.

## Environment

Experiments were run on the **NUS Vanda** HPC cluster using PBS. Commands below are intended for a Vanda Linux environment (not Windows PowerShell), unless otherwise noted. Run compute-intensive experiments through PBS rather than on the login node.

- Serial: GCC / `g++`, C++17, `-O3`.
- Pure MPI: `OpenMPI/4.1.4-GCC-11.3.0`, `mpicxx -O3 -std=c++17`.
- Hybrid CPU: same OpenMPI module, `mpicxx -O3 -fopenmp -std=c++17`.
- GPU: `nvhpc-hpcx-cuda12/26.1`, NVHPC `nvc++`, `-O3 -mp=gpu -gpu=cc86`, NVIDIA A40 GPU.
- Python 3 is used for correctness comparisons and GPU result aggregation.

PBS scripts include the project account used for the original runs (`personal-leyaozhang-nus`). **Change `#PBS -P` to your own authorized PBS project before submitting on another account.** Module names and available resources may also differ on other systems.

## Build and run the programs

From the project root, for a serial run:

```bash
g++ -O3 -std=c++17 serial.cpp -o serial
./serial 512 1000
./serial 64 100 --validate
```

For Pure MPI:

```bash
module load OpenMPI/4.1.4-GCC-11.3.0
mpicxx -O3 -std=c++17 mpi.cpp -o mpi
mpirun -np 4 ./mpi 4096 1000
mpirun -np 4 ./mpi 64 100 --validate
```

For Hybrid MPI + OpenMP (illustrative small run; request sufficient CPUs in PBS):

```bash
module load OpenMPI/4.1.4-GCC-11.3.0
mpicxx -O3 -fopenmp -std=c++17 hybrid.cpp -o hybrid
export OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE
mpirun -np 4 ./hybrid 64 100 --validate
```

The measured 32-core Hybrid configuration experiments used PBS allocations and a `taskset` launcher to assign non-overlapping permitted CPU sets to MPI ranks. For reproducing those measurements, use the PBS scripts described below rather than this illustrative command.

For GPU offloading, use a PBS GPU allocation. In `experiments/gpu/`:

```bash
module load nvhpc-hpcx-cuda12/26.1
bash build_gpu.sh
export OMP_TARGET_OFFLOAD=MANDATORY
export OMP_NUM_THREADS=1
mpirun -np 1 ./hybrid_gpu 64 100 --validate
```

`build_gpu.sh` creates `hybrid_gpu` in the GPU directory. `OMP_TARGET_OFFLOAD=MANDATORY` prevents silently falling back to host execution when GPU offloading is unavailable. A compatible GPU and CUDA/NVHPC runtime are required.

## Correctness validation

All variants use the same initial state, update formula, boundary conditions, and number of time steps. For a simple reference comparison, run serial and MPI with `N=64`, `Steps=100`, and `--validate` in separate working directories (or move the CSV outputs before running the next variant). Their outputs are `serial_result.csv` and `mpi_result.csv`.

The Hybrid CPU implementation writes `hybrid_result.csv`. The GPU implementation writes `hybrid_gpu_result.csv`. Compare final grids elementwise with an absolute tolerance of `1e-10`. The GPU package includes `validate_gpu.py`:

From `experiments/gpu/`, after producing the two matching grids:

```bash
python3 validate_gpu.py --reference mpi_result.csv --candidate hybrid_gpu_result.csv --atol 1e-10
```

The GPU validation PBS script expects the reference `mpi_result.csv` **in `experiments/gpu/`**. The archived reference is at `validation/mpi_result.csv` in the project root; copy it into the GPU working directory before submitting:

```bash
cd experiments/gpu
cp ../../validation/mpi_result.csv ./mpi_result.csv
qsub gpu_test.pbs
```

The recorded validations passed: serial vs Pure MPI, Pure MPI vs Hybrid CPU, and Pure MPI vs GPU. The measured GPU maximum absolute error was approximately `1.33e-15`. The archived Hybrid validation PBS script (`experiments/hybrid/validation/validate_hybrid_retry.pbs`) uses previously compiled `hybrid_validation` and a matching `mpi_result.csv`; it is not a standalone build script. Rebuild the validation executable with the Hybrid compiler command above before using it in a fresh environment.

## Performance experiments on Vanda

Run each command from the directory indicated. **Existing result files and submission locks are intentionally preserved; do not rerun submission scripts over an archived experiment directory.** To reproduce the experiments, use a clean working copy containing the source code and scripts, without archived result CSVs or submission locks. The Strong and Weak Scaling submission scripts check for existing result files even in dry-run mode. Inspect the generated PBS scripts before submitting any jobs.

### Serial scaling

```bash
cd experiments/serial
# serial_scaling.pbs compiles serial.cpp in the PBS working directory.
cp ../../serial.cpp ./serial.cpp
qsub serial_scaling.pbs
```

Grid sizes: `512, 1024, 2048, 4096`; `Steps=1000`; three repetitions per size. Output: `serial_scaling.csv`.

### Pure MPI strong scaling

```bash
cd experiments/mpi_strong
bash submit_strong.sh --dry-run
bash submit_strong.sh --submit
```

Fixed `N=4096`, `Steps=1000`; MPI ranks `1, 2, 4, 8, 16, 32, 64, 128, 256`; three repetitions each. Data: `results/strong_P*.csv`; PBS logs: `logs/`.

### Pure MPI weak scaling

```bash
cd experiments/mpi_weak
bash submit_weak.sh --dry-run
bash submit_weak.sh --submit
```

`(Ranks, N)` configurations: `(1,512), (4,1024), (16,2048), (64,4096), (256,8192)`; `Steps=1000`; three repetitions each. Data: `results/weak_P*.csv`.

### Hybrid MPI + OpenMP configuration comparison

```bash
cd experiments/hybrid/config_32
bash submit_hybrid32.sh --dry-run
bash submit_hybrid32.sh --submit
```

Fixed `N=4096`, `Steps=1000`, total 32 allocated CPU cores. Rank/thread pairs: `1×32`, `2×16`, `4×8`, `8×4`, `16×2`, `32×1`; three repetitions each. `submit_hybrid32.sh` generates the CPU-affinity launcher and PBS job files. Data: `results/hybrid_P*.csv`.

The local archive also includes **supplementary runs** of `2×16`, `4×8`, and `8×4` under `config_32/rerun/results/`. They were performed as separate PBS jobs on different nodes and should not be silently substituted for the original batch.

### GPU benchmark

```bash
cd experiments/gpu
bash submit_gpu_benchmark.sh
```

This submits `gpu_benchmark.pbs`, which builds the GPU program and runs one MPI rank on one GPU for `N=64, 512, 1024, 2048, 4096`, with `Steps=1000` and three repetitions per size (15 runs). Results are saved under `results_gpu/` in the PBS working directory:

- `gpu_benchmark.csv`: individual runs;
- `gpu_summary.csv`: grouped mean and standard deviation;
- `logs/`: per-run program output.

The optional `TEST_SHARED_RANKS=1` mode tests multiple ranks sharing **one GPU**; it is not multi-GPU strong scaling and was not part of the default benchmark.

In this downloaded project archive, the completed benchmark outputs are under `experiments/gpu/benchmark/results_gpu/` and the successful PBS log is under `experiments/gpu/benchmark/`. The PBS script itself writes to `experiments/gpu/results_gpu/` when submitted from `experiments/gpu/`; the `benchmark/` location is the local archival copy.

## Timing and interpretation

Performance outputs include `Runtime`, `CommunicationTime`, and `ComputationTime`. MPI communication time includes time waiting inside communication calls; it is **not** pure network transfer time. Timings may be reduced as separate maxima across ranks, so communication and computation times need not sum to total runtime.

For GPU offloading, the reported computation phase includes OpenMP target mapping and host–device transfer overhead, not just GPU kernel execution. The measured GPU implementation was slower than the tested CPU implementations at `N=4096`, `Steps=1000`; the per-iteration data-mapping strategy is a plausible contributor, but kernel and transfer times were not measured separately.

Hybrid CPU results showed substantial **between-job/node variability**. The original and supplementary batches should be reported separately, with the hardware/environment limitation disclosed; they do not establish a universally optimal rank/thread split.

## Reproducibility notes

The source code and scripts in this project document the experiments described in the accompanying report. PBS project allocations and module versions are Vanda-specific. For a fresh reproduction, use a clean working copy and follow the commands above; archived CSVs, binaries, and submission locks should not be treated as inputs to a new performance run. AI assistance is disclosed in the accompanying report.
