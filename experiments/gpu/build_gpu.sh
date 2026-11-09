#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
module load nvhpc-hpcx-cuda12/26.1
command -v nvc++ >/dev/null || { echo 'ERROR: nvc++ unavailable'; exit 1; }
command -v mpicxx >/dev/null || { echo 'ERROR: MPI C++ wrapper unavailable. Load a compatible MPI module.'; exit 1; }
# Prefer the MPI wrapper configured to use the NVHPC compiler.
if mpicxx --showme:compile >/dev/null 2>&1; then
  read -r -a mpi_cflags <<< "$(mpicxx --showme:compile)"
  read -r -a mpi_lflags <<< "$(mpicxx --showme:link)"
  nvc++ -O3 -mp=gpu -gpu=cc86 -Minfo=mp "${mpi_cflags[@]}" hybrid_gpu.cpp -o hybrid_gpu "${mpi_lflags[@]}"
else
  echo 'ERROR: MPI wrapper does not support OpenMPI --showme flags.'
  echo 'Please use an OpenMPI wrapper compatible with nvc++, or adjust MPI compile/link flags.'
  exit 1
fi
