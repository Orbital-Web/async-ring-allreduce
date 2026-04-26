#!/usr/bin/env bash
set -euo pipefail

# parse args
EXTRA_FLAGS=""

for arg in "$@"; do
    case $arg in
        -r)
            echo "Building for release"
            EXTRA_FLAGS="-O2 -DNDEBUG"
            ;;
    esac
done

module purge
module load PrgEnv-gnu
module load cudatoolkit
module load cray-mpich
module load nccl

# TODO: add new impls here
nvcc -arch=sm_80 -o benchmark \
    src/benchmark.cu src/utils.cu \
    src/naive_ring.cu \
    src/pipelined_ring_nccl.cu \
    src/naive_hd.cu \
    src/pipelined_hd_nccl.cu \
    src/naive_paard.cu \
    src/pipelined_paard.cu \
    -I${NCCL_HOME}/include -L${NCCL_HOME}/lib \
    -I${CRAY_MPICH_PREFIX}/include -L${CRAY_MPICH_PREFIX}/lib \
    -lnccl \
    -lmpi -lxpmem -lmpi_gtl_cuda \
    $EXTRA_FLAGS