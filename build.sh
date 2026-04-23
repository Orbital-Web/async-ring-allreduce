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
nvcc -o benchmark \
    -arch=sm_80 \
    src/benchmark.cu src/utils.cu \
    src/nccl_ringreduce.cu \
    src/naive_ringreduce.cu \
    src/pipelined_ringreduce_nccl.cu \
    src/hier_ringreduce.cu \
    -I${NCCL_HOME}/include -L${NCCL_HOME}/lib \
    -I${CRAY_MPICH_PREFIX}/include -L${CRAY_MPICH_PREFIX}/lib \
    -lnccl \
    -lmpi -lxpmem -lmpi_gtl_cuda \
    $EXTRA_FLAGS