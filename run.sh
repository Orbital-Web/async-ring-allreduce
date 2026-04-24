#!/bin/bash
# default topology: 2 nodes x 4 GPUs = 8 ranks. good for ring / hierarchical / HD.
# PAARD requires 2 GPUs/node and n_ranks=6 — for PAARD runs, change the three
# SBATCH directives below to `--nodes=3 --ntasks-per-node=2 --gres=gpu:2` and
# set N_RANKS=6. PAARD is skipped automatically if n_ranks != 6.
#SBATCH --job-name=allreduce_bench
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --constraint gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --account m4341_g
#SBATCH --time=00:30:00

#SBATCH --output=results/bench_%j.csv
#SBATCH --error=results/bench_%j.err

DEBUG="on"
N_RANKS=8

for arg in "$@"; do
    case $arg in
        -r)
            echo "Running in Release Mode"
            DEBUG="off"
            shift
            ;;
        -n=*)
            N_RANKS="${arg#*-n=}"
            shift
            ;;
    esac
done

if [[ "$DEBUG" == "on" ]]; then
    export NCCL_DEBUG=INFO
else
    export NCCL_DEBUG=WARN
fi

# export FI_CXI_ATS=0
# export OFI_NCCL_DISABLE_DMABUF=1
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_EAGER_SIZE=0

# synthetic inter-node link penalty (applied only to hierarchical impl), LogGP-style:
#   delay(bytes) = GLOBAL_PENALTY_US us  +  bytes / GLOBAL_BW_GBPS ns
# GLOBAL_PENALTY_US: fixed per-hop latency in microseconds (default 0)
# GLOBAL_BW_GBPS:    inter-node bandwidth cap in GB/s (default 0 = unlimited)
# example sweeps:
#   GLOBAL_PENALTY_US=100 sbatch run.sh         # latency-only
#   GLOBAL_BW_GBPS=5 sbatch run.sh              # bandwidth-only (scales with msg size)
#   GLOBAL_PENALTY_US=50 GLOBAL_BW_GBPS=10 sbatch run.sh   # both
export GLOBAL_PENALTY_US=${GLOBAL_PENALTY_US:-0}
export GLOBAL_BW_GBPS=${GLOBAL_BW_GBPS:-0}

# real inter-node slowdown (no simulation). two knobs, either may be effective
# depending on which layer of the stack honors it:
#   NCCL_NET_GDR_LEVEL=0  — NCCL-core hint (often shadowed by the OFI plugin on Perlmutter)
#   OFI_NCCL_DISABLE_DMABUF=1 — plugin-level disable of the kernel-bypass data path (more reliable)
# both force inter-node transfers to stage through host memory, affecting every impl.
# use for real-hardware validation of the simulated sweep.
# examples:
#   NCCL_NET_GDR_LEVEL=0     sbatch run.sh -r
#   OFI_NCCL_DISABLE_DMABUF=1 sbatch run.sh -r
if [[ -n "${NCCL_NET_GDR_LEVEL}" ]]; then
    export NCCL_NET_GDR_LEVEL
fi
if [[ -n "${OFI_NCCL_DISABLE_DMABUF}" ]]; then
    export OFI_NCCL_DISABLE_DMABUF
fi

module purge
module load PrgEnv-gnu
module load cudatoolkit
module load cray-mpich
module load nccl

srun -u --cpus-per-task=32 --cpu-bind=cores ./benchmark "$N_RANKS"