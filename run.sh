#!/bin/bash
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

for arg in "$@"; do
    case $arg in
        -r)
            echo "Running in Release Mode"
            DEBUG="off"
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

# synthetic inter-node link penalty, LogGP-style affine cost:
#   delay(bytes) = GLOBAL_PENALTY_US us  +  bytes / GLOBAL_BW_GBPS ns
# GLOBAL_PENALTY_US: fixed per-hop latency in microseconds (default 0)
# GLOBAL_BW_GBPS:    inter-node bandwidth cap in GB/s (default 0 = unlimited)
# example sweeps:
#   GLOBAL_PENALTY_US=100 sbatch run.sh         # latency-only
#   GLOBAL_BW_GBPS=5 sbatch run.sh              # bandwidth-only (scales with msg size)
#   GLOBAL_PENALTY_US=50 GLOBAL_BW_GBPS=10 sbatch run.sh   # both
export GLOBAL_PENALTY_US=${GLOBAL_PENALTY_US:-0}
export GLOBAL_BW_GBPS=${GLOBAL_BW_GBPS:-0}

module purge
module load PrgEnv-gnu
module load cudatoolkit
module load cray-mpich
module load nccl

srun -u --cpus-per-task=32 --cpu-bind=cores ./benchmark