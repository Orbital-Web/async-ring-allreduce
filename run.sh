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

module purge
module load PrgEnv-gnu
module load cudatoolkit
module load cray-mpich
module load nccl

srun -u --cpus-per-task=32 --cpu-bind=cores ./benchmark