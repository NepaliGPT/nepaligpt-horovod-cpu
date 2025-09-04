#!/usr/bin/env bash
#SBATCH -J hvd_cpu_sif
#SBATCH -p <your_cpu_partition>
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH -t 04:00:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

set -euo pipefail

module purge
module load singularity || module load apptainer

# Tune threads per rank; align with --cpus-per-task
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Build once (outside the job) or point to a prebuilt image
# singularity build horovod_cpu.sif singularity/horovod_cpu.def

# Run your training inside the container
srun singularity exec horovod_cpu.sif \
     python -u src/pytorch/pt_synthetic_cpu.py
