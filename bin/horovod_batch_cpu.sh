#!/usr/bin/env bash
#SBATCH -J hvd_cpu_demo                   # job name
#SBATCH -p <your_cpu_partition>           # CPU partition/queue
#SBATCH -N 2                              # nodes
#SBATCH --ntasks-per-node=4               # ranks per node
#SBATCH --cpus-per-task=2                 # threads per rank (match OMP_NUM_THREADS)
#SBATCH -t 04:00:00                       # walltime
#SBATCH -o %x_%j.out                      # stdout
#SBATCH -e %x_%j.err                      # stderr

set -euo pipefail

module purge
# Load MPI if required by your cluster
module load openmpi || true

# Activate your environment
source ~/.bashrc
conda activate hvd-cpu

# Threading & tokenizer hygiene for CPU runs
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Launch ranks via SLURM
srun python -u scripts/train_hvd_torch.py --config configs/training_cpu.yaml
