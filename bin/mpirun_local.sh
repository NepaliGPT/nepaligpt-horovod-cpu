#!/usr/bin/env bash
# Single-node MPI run (CPU) for quick tests
# Usage:
#   NP=4 OMP_NUM_THREADS=2 bash bin/mpirun_local.sh
set -euo pipefail

NP="${NP:-4}"                     # number of ranks on this node
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Optional: load MPI if your environment needs it
# module load openmpi

mpirun -np "${NP}" \
  -bind-to none -map-by slot \
  -x OMP_NUM_THREADS \
  -x TOKENIZERS_PARALLELISM \
  -x PATH -x LD_LIBRARY_PATH \
  python -u scripts/train_hvd_torch.py --config configs/training_cpu.yaml
