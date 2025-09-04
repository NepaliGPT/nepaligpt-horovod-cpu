#!/usr/bin/env bash
# Open an interactive SLURM session on CPU nodes
# Usage:
#   PARTITION=<your_cpu_partition> NODES=2 NTASKS_PER_NODE=4 TIME=02:00:00 bash bin/srun_interactive_cpu.sh
set -euo pipefail

PARTITION="${PARTITION:-<your_cpu_partition>}"
NODES="${NODES:-2}"
NTASKS_PER_NODE="${NTASKS_PER_NODE:-4}"
TIME="${TIME:-02:00:00}"

salloc -p "${PARTITION}" -N "${NODES}" --ntasks-per-node="${NTASKS_PER_NODE}" -t "${TIME}"
