#!/usr/bin/env bash
#SBATCH -J jupyter_cpu
#SBATCH -p <your_cpu_partition>
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -t 08:00:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

set -euo pipefail

module purge
source ~/.bashrc
conda activate hvd-cpu

# Pick a random high port and print an SSH tunnel helper
PORT_JU=$((RANDOM + 10000))
HOSTNAME_FQDN="$(hostname)"

# Replace <login-host> with your site’s SSH entry point (e.g., login.hpc.edu)
echo "ssh -L 8888:${HOSTNAME_FQDN}:${PORT_JU} ${USER}@<login-host>" > port_forwarding_command
echo "To open the tunnel, run:"
cat port_forwarding_command

# Start Jupyter Lab bound to all interfaces; token = your username
jupyter lab --ip=0.0.0.0 --port="${PORT_JU}" --NotebookApp.token="${USER}" --no-browser
