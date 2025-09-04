# Distributed Deep Learning using Horovod on TU-HPC (CPU-only)

This repository promotes large-scale **distributed deep learning on CPU clusters** and guides users to run their PyTorch or PyTorch Lightning codes across multiple **CPU nodes** using **Horovod (MPI/Gloo)** on TU-HPC or similar SLURM-managed systems.

---

## Contents

- TU-HPC CPU Cluster (context)
- Motivations for large-scale training under resource constraints
- Distributed DL training practices on supercomputers
- Installing Conda (CPU setup)
- Why Horovod for distributed DL
- Building Horovod (CPU-only)
- Horovod Usage (PyTorch)
- Running Horovod interactively (CPU)
- Submitting & Monitoring a Horovod batch job (CPU)
- Running Jupyter on a worker node (CPU)
- Why Singularity/Apptainer Containers
- Running Horovod with Singularity (CPU)
- Building a Horovod Singularity image on scratch
- Submitting & Monitoring a Horovod batch job using Singularity
- A glimpse of running PyTorch Lightning (CPU)
- Reference

---

## TU-HPC CPU Cluster

TU-HPC provides multi-node CPU resources (SLURM job scheduler, OpenMPI/Gloo for collectives, shared storage). This repository assumes:
- SLURM for resource management and scheduling.
- OpenMPI available as a module (or system package).
- Outbound internet access for conda/pip (or mirrored wheels/conda channels if air-gapped).
- Basic user storage (home/scratch) for environments, datasets, and logs.

---

## Motivations for Large-Scale Training under Resource Constraints

- Compute scarcity is common in universities and public research labs.
- CPUs are abundant, even when GPUs are limited.
- Can impactful LLM & NLP research be conducted on CPU clusters?  
  Consider:
  - Efficient data pipelines and sharding strategies.
  - Communication-aware training (fewer, larger per-rank batches).
  - Parameter-efficient fine-tuning (PEFT/LoRA) for downstream tasks.
- What can TU-HPC do for national NLP capacity building?
  - Provide accessible, documented workflows.
  - Offer fair scheduling policies and easy container interfaces.
  - Encourage shared datasets and reproducible benchmarks.

---

## Distributed DL Training Practices on Supercomputers

Establish routines to help researchers and cluster admins collaborate:

- Favor **simple, reproducible** launch patterns (SLURM + `srun`).
- Start single-node; scale up if throughput scales.
- Measure **tokens/sec**, wall time/epoch, and scaling efficiency.
- Use **node-local storage** where possible to avoid NFS bottlenecks.
- Keep environments consistent (Conda or Singularity).

---

## Installing Conda (CPU setup)

```bash
$ cat /etc/*release*


Download and install Miniconda to your scratch directory:

$ cd /scratch/$USER
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh --no-check-certificate
$ chmod 755 Miniconda3-latest-Linux-x86_64.sh
$ ./Miniconda3-latest-Linux-x86_64.sh
# Accept license; set install prefix to /scratch/$USER/miniconda3
# Allow conda init → yes
```

Finalize:
```bash
$ source ~/.bashrc
$ conda config --set auto_activate_base false
$ which conda
/scratch/$USER/miniconda3/condabin/conda
$ conda --version

```

Create the environment:

```bash
$ conda env create -f environment.yml
$ conda activate hvd-cpu
```

## Why Horovod for Distributed DL

Horovod (Uber, LF AI) simplifies scaling PyTorch/TensorFlow across nodes:

- Framework-friendly: works with PyTorch, Lightning, etc.

- Minimal code changes: wrap optimizer, broadcast states, scale LR.

- Backends: MPI or Gloo for CPU; (NCCL for GPUs—out of scope here).

- Familiar launchers: mpirun, srun, or horovodrun.

## Building Horovod (CPU-only)

Using the provided environment files:
```bash
# Inside hvd-cpu env
(hvd-cpu) $ python -c "import horovod.torch as hvd; hvd.init(); print('OK', hvd.rank(), hvd.size())"
```

Alternatively, install via pip:
```bash
(hvd-cpu) $ pip install "horovod[pytorch]>=0.28"
```

Verify:
```bash
(hvd-cpu) $ horovodrun -cb   # optional capability check if available
```

## Horovod Usage (PyTorch)

Five key steps in PyTorch:

1. Initialize
```bash
import horovod.torch as hvd
hvd.init()
```

2. Set CPU threading wisely
```bash
import torch
torch.set_num_threads(max(1, torch.get_num_threads() // max(1, hvd.local_size())))
```

3. Scale LR & wrap optimizer
```bash
opt = torch.optim.AdamW(model.parameters(), lr=base_lr * hvd.size())
opt = hvd.DistributedOptimizer(opt, named_parameters=model.named_parameters())
```

4. Broadcast parameters & optimizer state
```bash
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(opt, root_rank=0)
```

5. Shard data
```bash
from torch.utils.data import distributed
sampler = distributed.DistributedSampler(ds, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=True)
```

See:

```bash src/pytorch/pt_synthetic_cpu.py``` (tiny LM on synthetic data)

```bash src/pytorch/pt_mnist_cpu.py``` (MNIST classifier)

## Running Horovod Interactively (CPU)

Request an interactive CPU session:
```bash
(hvd-cpu) $ salloc -p <cpu_partition> -J debug --nodes=2 --time=02:00:00 --ntasks-per-node=4
```


Load MPI if needed and run:
```bash
$ module load openmpi
$ srun -n 8 python -u src/pytorch/pt_synthetic_cpu.py
# or
$ srun -n 8 python -u src/pytorch/pt_mnist_cpu.py

```

Single-node mpirun test:
```bash
(hvd-cpu) $ NP=4 OMP_NUM_THREADS=2 bash bin/mpirun_local.sh
```
## Submitting & Monitoring a Horovod Batch Job (CPU)

Edit and submit:
```bash
$ cat bin/horovod_batch_cpu.sh
$ sbatch bin/horovod_batch_cpu.sh
```

Monitor:
```bash
$ squeue -u $USER
$ sacct -j <jobid> --format=JobID,State,Elapsed
$ tail -f hvd_cpu_demo_<jobid>.out
```

Cancel:
```bash
$ scancel <jobid>
```

## Running Jupyter on a Worker Node (CPU)

Submit the helper:
```bash
$ sbatch bin/jupyter_run_cpu.sh
$ cat port_forwarding_command
ssh -L 8888:<worker-host>:<port> $USER@<login-host>
```

Open ```bash http://localhost:8888 ``` in your browser. Token is your $USER by default.

## Why Singularity/Apptainer Containers

Singularity is suited for HPC:

- Single image file; no root daemon.

- Plays well with MPI and multi-tenant clusters.

- Reproducible environments across nodes.

## Running Horovod with Singularity (CPU)

Once you have an image (build outside job or use a prebuilt one):
```bash
$ srun singularity exec horovod_cpu.sif \
      python -u src/pytorch/pt_synthetic_cpu.py
```

## Building a Horovod Singularity Image on Scratch

Example recipe: ```bash singularity/horovod_cpu.def```. Build:
```bash
$ singularity build horovod_cpu.sif singularity/horovod_cpu.def
```

## Submitting & Monitoring a Horovod Batch Job using Singularity
```bash
$ cat singularity/singularity_horovod_batch_cpu.sh
$ sbatch singularity/singularity_horovod_batch_cpu.sh
$ squeue -u $USER
```

## A Glimpse of Running PyTorch Lightning (CPU)

Install Lightning (already in ```bash environment.yml```), then inside an allocation:
```bash
(hvd-cpu) $ salloc -p <cpu_partition> -N 2 --ntasks-per-node=4 -t 02:00:00
$ module load openmpi
$ srun -n 8 python -u src/pytorch-lightning/pl_mnist_cpu.py
```

Lightning with HorovodStrategy handles samplers and process setup for you.

## Reference

Horovod documentation: https://horovod.ai

PyTorch: https://pytorch.org

Lightning: https://lightning.ai

For inspiration: [distributed training practices on national labs and academic supercomputers (NERSC, etc.).](https://github.com/hwang2006/distributed-training-on-perlmutter-using-horovod) Adapt parameters and scripts to match your site’s modules, partitions, and security policies.

### License: Apache-2.0 (see LICENSE)
### Project: NepaliGPT — resource-constrained, inclusive NLP for Nepali and related languages.