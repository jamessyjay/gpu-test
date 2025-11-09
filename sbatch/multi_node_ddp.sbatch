#!/bin/bash
#SBATCH -N 2
#SBATCH --gpus-per-node=2
#SBATCH -J gpu-accept-ddp
#SBATCH -o %x.%j.out

MASTER_ADDRESS=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_PORT=29500
NODES=${SLURM_NNODES}
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-2}
BACKEND=c10d
OWNER=${OWNER:-OWNER}

srun --container-image=ghcr.io/${OWNER}/gpu-cluster-acceptance:latest \
    bash -lc "torchrun \
      --rdzv_backend=${BACKEND} \
      --rdzv_endpoint=${MASTER_ADDRESS}:${MASTER_PORT} \
      --nnodes=${NODES} \
      --nproc_per_node=${GPUS_PER_NODE} \
      /app/src/ddp_tests.py --verbose --report-dir /app/reports --report-name ddp_tests_result.json"