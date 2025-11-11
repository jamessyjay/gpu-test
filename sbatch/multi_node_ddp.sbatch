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

# Optional toggles
TRAIN_SMOKE=${TRAIN_SMOKE:-0}
EPOCHS=${EPOCHS:-1}
STEPS=${STEPS:-10}
REPORT_DIR=${REPORT_DIR:-/app/reports}
VERBOSE=${VERBOSE:-1}

ARGS="--report-dir ${REPORT_DIR} --report-name ddp_tests_result.json"
if [ "${VERBOSE}" = "1" ]; then
  ARGS="--verbose ${ARGS}"
fi
if [ "${TRAIN_SMOKE}" = "1" ]; then
  ARGS="--train-smoke --epochs ${EPOCHS} --steps ${STEPS} ${ARGS}"
  # Change report name to separate file for training smoke, unless overridden externally
  ARGS=$(echo "$ARGS" | sed "s/ddp_tests_result.json/ddp_training_result.json/")
fi

srun --container-image=ghcr.io/${OWNER}/gpu-cluster-acceptance:latest \
    bash -lc "torchrun \
      --rdzv_backend=${BACKEND} \
      --rdzv_endpoint=${MASTER_ADDRESS}:${MASTER_PORT} \
      --nnodes=${NODES} \
      --nproc_per_node=${GPUS_PER_NODE} \
      /app/src/ddp_tests.py ${ARGS}"