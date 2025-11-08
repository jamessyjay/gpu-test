#!/bin/bash
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -J gpu-accept-smoke
#SBATCH -o %x.%j.out

# if you have Pyxis/Enroot enabled: use --container-image
OWNER=${OWNER:-OWNER}
srun --container-image=ghcr.io/${OWNER}/gpu-cluster-acceptance:latest \
    python src/gpu_tests.py --quick