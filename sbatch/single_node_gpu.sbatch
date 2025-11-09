#!/bin/bash
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -J gpu-accept-smoke
#SBATCH -o %x.%j.out

# if you have Pyxis/Enroot enabled: use --container-image
OWNER=${OWNER:-OWNER}
srun --container-image=ghcr.io/${OWNER}/gpu-cluster-acceptance:latest \
    python src/run_all_tests.py --quick --report-dir /app/reports --report-name summary.json