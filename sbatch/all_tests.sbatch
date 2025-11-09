#!/bin/bash
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -J gpu-accept-all
#SBATCH -o %x.%j.out

# OWNER: GHCR image owner (required)
OWNER=${OWNER:-OWNER}
# REPORT_DIR: where to write JSON reports inside the container
REPORT_DIR=${REPORT_DIR:-/app/reports}
# HOST_REPORTS: optional host path to persist reports (mounted into REPORT_DIR)
HOST_REPORTS=${HOST_REPORTS:-}
# QUICK: 1 to enable quick mode, 0 otherwise
QUICK=${QUICK:-1}
# VERBOSE: 1 to enable verbose logs, 0 otherwise
VERBOSE=${VERBOSE:-0}

ARGS="--report-dir ${REPORT_DIR} --report-name summary.json"
if [ "${QUICK}" = "1" ]; then
  ARGS="--quick ${ARGS}"
fi
if [ "${VERBOSE}" = "1" ]; then
  ARGS="--verbose ${ARGS}"
fi

# If Pyxis/Enroot is enabled, use --container-image. Otherwise adapt to your environment.
# Optionally mount host reports directory to persist artifacts.
MOUNTS_ARG=""
if [ -n "${HOST_REPORTS}" ]; then
  MOUNTS_ARG="--container-mounts=${HOST_REPORTS}:${REPORT_DIR}"
fi

srun ${MOUNTS_ARG} --container-image=ghcr.io/${OWNER}/gpu-cluster-acceptance:latest \
  bash -lc "python src/run_all_tests.py ${ARGS}"
