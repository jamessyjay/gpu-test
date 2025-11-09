# Slurm HOWTO: GPU Acceptance Tests

Short, practical guide for DevOps: how to run GPU acceptance checks on a Slurm cluster with containers (Pyxis/Enroot).

## Prerequisites
- NVIDIA GPUs and driver installed on nodes.
- Pyxis/Enroot enabled in Slurm (use `srun --container-image`).
- Access to the image: `ghcr.io/<OWNER>/gpu-cluster-acceptance:latest`.

## Quick start
- Single-node (1 node, 1 GPU):
  ```bash
  export OWNER=<your-ghcr-username>
  sbatch sbatch/single_node_gpu.sbatch
  ```
- Multi-node DDP (N nodes, M GPUs per node):
  ```bash
  export OWNER=<your-ghcr-username>
  # adjust -N and --gpus-per-node inside sbatch/multi_node_ddp.sbatch if needed
  sbatch sbatch/multi_node_ddp.sbatch
  ```

## What the scripts do
- `sbatch/single_node_gpu.sbatch`:
  - Runs the container and inside it `python src/run_all_tests.py --quick`.
  - Executes both tests: `gpu_tests.py` (compute + training) and `ddp_tests.py` (if GPUs>1), writes JSON to `/app/reports`.
- `sbatch/multi_node_ddp.sbatch`:
  - Configures rendezvous (`--rdzv_backend=c10d`, `--rdzv_endpoint=<master>:29500`).
  - Runs `torchrun --nnodes=$SLURM_NNODES --nproc_per_node=$SLURM_GPUS_ON_NODE /app/src/ddp_tests.py --verbose`.
  - Reports are written to `/app/reports`.

## Variables and parameters
- `OWNER` — owner of the GHCR image. Default: `OWNER` (override via export).
- In `single_node_gpu.sbatch`:
  - `--gpus-per-node=1` — increase up to available GPUs on the node if desired.
- In `multi_node_ddp.sbatch`:
  - `#SBATCH -N <N>` — number of nodes.
  - `#SBATCH --gpus-per-node=<M>` — GPUs per node.
  - Master node for rendezvous is taken from `$SLURM_NODELIST`.

## Collecting logs and reports
- Logs:
  ```bash
  # Single-node
  squeue -u $USER
  cat gpu-accept-smoke.<jobid>.out

  # Multi-node
  cat gpu-accept-ddp.<jobid>.out
  ```
- JSON reports:
  - No host volume is mounted by default; use logs as the primary source.
  - If you need persistent reports, adapt to Enroot/Pyxis `--container-mounts` to persist `/app/reports` on the host.

## What the tests do
- `src/gpu_tests.py`:
  - Auto-detects GPUs; if none, skips (exit 0).
  - Compute GFLOP/s (matrix multiplies) per GPU + short training (loss reduction).
  - Flags: `--quick`, `--verbose`, `--report-dir`, `--report-name`.
- `src/ddp_tests.py`:
  - NCCL backend, auto-scales `iters`/`numel` when set to 0.
  - Validates all-reduce result, logs on rank0 and writes JSON (if path provided).

## Troubleshooting
- No GPU inside the container: check Pyxis/Enroot config and `--gpus-per-node` limits.
- NCCL errors/hangs on init: verify network between nodes, port 29500 reachability and DNS records.
- CUDA/torch versions: the image should match the CUDA version used in the conda env.
