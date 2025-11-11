# GPU Cluster Acceptance — README

**Uncompressed image size (nvidia):** 20.20GB
**Uncompressed image size (mamba):** 10.10GB
**Uncompressed image size (latest):** 5.53GB

## Goal
- Check nodes and GPU cluster "as in real life", not only with synthetic data.
- Verify:
  - GPU are visible and working.
  - Computational performance has not degraded.
  - CUDA/PyTorch/NCCL stack is correctly configured.

Tests inside the container:
- **Compute test**: matrices on each GPU → GFLOP/s.
- **Training smoke**: tiny model, training for a few epochs → loss decreases.
- **DDP all-reduce**: communication/NCCL sanity via src/ddp_tests.py.
- **DDP training smoke**: tiny distributed training (`--train-smoke`) to verify loss decreases across ranks.

All of the above are orchestrated by the unified runner: `src/run_all_tests.py`.

Note: when no GPU is detected, the test script exits successfully (skips tests). This is used by CI where runners have no GPUs.


## Composition
- **Dockerfile.mamba**: multi-stage build with micromamba.
- GPU stage: `nvidia/cuda:12.6.1-runtime-ubuntu24.04` (last stage, builds by default).
- **src/gpu_tests.py**: main script. By default GPU image runs quick test via `CMD ["python","src/gpu_tests.py","--quick"]`.
- **sbatch/**: Slurm scripts for single node and multi-node DDP.


## Container building
```bash
# from the root of the repo
export IMAGE=ghcr.io/jamessyjay/gpu-cluster-acceptance

# GPU image (default, last stage)
docker build -f Dockerfile.mamba -t ${IMAGE}:latest .
```

## Running locally (Docker)
```bash
export IMAGE=ghcr.io/jamessyjay/gpu-cluster-acceptance

# Quick full suite (compute + train + DDP where applicable)
docker run --rm --gpus all ${IMAGE}:latest \
  python src/run_all_tests.py --quick --report-dir /app/reports

# Full suite (longer)
docker run --rm --gpus all ${IMAGE}:latest \
  python src/run_all_tests.py --report-dir /app/reports

# Multi-GPU DDP all-reduce only
docker run --rm --gpus all ${IMAGE}:latest \
  bash -lc 'torchrun --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) src/ddp_tests.py --iters=50 --numel=2000000 --report-dir /app/reports'

# Multi-GPU DDP training smoke (tiny distributed training)
docker run --rm --gpus all ${IMAGE}:latest \
  bash -lc 'torchrun --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) src/ddp_tests.py --train-smoke --epochs 1 --steps 10 --report-dir /app/reports'
```

Note: container uses `ENTRYPOINT ["/usr/bin/tini","--"]` with CMD to run the script.


## Running on the cluster (Slurm)
- Single node: 
  ```bash
  sbatch sbatch/single_node_gpu.sbatch
  ```

- Multi-node / Distributed Data Parallel (DDP): 
  ```bash
  sbatch sbatch/multi_node_ddp.sbatch
  ```

Notes:
- Scripts assume correct integration with containers (Pyxis/Enroot) and access to the image.
- Adjust any parameters (partition, gpus-per-node, ntasks-per-node, container-image, etc.) for your cluster.

## Deployment: Slurm (Pyxis/Enroot)
Minimal options for DevOps to run acceptance checks via Slurm.

```bash
# image is pushed by CI to GHCR
export IMAGE=ghcr.io/jamessyjay/gpu-cluster-acceptance:latest

# interactive run on a GPU node (Pyxis)
srun -A <account> -p <gpu-partition> -N 1 -n 1 --gpus 1 \
  --container-image=${IMAGE} \
  python /app/src/gpu_tests.py --quick

# single-node multi-GPU (use all visible GPUs)
srun -A <account> -p <gpu-partition> -N 1 --gpus-per-node=<N> \
  --container-image=${IMAGE} \
  bash -lc 'torchrun --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) /app/src/ddp_tests.py --iters=50 --numel=2000000'

# multi-node DDP (example)
srun -A <account> -p <gpu-partition> -N ${NNODES} --gpus-per-node=${GPUS} \
  --container-image=${IMAGE} \
  bash -lc 'torchrun --nnodes=${NNODES} --nproc_per_node=${GPUS} --rdzv_backend=c10d \
  --rdzv_endpoint=$SLURM_NODELIST:29500 /app/src/ddp_tests.py --iters=50 --numel=2000000'
```

Tips:
- Ensure Pyxis/Enroot is enabled and users have pull access to GHCR.
- NCCL envs can be tuned via `--container-env`: e.g. `NCCL_DEBUG=INFO`.
- For air-gapped: mirror the image to internal registry and adjust `IMAGE`.


## How to interpret the output
Sections with prefixes:
- `[ENV]` environment info: Python, torch, `cuda_device_count`, `nvidia-smi -L`.
- `[COMPUTE]` per GPU: GFLOP/s and test parameters.
- `[TRAIN]` training results: start/finish loss, improved=True/False.
- In `run_all_tests.py` summary you will see rc for `gpu_tests`, `ddp_tests` and `ddp_train`.
- Result:
  - `[RESULT] SUCCESS` — everything is fine. GPU calculates, losses decrease.
  - `[RESULT] FAILURE: [...]` — error(s). Process will exit with code ≠ 0.

For Distributed Data Parallel (DDP):
- All-reduce: expect `[DDP] ... ok=True` on rank0.
- DDP training smoke: JSON `ddp_training_result.json` includes `improved=true/false` and start/end loss.
- If not — check NCCL, network, GPU visibility, environment variables.

Example of expected "healthy" output:
- GFLOP/s per each available GPU — numbers in the same order as `nvidia-smi -L`.
- In `TRAIN` loss decreases noticeably (script checks improvement by a threshold).
- Final `[RESULT] SUCCESS`.


## Tuning and flags
- Runner `run_all_tests.py`:
  - `--quick` — quick run (fewer iterations/steps).
  - `--verbose` — detailed DEBUG logs.
  - `--report-dir`, `--report-name` — write JSONs.
- DDP `src/ddp_tests.py`:
  - All-reduce: `--iters`, `--numel`, `--verbose`, `--report-dir`, `--report-name`.
  - Training smoke: `--train-smoke` with optional `--epochs`, `--batch-size`, `--steps`.

## Changes in this revision
- Removed CPU tests and CPU CI job. CI now only builds the GPU image.
- Test script skips and exits 0 when no GPU is detected (useful on GitHub-hosted runners).


## Limitations and further extensions
- No strict memory/stress tests.
- No IO-bench: consider adding fio/NVMe tests.
- Full NCCL-bench integration (e.g., `all_reduce_perf` from nccl-tests) and comparing with benchmarks.
- Can add throughput tests (ResNet-50 FP16/AMP inference), baseline numbers for specific GPUs.
- Save history of results for monitoring degradation over time.

## Common causes of problems
- NVIDIA driver is incompatible with the CUDA version in the container.
- NCCL/Network environment variables (P2P, IB/RoCE) are not configured.
- No GPU visibility in the container (check Pyxis/Enroot launch or `--gpus all`).
- Nodes are busy — real GFLOP/s drop.

## Debugging
- Run with `--verbose`.
- Compare `[ENV]` and `nvidia-smi -L` with the host.
- Locally: `docker run --rm --gpus all nvidia/cuda:12.6.1-base-ubuntu24.04 nvidia-smi` — runtime check.
- On the cluster: check Slurm allocations `sinfo`, `squeue`, partition quotas.

## Deployment: Kubernetes (GPU nodes)
Minimal Job spec (requests 1 NVIDIA GPU):

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: gpu-accept
spec:
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: tester
          image: ghcr.io/jamessyjay/gpu-cluster-acceptance:latest
          resources:
            limits:
              nvidia.com/gpu: 1
          command: ["python", "src/gpu_tests.py", "--quick"]
```

Notes:
- Requires NVIDIA device plugin on the cluster.
- For multi-GPU per pod: set `nvidia.com/gpu: <N>` and use `torchrun` similarly to Docker example.
- For private GHCR: create an `imagePullSecret` and reference it in the pod spec.
