# GPU Cluster Acceptance — README


## Goal
- Check nodes and GPU cluster "as in real life", not only with synthetic data.
- Verify:
  - GPU are visible and working.
  - Computational performance has not degraded.
  - CUDA/PyTorch/NCCL stack is correctly configured.

Tests inside the container:
- **Compute test**: matrices on each GPU → GFLOP/s.
- **Training smoke**: tiny model, training for a few epochs → loss decreases.
- _[*Optionally*]_: DDP/multi-node checks via sbatch examples.


## Composition
- **Dockerfile**: base image `nvcr.io/nvidia/pytorch:24.07-py3`, minimal dependencies.
- **src/gpu_tests.py**: main script, runs by default (`ENTRYPOINT ["python", "src/gpu_tests.py"]`).
- **sbatch/**: Slurm scripts for single node and multi-node DDP.


## Container building
```bash
# from the root of the repo
docker buildx build -t gpu-accept:latest .
```

## Running locally (Docker)
```bash
# full run on all visible GPUs
docker run --rm --gpus all gpu-accept:latest

# quick run (fewer iterations/matrix sizes)
docker run --rm --gpus all gpu-accept:latest --quick

# CPU-режим (для CI без GPU)
docker run --rm gpu-accept:latest --cpu-only

# подробные логи
docker run --rm --gpus all gpu-accept:latest --verbose
```

Arguments are passed directly, since `ENTRYPOINT` is already set.


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


## How to interpret the output
Sections with prefixes:
- `[ENV]` environment info: Python, torch, `cuda_device_count`, `nvidia-smi -L`.
- `[COMPUTE]` per GPU: GFLOP/s and test parameters.
- `[TRAIN]` training results: start/finish loss, improved=True/False.
- Result:
  - `[RESULT] SUCCESS` — everything is fine. GPU calculates, losses decrease.
  - `[RESULT] FAILURE: [...]` — error(s). Process will exit with code ≠ 0.

For Distributed Data Parallel (DDP):
- When running multi-node/multi-GPU test, expect a line like:
  - `[ddp] ... ok=True` — communication/synchronization is ok.
  - If not — check NCCL, network, GPU visibility, environment variables.

Example of expected "healthy" output:
- GFLOP/s per each available GPU — numbers in the same order as `nvidia-smi -L`.
- In `TRAIN` loss decreases noticeably (script checks improvement by a threshold).
- Final `[RESULT] SUCCESS`.


## Tuning and flags
- `--quick` — quick run (fewer iterations and matrix sizes).
- `--cpu-only` — force CPU path (useful in CI).
- `--verbose` — detailed DEBUG logs.


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
- Locally: `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` — runtime check.
- On the cluster: check Slurm allocations `sinfo`, `squeue`, partition quotas.

