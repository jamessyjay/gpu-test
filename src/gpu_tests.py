# gpu-cluster-acceptance/src/gpu_tests.py
from argparse import ArgumentParser, Namespace
from logging import basicConfig, getLogger, INFO, DEBUG
from subprocess import check_output, CalledProcessError, STDOUT
from sys import version, exit
from time import time
from typing import Dict, Any, Tuple

# torch: Neural Network library
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


basicConfig(level=INFO)
logger = getLogger(__name__)


def set_logger_level(level: int = INFO) -> None:
    """
    Changes the logger level.
    :param level: int - logging level
    """
    global logger
    logger.setLevel(level)
    return logger


class ToyDataset(Dataset):
    """
    This class generates a synthetic dataset with linear separability.
    The model should learn fast.
    :param n_samples: int = 4000 - number of samples in the dataset
    :param input_features: int = 128 - number of input features required by the model for learning
    :param k_classes: int = 4 - number of classes required by the model for learning
    :param seed: int = 42 - random seed for reproducibility
    :param device: str = "cpu" - device to use for training
    """
    def __init__(self, 
                 n_samples: int = 4000, 
                 input_features: int = 128, 
                 k_classes: int = 4, 
                 seed: int = 42, 
                 device: str = "cpu") -> None:
        """
        Initialize the dataset with random input features (x) and labels (y).
        """
        # Initialize random number generator with seed and device.
        g = torch.Generator(device=device).manual_seed(seed)
        logger.debug(f"[DATASET] seed: {seed}")
        # Generate random input features (x) for the dataset.
        self.x = torch.randn(n_samples, input_features, generator=g, device=device)
        logger.debug(f"[DATASET] x.shape: {self.x.shape}")
        # Generate random weights (W) for the model.
        W = torch.randn(input_features, k_classes, generator=g, device=device)
        logger.debug(f"[DATASET] W.shape: {W.shape}")
        # Calculate logits for each data point using the input features and weights.
        logits = self.x @ W
        logger.debug(f"[DATASET] logits.shape: {logits.shape}")
        # Calculate labels for each data point based on the logits.
        self.y = logits.argmax(dim=1)
        logger.debug(f"[DATASET] y.shape: {self.y.shape}")
        # Store the dataset size, feature dimension, and number of classes.
        self.n, self.d, self.k = n_samples, input_features, k_classes
        # Store the device where the dataset is stored.
        self.device = device
    
    def __len__(self) -> int: 
        """
        Return the length of the dataset.
        """
        return self.n
    
    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]: 
        """
        Return the data point at index i along with its label.
        """
        return self.x[i], self.y[i]


class TinyMLP(nn.Module):
    """
    This class is inherited from nn.Module and generates a synthetic model with 
    linear separability. The model should learn fast.
    :param input_features: int = 128 - number of input features required by the 
    model for learning
    :param k_classes: int = 4 - number of classes required by the model for learning
    """
    def __init__(self, input_features: int = 128, k_classes: int = 4) -> None:
        """
        Initialize the model with a sequential neural network.
        """
        super().__init__()
        out_features = 256      # 256 because it is a small model
        logger.debug(f"[MODEL] out_features: {out_features}")
        self.net = nn.Sequential(
            nn.Linear(input_features, out_features), nn.ReLU(),
            nn.Linear(out_features, k_classes)
        )
        logger.debug(f"[MODEL] net: {self.net}")    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        """
        Forward pass of the model.
        :param x: torch.Tensor - input features
        :return: torch.Tensor - output logits
        """
        logger.debug(f"[MODEL] x.shape: {x.shape}")
        return self.net(x)


def shell(cmd: str) -> str:
    """
    Execute a local shell command and return its output.
    :param cmd: str - command to execute
    :return: str - output of the command
    """
    try:
        logger.debug(f"[SHELL] cmd: {cmd}")
        out = check_output(cmd, shell=True, stderr=STDOUT, text=True)
        logger.debug(f"[SHELL] out: {out}")
        return out.strip()
    except CalledProcessError as e:
        logger.debug(f"[SHELL] e: {e}")
        return f"[ERROR] shell error:\n{e.output}"


def _get_environment() -> Dict[str, Any]:
    """
    Get the environment and return a dictionary of information.
    :return: Dict[str, Any] - dictionary of information about the environment
    """
    env_info = {
        "python": version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "nvidia_smi": shell("nvidia-smi -L") if torch.cuda.is_available() else "no GPU"
    }
    logger.debug(f"[ENV] {env_info}")
    return env_info


def _multiply_matrix(device: str, matrix_size: int = 2048) -> float:
    """
    multiply_matrix defines a simple compute test for testing the GPUs.
    It creates a synthetic dataset with linear separability.
    The model should learn fast.
    :param device: str - device to use for training
    :param matrix_size: int - size of the matrix
    :return: float - result of matrix multiplication
    """
    a = torch.randn(matrix_size, matrix_size, device=device)
    logger.debug(f"[MUL] a.shape: {a.shape}")
    b = torch.randn(matrix_size, matrix_size, device=device)
    logger.debug(f"[MUL] b.shape: {b.shape}")
    c = a @ b
    logger.debug(f"[MUL] c.shape: {c.shape}")
    return c.mean().item()


def compute_test_gpu(device: str, 
                     matrix_size: int = 2048, 
                     iterations: int = 20) -> Dict[str, Any]:
    """
    compute_test_gpu defines a simple compute test for testing the GPUs.
    It creates a synthetic dataset with linear separability.
    The model should learn fast.
    :param device: str - device to use for training
    :param matrix_size: int = 2048 - size of the matrix
    :param iterations: int = 20 - number of iterations
    :return: Dict[str, Any] - dictionary of information about the training
    """
    torch.cuda.set_device(device)
    torch.cuda.synchronize()
    start = time()
    
    # create random matrices and perform matrix multiplication to measure 
    # compute performance
    with torch.no_grad():
        # This loop performs matrix multiplication and computes the mean of the 
        # resulting tensor for each iteration.
        # This is done to measure the compute performance of the GPU.
        # The reason for performing matrix multiplication and computing the mean 
        # is because it is a computationally intensive operation.
        for _ in range(iterations):
            _ = _multiply_matrix(device, matrix_size)
    # synchronize GPU
    torch.cuda.synchronize()
    total_time = time() - start
    # ~2*N^3 FLOPs per matmul (multiplication and summation) * iters
    gigaflops = (2.0 * (matrix_size**3) * iterations) / total_time / 1e9
    logger.info(f"[COMPUTE] {device}: {gigaflops:.1f} GFLOP/s"
                 f"{iterations}x{matrix_size}x{matrix_size}")
    output = {
        "device": device, 
        "GFLOPpersec": gigaflops, 
        "time": total_time
        }
    return output


def training_smoke(device:str="cpu", epochs:int=3, batch:int=256) -> Dict[str, Any]:
    """
    training_smoke defines a simple training loop for testing the GPUs.
    It creates a synthetic dataset and model with linear separability.
    The model should learn fast.
    :param device: str - device to use for training
    :param epochs: int - number of epochs to train for
    :param batch: int - batch size for training
    :return: Dict[str, Any] - dictionary of information about the training
    """
    # Initialize the dataset and data loader for training
    data_set = ToyDataset(device=device)
    data_loader = DataLoader(data_set, batch_size=batch, shuffle=True)
    # Initialize the model, optimizer, criterion for training
    model = TinyMLP(input_features=data_set.d, k_classes=data_set.k).to(device)
    # LR tensor is required for AdamW optimizer (default is float)
    lr = 3e-3
    logger.debug(f"[TRAIN] lr: {lr}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    logger.debug(f"[TRAIN] optimizer: {optimizer}")
    criterion = nn.CrossEntropyLoss()
    logger.debug(f"[TRAIN] criterion: {criterion}")
    losses = []

    start = time()
    logger.debug(f"[TRAIN] start: {start}")
    # Iterate over the specified number of epochs:
    for epoch in range(epochs):
        # Reset the epoch-wise loss
        epoch_loss = 0.0
        # Iterate over the batches in the data loader
        for batch_x, batch_y in data_loader:
            # Clear the gradients for the model parameters
            optimizer.zero_grad(set_to_none=True)    # (set_to_none=True instead of setting to zero)
            # Compute the logits for the input batch
            logits = model(batch_x)
            # Compute the loss for the predicted logits and ground truth labels
            loss = criterion(logits, batch_y)
            # Backpropagate the loss to compute the gradients
            loss.backward()
            # Update the model parameters using gradient descent
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
        # Compute the average loss for the current epoch
        losses.append(epoch_loss / len(data_set))
        # Log the epoch number, loss value, and time elapsed
        loss_last = losses[-1]
        logger.debug(f"[TRAIN] loss_last: {loss_last}")
        loss_first = losses[0]
        logger.debug(f"[TRAIN] loss_first: {loss_first}")
        next_epoch = epoch + 1
        logger.debug(f"[TRAIN] next_epoch: {next_epoch}")
        logger.debug(f"[TRAIN] epoch {next_epoch}/{epochs}, loss={loss_last:.4f}")
    total_time = time() - start
    # Check if the training loss improved by a certain factor
    ok = loss_last < loss_first * 0.7   # 0.7 means 30% improvement
    logger.debug(f"[TRAIN] ok: {ok}")
    dataset = {
        "start_loss": loss_first, 
        "end_loss": loss_last, 
        "improved": ok, 
        "time": total_time
        }
    logger.debug(f"[TRAIN] result: {dataset}")
    return dataset


def parse_args() -> Namespace:
    ap = ArgumentParser()
    # Quick mode for fast runs / smoke tests
    ap.add_argument("--quick", action="store_true", help="quick run")
    ap.add_argument("--verbose", action="store_true", help="verbose output")
    return ap.parse_args()


def main():
    args = parse_args()
    set_logger_level(INFO if not args.verbose else DEBUG)
    environment = _get_environment()
    failures = []
    have_cuda_devices = environment["cuda_available"] and environment["cuda_device_count"] > 0
    logger.info(f"[ENV] have CUDA devices: {have_cuda_devices}")
    use_gpu = have_cuda_devices
    logger.info(f"[ENV] use GPU: {use_gpu}")

    # If no GPU is available, skip tests successfully (used by CI without GPUs)
    if not use_gpu:
        logger.info("[COMPUTE] GPU tests skipped (no GPU detected)")
        logger.info("[RESULT] SKIPPED (no GPU)")
        exit(0)

    # 1) compute test (for each GPU if available)
    for device in range(environment["cuda_device_count"]):
        try:
            logger.debug(f"[COMPUTE] compute_test_gpu[cuda:{device}]")
            if args.quick:
                matrix_size, iterations = 2048, 10
            else:
                matrix_size, iterations = 3072, 20
            logger.debug(f"[COMPUTE] compute_test_gpu[cuda:{device}]: "
                         f"matrix_size={matrix_size}, iterations={iterations}")
            compute_test_gpu(f"cuda:{device}", 
                             matrix_size=matrix_size, 
                             iterations=iterations)
        except Exception as err:
            failures.append(f"compute[cuda: {device}]: {err}")

    # 2) training smoke (on device0)
    try:
        device0 = "cuda:0"
        logger.debug(f"[TRAIN] device0: {device0}")
        epochs = 2 if args.quick else 3
        logger.debug(f"[TRAIN] epochs: {epochs}")
        train_result = training_smoke(device=device0, epochs=epochs)
        logger.info(f"[TRAIN] result: {train_result}")
        if not train_result["improved"]:
            failures.append("training did not improve loss enough")
    except Exception as err:
        failures.append(f"training: {err}")

    # 3) DDP/NCCL reminder
    logger.info("[DDP] For inter-GPU/inter-node test run ddp_tests.py "
                 "via torchrun (see sbatch/*.sbatch).")

    if failures:
        logger.error(f"[RESULT] FAILURE: {failures}")
        exit(2)
    logger.info("[RESULT] SUCCESS")
    exit(0)


if __name__ == "__main__":
    main()
