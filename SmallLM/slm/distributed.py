"""Multi-GPU training with DistributedDataParallel (DDP).

Launch with torchrun, one process per GPU:

    torchrun --nproc_per_node 4 train.py --config configs/small.json

Every process holds a full copy of the model and trains on different batches.
After each backward pass, DDP averages the gradients across processes, so all
copies take the same optimizer step and stay identical. Only process 0
("main") prints, logs and saves checkpoints.

Run without torchrun and everything here quietly becomes single-process.
"""

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class DistributedContext:
    rank: int = 0  # this process's index among all processes
    local_rank: int = 0  # this process's index on this machine (picks the GPU)
    world_size: int = 1  # total number of processes
    device: str = "cpu"

    @property
    def is_main(self):
        return self.rank == 0

    @property
    def enabled(self):
        return self.world_size > 1


def setup_distributed(device):
    """Join the process group if launched by torchrun; otherwise run alone on `device`."""
    if "RANK" not in os.environ:
        return DistributedContext(device=device)

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if device.startswith("cuda"):
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        backend = "gloo"  # CPU backend; useful for testing DDP without GPUs
    dist.init_process_group(backend=backend)
    return DistributedContext(rank, local_rank, world_size, device)


def average_across_processes(value, context):
    """Average a Python number over all processes (e.g. validation loss)."""
    if not context.enabled:
        return value
    tensor = torch.tensor(value, device=context.device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.item() / context.world_size


def cleanup_distributed(context):
    if context.enabled:
        dist.destroy_process_group()
