# 6. Multi-GPU training

**Code:** [`slm/distributed.py`](../slm/distributed.py), and the DDP parts of [`slm/trainer.py`](../slm/trainer.py)

## What it is

With several GPUs, you can process several batches at once. This code uses
**data parallelism** through PyTorch's DistributedDataParallel (DDP):

```
            ┌── GPU 0: full model copy, batch A ──┐
 launch ────┼── GPU 1: full model copy, batch B ──┼── average gradients ── identical step on every GPU
            └── GPU 2: full model copy, batch C ──┘
```

Each GPU holds a full copy of the model and trains on *different* data. After
the backward pass, the gradients are averaged across all GPUs (an
**all-reduce**). Every copy then takes the same optimizer step, so the copies
stay identical. With 4 GPUs you process 4× the tokens per step.

## How to run it

```bash
# 4 GPUs on one machine
torchrun --nproc_per_node 4 train.py --config configs/medium.json --data data/tinystories_bpe

# 2 machines with 8 GPUs each (run on both, with the first machine's address)
torchrun --nnodes 2 --nproc_per_node 8 --rdzv_endpoint HOST:29500 train.py ...
```

`torchrun` starts one process per GPU and sets environment variables (`RANK`,
`LOCAL_RANK`, `WORLD_SIZE`) telling each process who it is. Without
`torchrun`, the code runs as a single ordinary process. `train_sft.py` works
the same way.

## What the code does

### Setup (`setup_distributed`)

| Name | Meaning |
|---|---|
| `rank` | this process's number among all processes (0, 1, 2, …) |
| `local_rank` | its number on this machine; chooses which GPU it uses |
| `world_size` | total number of processes |

Each process joins a **process group** so they can communicate. The backend is
NCCL on NVIDIA GPUs (fast GPU-to-GPU links) and Gloo on CPUs. Gloo means you
can test multi-process training without GPUs, which is how this code was
tested.

### Different data on each GPU

`train.py` seeds the random generator with `seed + rank`, so each process picks
different random windows. Without that, every GPU would train on the same
batch and the extra GPUs would be wasted.

### Gradient averaging with accumulation

DDP averages gradients during `backward()`. With gradient accumulation, that
would happen on every micro-batch, which is wasted communication. The trainer
sets `require_backward_grad_sync = False` on all but the last micro-batch, so
GPUs only talk once per optimizer step.

### Only one process writes

Every process computes, but only rank 0 (`is_main`) prints, logs metrics and
saves checkpoints. Otherwise you'd get N copies of every message and N
processes fighting over the same file. Validation losses are averaged across
processes first (`average_across_processes`), so the printed number covers
all GPUs' batches.

### The effective batch size

```
tokens per optimizer step = batch_size × block_size × grad_accum × world_size
```

`configs/medium.json` on 4 GPUs: 16 × 1024 × 8 × 4 ≈ 524k tokens per step,
similar to the batch sizes used for GPT-2-scale models. When you change the
number of GPUs, adjust `grad_accum` to keep this roughly constant, so learning
rates that worked still work.

## When data parallelism isn't enough

DDP needs the whole model, its gradients and the optimizer's state to fit on
*each* GPU. In float32 with Adam, that's about 16 bytes per parameter: roughly
1.3 GB for the 80M-parameter `medium` model, which is easy. Beyond a few
billion parameters, other strategies split the model itself across GPUs:

* **FSDP / ZeRO**: each GPU stores a slice of the weights, gradients and
  optimizer state, gathering what it needs layer by layer.
* **Tensor parallelism**: split individual matrices across GPUs.
* **Pipeline parallelism**: put different layers on different GPUs.

These aren't needed at this project's sizes, but they're the next step.

## Common problems

* **Hangs at startup**: processes can't reach each other. Check ports and the
  rendezvous address; `--standalone` works on a single machine.
* **Out of memory on GPU 0 only**: something is creating tensors on the
  default GPU. Each process must use `cuda:{local_rank}`, as `setup_distributed` does.
* **Socket warnings on CPU** ("Address family not supported"): IPv6 lookups
  failing; harmless.

## Try it (no GPUs needed)

```bash
torchrun --standalone --nproc_per_node 2 train.py --config configs/tiny.json \
    --data data/shakespeare_char --max-iters 40 --eval-interval 20
```

You'll see `device cpu x 2`: two processes training one model together.
