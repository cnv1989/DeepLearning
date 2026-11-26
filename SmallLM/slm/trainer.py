"""The training loop, shared by pretraining (train.py) and fine-tuning (train_sft.py).

The two differ only in where batches come from, so the Trainer takes a
`get_batch(split)` function returning (inputs, targets) and handles everything
else:

    * learning-rate warmup and cosine decay
    * gradient accumulation (simulate a big batch with several small ones)
    * mixed precision (bfloat16 / float16) and gradient clipping
    * multi-GPU training with DDP
    * periodic evaluation, metrics logging and checkpoints:
          <out>/ckpt.pt   the best model so far (lowest validation loss)
          <out>/last.pt   the latest state, for --resume
"""

import math
import os
import time

import torch
from torch.nn.parallel import DistributedDataParallel

from slm.checkpoint import save_checkpoint
from slm.distributed import average_across_processes
from slm.utils import autocast, cosine_lr, pick_dtype, set_learning_rate


class Trainer:
    def __init__(self, model, tokenizer, settings, get_batch, context, logger):
        self.model = model.to(context.device)  # the plain model; used for saving
        self.tokenizer = tokenizer
        self.settings = settings
        self.get_batch = get_batch
        self.context = context
        self.logger = logger

        self.dtype = pick_dtype(settings.dtype, context.device)
        self.optimizer = model.configure_optimizer(settings.lr, settings.weight_decay)
        # float16 needs loss scaling to avoid gradients underflowing to zero.
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.dtype == torch.float16)

        self.step = 0
        self.best_val_loss = float("inf")
        self.last_path = os.path.join(settings.out, "last.pt")
        self.best_path = os.path.join(settings.out, "ckpt.pt")
        if settings.resume and os.path.exists(self.last_path):
            self._resume()

        # The version of the model the training loop calls: maybe compiled, maybe DDP-wrapped.
        self.train_model = torch.compile(self.model) if settings.compile else self.model
        if context.enabled:
            device_ids = [context.local_rank] if context.device.startswith("cuda") else None
            self.train_model = DistributedDataParallel(self.train_model, device_ids=device_ids)

    def print(self, *args):
        if self.context.is_main:
            print(*args, flush=True)

    # ------------------------------------------------------------------ #

    def train(self):
        s = self.settings
        start_time = last_log_time = time.time()
        tokens_since_log = 0

        while True:
            lr = cosine_lr(self.step, s.lr, s.min_lr, s.warmup_iters, s.max_iters)
            set_learning_rate(self.optimizer, lr)

            if self.step % s.eval_interval == 0 or self.step == s.max_iters:
                self.evaluate_and_save(elapsed=time.time() - start_time)
            if self.step == s.max_iters:
                break

            loss, tokens = self.train_step()
            tokens_since_log += tokens * self.context.world_size

            if self.step % s.log_interval == 0:
                now = time.time()
                tokens_per_sec = tokens_since_log / (now - last_log_time)
                self.logger.log(self.step, train_loss=loss, lr=lr, tokens_per_sec=tokens_per_sec)
                last_log_time, tokens_since_log = now, 0
            self.step += 1

        self.print(f"done. best val loss {self.best_val_loss:.4f}, saved to {self.best_path}")

    def train_step(self):
        """One optimizer step, accumulated over `grad_accum` micro-batches.

        Returns (average loss, number of tokens processed on this process).
        """
        s = self.settings
        total_loss, tokens = 0.0, 0
        for micro_step in range(s.grad_accum):
            inputs, targets = self.get_batch("train")
            if self.context.enabled:
                # Only average gradients across GPUs after the last micro-batch.
                self.train_model.require_backward_grad_sync = micro_step == s.grad_accum - 1
            with autocast(self.context.device, self.dtype):
                _, loss = self.train_model(inputs, targets)
            self.scaler.scale(loss / s.grad_accum).backward()
            total_loss += loss.item() / s.grad_accum
            tokens += inputs.numel()

        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), s.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        return total_loss, tokens

    @torch.no_grad()
    def estimate_loss(self, split):
        self.model.eval()
        total = 0.0
        for _ in range(self.settings.eval_iters):
            inputs, targets = self.get_batch(split)
            with autocast(self.context.device, self.dtype):
                _, loss = self.model(inputs, targets)
            total += loss.item()
        self.model.train()
        return average_across_processes(total / self.settings.eval_iters, self.context)

    def evaluate_and_save(self, elapsed):
        train_loss, val_loss = self.estimate_loss("train"), self.estimate_loss("val")
        self.print(f"step {self.step:5d} | train loss {train_loss:.4f} | val loss {val_loss:.4f} "
                   f"| val perplexity {math.exp(min(val_loss, 50)):7.2f} | {elapsed:.0f}s")
        self.logger.log(self.step, eval_train_loss=train_loss, val_loss=val_loss)

        if not self.context.is_main:
            return
        improved = val_loss < self.best_val_loss
        self.best_val_loss = min(self.best_val_loss, val_loss)
        state = dict(optimizer=self.optimizer.state_dict(), step=self.step,
                     best_val_loss=self.best_val_loss, settings=vars(self.settings))
        save_checkpoint(self.last_path, self.model, self.tokenizer, **state)
        if improved:
            save_checkpoint(self.best_path, self.model, self.tokenizer, **state)

    def _resume(self):
        checkpoint = torch.load(self.last_path, map_location=self.context.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.step = checkpoint["step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.print(f"resuming from step {self.step}")
