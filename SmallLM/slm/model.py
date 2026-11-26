"""A small decoder-only transformer language model, in the style of Llama.

The model reads a sequence of token ids and predicts, at every position,
a probability distribution over the next token.

    token ids ─► embedding ─► N x Block ─► RMSNorm ─► logits over vocab

Each Block is:   x = x + Attention(RMSNorm(x))
                 x = x + FeedForward(RMSNorm(x))

Compared with the original GPT-2 design, it uses the components most modern
open models use:

    RMSNorm            a simpler, cheaper LayerNorm (no mean subtraction, no bias)
    Rotary embeddings  positions are encoded by rotating queries and keys,
    (RoPE)             instead of adding a learned position vector
    SwiGLU             a gated feed-forward layer that trains better than GELU
    Grouped-query      several query heads share one key/value head, which
    attention (GQA)    shrinks the KV cache
    KV cache           during generation, keys and values of earlier tokens are
                       kept, so each new token costs one position, not the whole
                       sequence
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class GPTConfig:
    vocab_size: int = 65  # number of distinct tokens
    block_size: int = 256  # longest sequence the model can read (context length)
    n_layer: int = 6  # number of transformer blocks
    n_head: int = 6  # query heads per block
    n_kv_head: int = 0  # key/value heads; 0 means "same as n_head" (no grouping)
    n_embd: int = 384  # width of every hidden vector
    dropout: float = 0.0  # 0.0 is usual for small models with plenty of data
    rope_theta: float = 10000.0  # base frequency of the rotary embeddings

    def __post_init__(self):
        if self.n_kv_head == 0:
            self.n_kv_head = self.n_head
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        assert self.n_head % self.n_kv_head == 0, "n_head must be divisible by n_kv_head"

    @property
    def head_dim(self):
        return self.n_embd // self.n_head


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #


class RMSNorm(nn.Module):
    """Scale each vector to unit root-mean-square, then by a learned gain."""

    def __init__(self, width, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(width))

    def forward(self, x):
        rms = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x.float() * rms).type_as(x) * self.gain


def rotary_tables(head_dim, max_len, theta):
    """Precompute the cos/sin tables used by rotary position embeddings.

    Each pair of features is rotated by an angle that grows with position, at a
    different speed per pair. The dot product of a rotated query and key then
    depends only on how far apart they are, which is what attention needs.
    Returns two tensors of shape (max_len, head_dim / 2).
    """
    speeds = 1.0 / theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
    angles = torch.outer(torch.arange(max_len).float(), speeds)
    return angles.cos(), angles.sin()


def apply_rotary(x, cos, sin):
    """Rotate feature pairs of x (batch, heads, seq, head_dim) by the given angles."""
    first, second = x.float().chunk(2, dim=-1)
    cos, sin = cos[None, None], sin[None, None]  # broadcast over batch and heads
    rotated = torch.cat([first * cos - second * sin, first * sin + second * cos], dim=-1)
    return rotated.type_as(x)


class KVCache:
    """Keys and values of tokens already processed, one entry per layer."""

    def __init__(self, n_layer):
        self.keys = [None] * n_layer
        self.values = [None] * n_layer

    @property
    def length(self):
        """How many positions are stored."""
        return 0 if self.keys[0] is None else self.keys[0].shape[2]

    def append(self, layer, keys, values):
        """Add new keys/values for a layer; return everything stored for it."""
        if self.keys[layer] is not None:
            keys = torch.cat([self.keys[layer], keys], dim=2)
            values = torch.cat([self.values[layer], values], dim=2)
        self.keys[layer], self.values[layer] = keys, values
        return keys, values


class Attention(nn.Module):
    """Causal self-attention with rotary embeddings and grouped-query heads."""

    def __init__(self, config: GPTConfig, layer_index):
        super().__init__()
        self.layer_index = layer_index
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.head_dim
        self.dropout = config.dropout
        self.to_queries = nn.Linear(config.n_embd, config.n_head * config.head_dim, bias=False)
        self.to_keys = nn.Linear(config.n_embd, config.n_kv_head * config.head_dim, bias=False)
        self.to_values = nn.Linear(config.n_embd, config.n_kv_head * config.head_dim, bias=False)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x, cos, sin, cache=None):
        batch, seq_len, _ = x.shape

        # (batch, seq, heads * head_dim) -> (batch, heads, seq, head_dim)
        def split_heads(t, heads):
            return t.view(batch, seq_len, heads, self.head_dim).transpose(1, 2)

        queries = split_heads(self.to_queries(x), self.n_head)
        keys = split_heads(self.to_keys(x), self.n_kv_head)
        values = split_heads(self.to_values(x), self.n_kv_head)

        queries = apply_rotary(queries, cos, sin)
        keys = apply_rotary(keys, cos, sin)

        # During generation, attend to the cached earlier tokens too.
        # (Check this layer's own entry: earlier layers have already written theirs.)
        is_first_pass = cache is None or cache.keys[self.layer_index] is None
        if cache is not None:
            keys, values = cache.append(self.layer_index, keys, values)

        # Grouped-query attention: each key/value head serves several query heads.
        group = self.n_head // self.n_kv_head
        keys = keys.repeat_interleave(group, dim=1)
        values = values.repeat_interleave(group, dim=1)

        # On the first pass every position must not see later ones (causal mask).
        # When generating one new token with a cache, it may see everything cached.
        attended = F.scaled_dot_product_attention(
            queries, keys, values,
            is_causal=is_first_pass and seq_len > 1,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.out_proj(attended)


class FeedForward(nn.Module):
    """SwiGLU: out = W_out( silu(W_gate x) * (W_up x) )."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        # 8/3 x width keeps the parameter count equal to a 4x GELU MLP; round up to 64.
        hidden = 64 * math.ceil(8 * config.n_embd / 3 / 64)
        self.gate = nn.Linear(config.n_embd, hidden, bias=False)
        self.up = nn.Linear(config.n_embd, hidden, bias=False)
        self.out_proj = nn.Linear(hidden, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.out_proj(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    """One transformer block: attention then feed-forward, each with a residual connection."""

    def __init__(self, config: GPTConfig, layer_index):
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd)
        self.attn = Attention(config, layer_index)
        self.ffn_norm = RMSNorm(config.n_embd)
        self.ffn = FeedForward(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, cos, sin, cache=None):
        x = x + self.dropout(self.attn(self.attn_norm(x), cos, sin, cache))
        x = x + self.ffn(self.ffn_norm(x))
        return x


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config, i) for i in range(config.n_layer)])
        self.final_norm = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: the input embedding and the output layer share one matrix.
        # This saves parameters and usually trains better for small models.
        self.lm_head.weight = self.token_embedding.weight

        cos, sin = rotary_tables(config.head_dim, config.block_size, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        # Trade compute for memory: recompute activations in the backward pass.
        self.activation_checkpointing = False

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # GPT-2 trick: shrink the layers that write into the residual stream,
        # so the stream doesn't grow as more blocks are stacked.
        residual_std = 0.02 / math.sqrt(2 * self.config.n_layer)
        for name, param in self.named_parameters():
            if name.endswith("out_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=residual_std)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def hidden_states(self, token_ids, cache=None):
        """Run the transformer and return the final hidden vector at every position.

        token_ids: (batch, seq_len) -> returns (batch, seq_len, n_embd)
        cache:     a KVCache to read earlier tokens from and append these to
        """
        start = cache.length if cache is not None else 0
        end = start + token_ids.shape[1]
        assert end <= self.config.block_size, (
            f"sequence of length {end} is longer than block_size {self.config.block_size}"
        )
        cos, sin = self.rope_cos[start:end], self.rope_sin[start:end]

        x = self.token_embedding(token_ids)
        for block in self.blocks:
            if self.activation_checkpointing and self.training:
                x = checkpoint(block, x, cos, sin, use_reentrant=False)
            else:
                x = block(x, cos, sin, cache)
        return self.final_norm(x)

    def forward(self, token_ids, targets=None, cache=None):
        """Return next-token logits at every position, plus the loss if targets are given.

        token_ids: (batch, seq_len)
        targets:   (batch, seq_len), the token that should follow each position.
                   Positions with target -100 are ignored (used to train only on
                   answers in supervised fine-tuning).
        """
        logits = self.lm_head(self.hidden_states(token_ids, cache))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1), ignore_index=-100
        )
        return logits, loss

    def configure_optimizer(self, learning_rate, weight_decay, betas=(0.9, 0.95)):
        """AdamW, with weight decay on weight matrices only (not norms)."""
        matrices = [p for p in self.parameters() if p.requires_grad and p.dim() >= 2]
        vectors = [p for p in self.parameters() if p.requires_grad and p.dim() < 2]
        groups = [
            {"params": matrices, "weight_decay": weight_decay},
            {"params": vectors, "weight_decay": 0.0},
        ]
        fused = torch.cuda.is_available() and next(self.parameters()).is_cuda
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, fused=fused)

    @torch.no_grad()
    def generate(self, token_ids, max_new_tokens, temperature=1.0, top_k=None, eos_id=None):
        """Extend each sequence by sampling one token at a time.

        temperature  0 picks the most likely token (greedy); < 1 is more predictable;
                     > 1 is more random.
        top_k        keep only the k most likely tokens at each step.
        eos_id       stop a sequence once it produces this token; sequences that
                     have stopped are padded with it until all have stopped.
        """
        block_size = self.config.block_size
        cache = KVCache(self.config.n_layer)
        logits, _ = self(token_ids[:, -block_size:], cache=cache)
        finished = torch.zeros(token_ids.shape[0], dtype=torch.bool, device=token_ids.device)

        for _ in range(max_new_tokens):
            next_token = sample_next_token(logits[:, -1, :], temperature, top_k)
            if eos_id is not None:
                next_token = torch.where(finished, torch.full_like(next_token, eos_id), next_token)
                finished |= next_token == eos_id
            token_ids = torch.cat([token_ids, next_token[:, None]], dim=1)
            if finished.all():
                break

            if cache.length < block_size:
                # Fast path: feed only the new token; earlier ones come from the cache.
                logits, _ = self(next_token[:, None], cache=cache)
            else:
                # The context window is full: slide it and start a fresh cache.
                cache = KVCache(self.config.n_layer)
                logits, _ = self(token_ids[:, -block_size:], cache=cache)
        return token_ids


def sample_next_token(logits, temperature=1.0, top_k=None):
    """Pick one token id per row from next-token logits (batch, vocab)."""
    if temperature == 0:
        return logits.argmax(dim=-1)
    logits = logits / temperature
    if top_k is not None:
        kth_best = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
        logits = logits.masked_fill(logits < kth_best, -float("inf"))
    probs = F.softmax(logits.float(), dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
