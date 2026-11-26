"""A small decoder-only transformer (GPT-style) language model.

The model reads a sequence of token ids and predicts, at every position,
a probability distribution over the next token.

    token ids ─► embeddings ─► N x Block ─► LayerNorm ─► logits over vocab

Each Block is:   x = x + Attention(LayerNorm(x))
                 x = x + MLP(LayerNorm(x))
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 65  # number of distinct tokens
    block_size: int = 256  # longest sequence the model can read (context length)
    n_layer: int = 6  # number of transformer blocks
    n_head: int = 6  # attention heads per block
    n_embd: int = 384  # width of every hidden vector
    dropout: float = 0.1  # set to 0.0 for RL and inference
    bias: bool = False  # whether Linear/LayerNorm layers have a bias term


class CausalSelfAttention(nn.Module):
    """Each position looks at itself and earlier positions, never later ones."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = config.n_head
        self.dropout = config.dropout
        # One matrix computes queries, keys and values together.
        self.to_qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.out_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        batch, seq_len, width = x.shape
        head_dim = width // self.n_head

        queries, keys, values = self.to_qkv(x).split(width, dim=2)

        # Split the width into heads: (batch, seq, width) -> (batch, heads, seq, head_dim)
        def split_heads(t):
            return t.view(batch, seq_len, self.n_head, head_dim).transpose(1, 2)

        queries, keys, values = split_heads(queries), split_heads(keys), split_heads(values)

        # softmax(Q Kᵀ / sqrt(d)) V with a causal mask, using PyTorch's fused kernel.
        attended = F.scaled_dot_product_attention(
            queries, keys, values,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )

        # Merge heads back: (batch, heads, seq, head_dim) -> (batch, seq, width)
        attended = attended.transpose(1, 2).contiguous().view(batch, seq_len, width)
        return self.out_dropout(self.out_proj(attended))


class MLP(nn.Module):
    """A two-layer feed-forward network applied to each position separately."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.expand = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.out_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.out_proj(F.gelu(self.expand(x))))


class Block(nn.Module):
    """One transformer block: attention then MLP, each with a residual connection."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attn_norm = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.mlp_norm = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.embed_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.final_norm = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: the input embedding and the output layer share one matrix.
        # This saves parameters and usually trains better for small models.
        self.lm_head.weight = self.token_embedding.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        # GPT-2 trick: shrink the layers that write into the residual stream,
        # so the stream doesn't grow as more blocks are stacked.
        residual_std = 0.02 / math.sqrt(2 * self.config.n_layer)
        for name, param in self.named_parameters():
            if name.endswith("out_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=residual_std)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def hidden_states(self, token_ids):
        """Run the transformer and return the final hidden vector at every position.

        token_ids: (batch, seq_len) -> returns (batch, seq_len, n_embd)
        """
        seq_len = token_ids.shape[1]
        assert seq_len <= self.config.block_size, (
            f"sequence of length {seq_len} is longer than block_size {self.config.block_size}"
        )
        positions = torch.arange(seq_len, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        x = self.embed_dropout(x)
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)

    def forward(self, token_ids, targets=None):
        """Return next-token logits at every position, plus the loss if targets are given.

        token_ids: (batch, seq_len)
        targets:   (batch, seq_len), the token that should come after each position
        """
        logits = self.lm_head(self.hidden_states(token_ids))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def configure_optimizer(self, learning_rate, weight_decay, betas=(0.9, 0.95)):
        """AdamW, with weight decay on weight matrices only (not biases or norms)."""
        matrices = [p for p in self.parameters() if p.requires_grad and p.dim() >= 2]
        vectors = [p for p in self.parameters() if p.requires_grad and p.dim() < 2]
        groups = [
            {"params": matrices, "weight_decay": weight_decay},
            {"params": vectors, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas)

    @torch.no_grad()
    def generate(self, token_ids, max_new_tokens, temperature=1.0, top_k=None):
        """Extend each sequence by sampling one token at a time.

        temperature < 1 makes output more predictable; > 1 more random.
        top_k keeps only the k most likely tokens at each step.
        """
        for _ in range(max_new_tokens):
            context = token_ids[:, -self.config.block_size:]  # crop to the context window
            logits, _ = self(context)
            next_logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                kth_best = torch.topk(next_logits, min(top_k, next_logits.size(-1))).values[:, [-1]]
                next_logits[next_logits < kth_best] = -float("inf")
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_ids = torch.cat([token_ids, next_token], dim=1)
        return token_ids
