"""
Llama style Language Model.
References:
1) Llama inference code:
https://github.com/facebookresearch/llama/blob/main/llama/model.py
2) Mistral one file ref:
https://github.com/mistralai/mistral-src/blob/main/one_file_ref.py
3) Llama paper:
https://arxiv.org/pdf/2302.13971.pdf
 
Main differences from GPT2:
* Uses RMSNorm instead of LayerNorm
* Uses a slightly different MLP (SwiGLU)
* rotary embeddings (RoPE)
"""

import math

import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F
from models.base import CausalSelfAttention, GPTBase


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def _reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    freqs_cis: complex - (seq_len, head_dim / 2)
    x: complex - (bsz, seq_len, head_dim / 2)
    """
    ndim = x.ndim
    assert 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), (
        freqs_cis.shape,
        (x.shape[1], x.shape[-1]),
    )
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(q, k, freqs_cis):
    # q, k: (B, T, nh, hs)
    # freq_cis: (T, hs)
    # return: (B, T, nh, hs), (B, T, nh, hs)
    q_ = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k_ = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
    freqs_cis = _reshape_for_broadcast(freqs_cis, q_)
    xq_out = torch.view_as_real(q_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(k_ * freqs_cis).flatten(3)
    return xq_out.type_as(q), xk_out.type_as(k)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        hidden_dim = config.n_embd * 4
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = config.multiple_of * (
            (hidden_dim + config.multiple_of - 1) // config.multiple_of
        )

        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(nn.functional.silu(self.w1(x)) * self.w2(x))


class LlamaAttention(CausalSelfAttention):
    def forward(self, x, freqs_cis_short, freqs_cis_long):
        # batch size, sequence length, embedding dimensionality (n_embd)
        (
            B,
            T,
            C,
        ) = x.size()

        y_short, y_long = None, None

        if self.n_heads_short > 0:
            # Compute q, k, v for short heads
            qk_short = self.qk_short(x).split(self.n_heads_short * self.short_heads_dim, dim=2)
            q_short, k_short = [t.view(B, T, self.n_heads_short, self.short_heads_dim) for t in qk_short]  # NO transpose yet!
            v_short = self.v_short(x).view(B, T, self.n_heads_short, self.head_dim)

            # Apply RoPE before transposing
            q_short, k_short = apply_rotary_emb(q_short, k_short, freqs_cis_short)

            # NOW transpose to [B, num_heads, T, head_dim]
            q_short, k_short = q_short.transpose(1, 2), k_short.transpose(1, 2)
            v_short = v_short.transpose(1, 2)

            # Short-range attention
            mask_short = self.mask_short[:T, :T] if self.context_short < T else None
            if mask_short is not None:
                print(f"Attention short mask: {mask_short[:5,0]}")
                print(f"type mask short: {type(mask_short[0,0])}")
            y_short = F.scaled_dot_product_attention(
                q_short, k_short, v_short, attn_mask=mask_short,
                dropout_p=self.attn_dropout.p, is_causal=mask_short is None
            )

        if self.n_heads_long > 0:
            # Compute q, k, v for long heads
            qk_long = self.qk_long(x).split(self.n_heads_long * self.long_heads_dim, dim=2)
            q_long, k_long = [t.view(B, T, self.n_heads_long, self.long_heads_dim) for t in qk_long]  # NO transpose yet!
            v_long = self.v_long(x).view(B, T, self.n_heads_long, self.head_dim)

            # Apply RoPE before transposing
            q_long, k_long = apply_rotary_emb(q_long, k_long, freqs_cis_long)

            # NOW transpose to [B, num_heads, T, head_dim]
            q_long, k_long = q_long.transpose(1, 2), k_long.transpose(1, 2)
            v_long = v_long.transpose(1, 2)

            # Long-range attention
            mask_long = self.mask_long[:T, :T] if self.context_long < T else None
            if mask_long is not None:
                print(f"Attention long mask: {mask_long[:5,0]}")
                print(f"type mask long: {type(mask_long[0,0])}")
            y_long = F.scaled_dot_product_attention(
                q_long, k_long, v_long, attn_mask=mask_long,
                dropout_p=self.attn_dropout.p, is_causal=mask_long is None
            )


        # Merge outputs from both attentions
        if y_short is not None and y_long is not None:
            y = torch.cat([y_short, y_long], dim=1)
        elif y_short is not None:
            y = y_short
        elif y_long is not None:
            y = y_long
        else:
            raise ValueError("Both short and long heads are disabled!")

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

        # # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # # (B, T, nh, hs)
        # k = k.view(B, T, self.n_head, C // self.n_head)
        # q = q.view(B, T, self.n_head, C // self.n_head)
        # q, k = apply_rotary_emb(q, k, freqs_cis)
        # # (B, nh, T, hs)
        # q, k = q.transpose(1, 2), k.transpose(1, 2)

        # # (B, nh, T, hs)
        # v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # if self.flash:
        #     # efficient attention using Flash Attention CUDA kernels
        #     y = torch.nn.functional.scaled_dot_product_attention(
        #         q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True
        #     )
        # else:
        #     # manual implementation of attention
        #     att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        #     att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        #     att = F.softmax(att, dim=-1)
        #     att = self.attn_dropout(att)
        #     y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # y = (
        #     y.transpose(1, 2).contiguous().view(B, T, C)
        # )  # re-assemble all head outputs side by side

        # # output projection
        # y = self.resid_dropout(self.c_proj(y))
        # return y


class LlamaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd, eps=config.rmsnorm_eps)
        self.attn = LlamaAttention(config)
        self.ln_2 = RMSNorm(config.n_embd, eps=config.rmsnorm_eps)
        self.mlp = LlamaMLP(config)

    def forward(self, x, freqs_cis_short, freqs_cis_long):
        x = x + self.attn(self.ln_1(x), freqs_cis_short, freqs_cis_long)
        x = x + self.mlp(self.ln_2(x))
        return x


class Llama(GPTBase):
    def __init__(self, config):
        super().__init__(config)
        assert config.vocab_size is not None
        assert config.sequence_length is not None
        self.config = config
        self.tokenizer = tiktoken.get_encoding("gpt2")

        # create the token and position embeddings
        self.head_dim = config.n_embd // config.n_head
        # self.freqs_cis = precompute_freqs_cis(self.head_dim, config.sequence_length)

        self.freqs_cis_short = precompute_freqs_cis(config.short_heads_dim, config.sequence_length)
        self.freqs_cis_long = precompute_freqs_cis(config.long_heads_dim, config.sequence_length)

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([LlamaBlock(config) for _ in range(config.n_layer)]),
                ln_f=RMSNorm(config.n_embd, eps=config.rmsnorm_eps),
            )
        )

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = (
            self.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )


    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default)
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def forward(self, idx, targets=None, get_logits=False):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.sequence_length
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.sequence_length}"
        # shape (1, t)
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)

        x = self.transformer.drop(tok_emb)
        # freqs_cis = self.freqs_cis.to(x.device)[pos]
        # freqs_cis_short = self.freqs_cis_short[:t]
        # freqs_cis_long = self.freqs_cis_long[:t]
        freqs_cis_short = self.freqs_cis_short.to(x.device)[pos]
        freqs_cis_long = self.freqs_cis_long.to(x.device)[pos]

        for block_idx, block in enumerate(self.transformer.h):
            x = block(x, freqs_cis_short=freqs_cis_short, freqs_cis_long=freqs_cis_long)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            loss = None

        logits = logits if get_logits else None

        return {
            "logits": logits,
            "loss": loss,
        }
