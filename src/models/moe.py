import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def log_mean(x, dim):
    return torch.logsumexp(x, dim=dim) - torch.log(
        torch.tensor(x.shape[dim], dtype=torch.float32)
    )


def entropy_reg(logits: torch.Tensor, mean_over_batch: bool = True):
    """Entropy regularization for the router."""

    entropy_l = lambda l: -(l * l.exp()).sum(-1)
    # softmax over experts
    # logits: [batch_size * sequence_length, num_experts]
    logprobs = F.log_softmax(logits, dim=-1)
    if mean_over_batch:
        # take mean probability over batch
        logprobs = log_mean(logprobs, 0)

    return -entropy_l(logprobs).mean()


# two losses below are adapted from
# https://github.com/google/flaxformer/blob/b725bd2a51d70e866d819c92de166fbf24425e6a/flaxformer/architectures/moe/routing.py
def load_balancing_loss_(router_probs, expert_indices) -> float:
    """Computes auxiliary load balancing loss as in Switch Transformer.

    See Switch Transformer (https://arxiv.org/abs/2101.03961). This function
    implements the loss function presented in equations (4) - (6). It aims to
    penalize those cases where the routing between experts is unbalanced.

    Args:
      router_probs: Probability assigned to each expert per token. Shape:
        <float32>[batch_size * sequence_length, num_experts].
      expert_indices: <int>[batch_size * sequence_length, num_selected_experts]
        indices identifying the top num_selected_experts for a given token.

    Returns:
      The auxiliary loss.
    """
    # num_token = batch_size * sequence_length
    num_token, num_experts = router_probs.shape

    # Shape: [batch_size * sequence_length, num_selected_experts, num_experts].
    expert_mask = F.one_hot(expert_indices, num_experts)
    # For a given token, determine if it was routed to a given expert.
    # Shape: [batch_size * sequence_length, num_experts]
    expert_mask, _ = torch.max(expert_mask, dim=-2)

    # shape [num_experts]
    tokens_per_expert = torch.mean(expert_mask, dim=0, dtype=torch.float32)
    # shape [num_experts]
    router_prob_per_expert = torch.mean(router_probs, dtype=torch.float32, dim=0)
    return (
        torch.mean(
            tokens_per_expert * router_prob_per_expert,
            dtype=torch.float32,
        )
        * num_experts
    )


def load_balancing_loss(logits, expert_indices) -> float:
    """Computes auxiliary load balancing loss as in Switch Transformer.

    See Switch Transformer (https://arxiv.org/abs/2101.03961). This function
    implements the loss function presented in equations (4) - (6). It aims to
    penalize those cases where the routing between experts is unbalanced.

    Args:
      logits: logits assigned to each expert per token. Shape:
        <float32>[batch_size * sequence_length, num_experts].
      expert_indices: <int>[batch_size * sequence_length, num_selected_experts]
        indices identifying the top num_selected_experts for a given token.

    Returns:
      The auxiliary loss.
    """
    # num_token = batch_size * sequence_length
    num_token, num_experts = logits.shape

    # Shape: [batch_size * sequence_length, num_selected_experts, num_experts].
    expert_mask = F.one_hot(expert_indices, num_experts)
    # For a given token, determine if it was routed to a given expert.
    # Shape: [batch_size * sequence_length, num_experts]
    expert_mask, _ = torch.max(expert_mask, dim=-2)

    # shape [num_experts]
    tokens_per_expert = torch.mean(expert_mask, dim=0, dtype=torch.float32)

    # compute router probability per expert in log space for numerical stability
    logprobs = F.log_softmax(logits, dim=-1)
    # take mean probability over batch
    # shape [num_experts]
    logprobs = log_mean(logprobs, dim=0)
    router_prob_per_expert = torch.exp(logprobs)
    return (
        torch.mean(  # mean over experts
            tokens_per_expert * router_prob_per_expert,
            dtype=torch.float32,
        )
        * num_experts
    )


def router_z_loss(router_logits) -> float:
    """Compute router z-loss.

     The router z-loss was introduced in Designing Effective Sparse Expert Models
     (https://arxiv.org/abs/2202.08906). It encourages router logits to remain
     small in an effort to improve stability.

    Args:
      router_logits: <float>[batch_size * sequence_length, num_experts]
        router logits

    Returns:
      Scalar router z-loss.
    """
    num_tokens, _ = router_logits.shape
    log_z = torch.logsumexp(router_logits, dim=-1)
    z_loss = log_z**2
    return torch.sum(z_loss, dtype=torch.float32) / (num_tokens)


class MoE(nn.Module):
    def __init__(self, config, mlp):
        super().__init__()
        assert config.moe_num_experts > 0
        self.experts = nn.ModuleList(
            [mlp(config=config) for _ in range(config.moe_num_experts)]
        )
        self.n_shared_experts = config.moe_num_shared_experts
        self.router = nn.Linear(
            config.n_embd, config.moe_num_experts - self.n_shared_experts, bias=False
        )
        self.top_k = config.moe_num_experts_per_tok
        self.softmax_order = config.moe_softmax_order

    def forward(self, inputs: torch.Tensor):
        # [batch_size * sequence_length, n_embd]
        inputs_squashed = inputs.view(-1, inputs.shape[-1])
        # [batch_size * sequence_length, num_experts]
        router_logits = self.router(inputs_squashed)

        # note that selected experts will be the same for all orders:
        # softmax doesnt change top-k, but the weights are different
        if self.softmax_order == "softmax_topk":
            all_probs = F.softmax(router_logits, dim=1, dtype=torch.float32)
            weights, selected_experts = torch.topk(all_probs, self.top_k)
        elif self.softmax_order == "topk_softmax":
            weights, selected_experts = torch.topk(router_logits, self.top_k)
            weights = F.softmax(weights, dim=-1, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown softmax_order: {self.softmax_order}")

        results = torch.zeros_like(inputs_squashed)
        for i, expert in enumerate(self.experts):
            if i < self.n_shared_experts:
                # always activate shared experts
                output, _ = expert(inputs_squashed)
                results += output
            else:
                batch_idx, nth_expert = torch.where(
                    selected_experts == i - self.n_shared_experts
                )
                output, _ = expert(inputs_squashed[batch_idx])
                results[batch_idx] += weights[batch_idx, nth_expert, None] * output
        return results.view_as(inputs), {
            "router_logits": router_logits,
            "selected_experts": selected_experts,
        }


class ExpertChoiceMoE(nn.Module):
    def __init__(self, config, mlp):
        super().__init__()
        assert config.moe_num_experts > 0
        self.n_experts = config.moe_num_experts
        self.experts = nn.ModuleList(
            [mlp(config=config) for _ in range(config.moe_num_experts)]
        )
        self.router = nn.Linear(config.n_embd, config.moe_num_experts, bias=False)
        self.capacity_factor = config.capacity_factor
        self.softmax_order = config.moe_softmax_order
        self.top_k = int(
            self.capacity_factor
            * config.batch_size
            * config.sequence_length
            / config.moe_num_experts
        )

    def forward(self, inputs: torch.Tensor):
        # [batch_size * sequence_length, n_embd]
        inputs_squashed = inputs.view(-1, inputs.shape[-1])
        num_tokens = inputs_squashed.shape[0]
        top_k = min(self.top_k, int(self.capacity_factor * num_tokens / self.n_experts))
        # [batch_size * sequence_length, num_experts]
        router_logits = self.router(inputs_squashed)

        # note that selected experts will be the same for all orders:
        # softmax doesnt change top-k, but the weights are different
        if self.softmax_order == "softmax_topk":
            all_probs = F.softmax(router_logits, dim=1, dtype=torch.float32)
            # selection over tokens!
            # weights and selected tokens: [num_experts, top_k]
            weights, selected_tokens = torch.topk(all_probs.T, top_k)
        elif self.softmax_order == "topk_softmax":
            # weights and selected tokens: [num_experts, top_k]
            weights, selected_tokens = torch.topk(router_logits.T, top_k)
            weights = F.softmax(weights, dim=0, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown softmax_order: {self.softmax_order}")

        """ this is the full parallel version with einsum """
        # [num_experts, top_k, num_tokens]
        # P = F.one_hot(selected_tokens, num_tokens).type_as(inputs_squashed)
        # # [num_experts, top_k, n_embd]
        # x_in = torch.matmul(P, inputs_squashed)
        # # [num_experts, num_tokens, n_embd]
        # experts_out = torch.stack(
        #     [expert(x)[0] for expert, x in zip(self.experts, x_in)], dim=0
        # )
        # results = torch.einsum("ijl,ij,ijd->ld", P, weights, experts_out)

        """ this is the loop version """
        # need to loop through experts because of memory growing too large
        # when doing everything in parallel?
        results = torch.zeros_like(inputs_squashed)
        for i, expert in enumerate(self.experts):
            # [top_k]
            batch_idx = selected_tokens[i]
            # [top_k, n_embd]
            output, _ = expert(inputs_squashed[batch_idx])
            results[batch_idx] += weights[i, :, None] * output
        return results.view_as(inputs), {
            "router_logits": router_logits,
            "selected_experts": selected_tokens,
        }
