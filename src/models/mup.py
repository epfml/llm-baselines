"""
Full definition of a GPT Language Model __with mup parametrization__.
The changes to the normal GPT-2 model are marked with 'mup change here!'.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
3) https://github.com/microsoft/mup/blob/main/examples/Transformer/model.py
"""

import math

import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F

from models.base import LayerNorm
from models.moe import (ExpertChoiceMoE, MoE, entropy_reg, load_balancing_loss,
                        router_z_loss)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(
                    torch.ones(config.sequence_length, config.sequence_length)
                ).view(1, 1, config.sequence_length, config.sequence_length),
            )

    def forward(self, x):
        # batch size, sequence length, embedding dimensionality (n_embd)
        (
            B,
            T,
            C,
        ) = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # (B, T, nh, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout,
                is_causal=True,
                scale=1 / q.size(-1),  # mup change here!
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1)))  # mup change here!
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config, exp_factor=1.0):
        super().__init__()
        self.dim_exp_factor = exp_factor * 4

        self.c_fc = nn.Linear(
            config.n_embd, int(self.dim_exp_factor * config.n_embd), bias=config.bias
        )
        self.c_proj = nn.Linear(
            int(self.dim_exp_factor * config.n_embd), config.n_embd, bias=config.bias
        )
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x, {}


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.parallel = config.parallel_block
        if not self.parallel:
            self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        if config.moe:
            if config.moe_routing == "standard_gating":
                self.mlp = MoE(config, MLP)
            elif config.moe_routing == "expert_choice":
                self.mlp = ExpertChoiceMoE(config, MLP)
            elif config.moe_routing == "soft_moe":
                self.mlp = SoftMoE(config, MLP)
            elif config.moe_routing == "tree":
                self.mlp = TreeRouter(config, MLP)
            else:
                raise ValueError(f"Unknown routing: {config.routing}")
        else:
            self.mlp = MLP(config)

        self.scale_depth = config.scale_depth  # mup change here!
        self.n_layer = config.n_layer  # mup change here!

    def forward(self, x, *args, **kwargs):
        if self.parallel:
            # from GPT-J 6B https://github.com/kingoflolz/mesh-transformer-jax/blob/f8315e3003033b23f21d78361b288953064e0e76/mesh_transformer/layers.py#L299
            x_ln = self.ln_1(x, *args, **kwargs)
            x_attn = self.attn(x_ln)
            x_ffn, logits_and_experts = self.mlp(x_ln)
            x = x + x_attn + x_ffn
        else:
            x_ = self.attn(self.ln_1(x, *args, **kwargs))
            # mup change here!
            x = x + x_ * self.scale_depth / math.sqrt(self.n_layer)
            x_, logits_and_experts = self.mlp(self.ln_2(x, *args, **kwargs))
            # mup change here!
            x = x + x_ * self.scale_depth / math.sqrt(self.n_layer)
        return x, logits_and_experts


class MuPGPTBase(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.sequence_length is not None
        self.config = config
        self.tokenizer = tiktoken.get_encoding("gpt2")

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.sequence_length, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        if not config.untied_embeds:
            self.transformer.wte.weight = (
                self.lm_head.weight
            )  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p,
                    mean=0.0,
                    std=self.config.init_std
                    / math.sqrt(
                        2
                        * config.n_layer
                        * config.n_embd
                        / self.config.scale_base_model
                    ),  # mup change here!
                )
            if pn.endswith("router.weight"):
                # special scaled init to moe router?
                with torch.no_grad():
                    dim = 1 if config.moe_routing == "standard_gating" else 0
                    std = p.std()
                    p.div_(p.sum(dim=dim, keepdim=True))
                    p.mul_(std / p.std())

    def get_router_losses(self, logits, selected_experts, eval=False):
        # logits: (b * seq_len, n_experts)
        # selected_experts: (b * seq_len, topk)
        if eval:  # eval mode, compute all losses
            return {
                "moe_entropy_loss": entropy_reg(logits),
                "moe_aux_loss": load_balancing_loss(logits, selected_experts),
                "moe_z_loss": router_z_loss(logits),
            }
        if self.config.moe_router_loss == "entropy":
            return {
                "moe_entropy_loss": entropy_reg(logits),
            }
        elif self.config.moe_router_loss == "load_balancing_only":
            return {
                "moe_aux_loss": load_balancing_loss(logits, selected_experts),
            }
        elif self.config.moe_router_loss == "load_balancing_z_loss":
            return {
                "moe_aux_loss": load_balancing_loss(logits, selected_experts),
                "moe_z_loss": router_z_loss(logits),
            }
        return {}

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)

    def forward(self, idx, targets=None, get_logits=False, moe=False):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.sequence_length
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.sequence_length}"
        # shape (1, t)
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(
            pos
        )  # position embeddings of shape (1, t, n_embd)
        # mup change here!
        x = self.transformer.drop((tok_emb + pos_emb) * self.config.scale_emb)

        # router logits is a list for each layer's routing, each of shape (b * seq_len, n_experts)
        router_logits = []
        # experts is a list for each layer's selected experts, shape (b * seq_len, topk)
        experts = []

        # forward pass through all the transformer blocks
        for block in self.transformer.h:
            x, logits_and_experts = block(x)
            if len(logits_and_experts) > 0:
                router_logits.append(logits_and_experts["router_logits"])
                experts.append(logits_and_experts["selected_experts"])
        x = self.transformer.ln_f(x)

        # aux_losses is a dict with keys for different auxiliary losses
        aux_losses = {}

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            # mup change here!
            logits = logits / (self.config.n_embd / self.config.scale_base_model)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
            if moe and self.config.moe_routing == "standard_gating":
                # calculate the router losses per layer
                for logit, expert_choice in zip(router_logits, experts):
                    router_losses = self.get_router_losses(
                        logit, expert_choice, eval=not self.training
                    )
                    for k, v in router_losses.items():
                        aux_losses[k] = aux_losses.get(k, 0.0) + v
                        if self.training:
                            loss += (
                                v
                                * getattr(self.config, k + "_factor")
                                / self.config.n_layer
                            )

        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            loss = None
        logits = logits if get_logits else None
        router_logits = (
            torch.stack(router_logits, dim=0) if len(router_logits) > 0 else None
        )
        return {
            "logits": logits,
            "loss": loss,
            "aux_losses": aux_losses,
            "router_logits": router_logits,
        }

    def crop_sequence_length(self, sequence_length):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert sequence_length <= self.config.sequence_length
        self.config.sequence_length = sequence_length
        self.transformer.wpe.weight = nn.Parameter(
            self.transformer.wpe.weight[:sequence_length]
        )
        for block in self.transformer.h:
            block.attn.bias = block.attn.bias[:, :, :sequence_length, :sequence_length]

    def convert_dense_to_sparse(self, state_dict):
        """
        Convert the dense model to sparse model.
        """
        state_to_load = {}
        for k, v in state_dict.items():
            vals = k.split(".")
            print(vals)
            if len(vals) >= 5 and vals[4] == "mlp":
                # for layer i, go from '_orig_mod.transformer.h.i.mlp.c_fc.weight' to
                # '_orig_mod.transformer.h.i.mlp.experts.e.c_fc.weight'
                for e in range(self.config.moe_num_experts):
                    state_to_load[
                        ".".join(vals[1:5] + ["experts", str(e)] + vals[5:])
                    ] = v
                    # add router weight from already initialized weights above
                    state_to_load[".".join(vals[1:5] + ["router", "weight"])] = (
                        self.transformer.h[int(vals[3])].mlp.router.weight
                    )
            else:
                state_to_load[".".join(k.split(".")[1:])] = v
        return state_to_load

    def convert_n_dense_to_sparse(self, state_dicts):
        """
        Convert the dense model to sparse model.
        """
        assert (
            len(state_dicts) == self.config.moe_num_experts
        ), f"len(state_dict)={len(state_dicts)} != {self.config.moe_num_experts}."
        state_to_load = {}
        for e in range(self.config.moe_num_experts):
            state_dict = state_dicts[e]
            for k, v in state_dict.items():
                vals = k.split(".")
                print(vals)
                if len(vals) >= 5 and vals[4] == "mlp":
                    # for layer i, go from '_orig_mod.transformer.h.i.mlp.c_fc.weight' to
                    # '_orig_mod.transformer.h.i.mlp.experts.e.c_fc.weight'
                    state_to_load[
                        ".".join(vals[1:5] + ["experts", str(e)] + vals[5:])
                    ] = v
                    # add router weight from already initialized weights above
                    state_to_load[".".join(vals[1:5] + ["router", "weight"])] = (
                        self.transformer.h[int(vals[3])].mlp.router.weight
                    )
                else:
                    state_to_load[".".join(k.split(".")[1:])] = v
        return state_to_load

    def from_pretrained(
        self,
        model_path,
        from_dense: bool = True,
    ):
        paths = model_path.split(",")
        if len(paths) == 1:
            # TODO: with distributed?
            loaded_state = torch.load(
                str(model_path + "/ckpt.pt"),
                map_location=torch.device(self.config.device),
            )
            state_to_load = loaded_state["model"]

            if self.config.moe and from_dense:
                # load the dense model and convert to sparse
                state_to_load = self.convert_dense_to_sparse(state_to_load)
            else:
                # load the sparse model
                state_to_load = {
                    ".".join(k.split(".")[1:]): v  # drop _orig_mod from keys
                    for k, v in state_to_load.items()
                }
        else:
            loaded_states = []
            for path in paths:
                loaded_state = torch.load(
                    str(path + "/ckpt.pt"),
                    map_location=torch.device(self.config.device),
                )
                loaded_states.append(loaded_state["model"])
            if self.config.moe and from_dense:
                # load the dense model and convert to sparse
                print(f"Loading from {len(paths)} dense models.")
                state_to_load = self.convert_n_dense_to_sparse(loaded_states)
            else:
                raise NotImplementedError("Multiple paths -> load from dense.")
        super().load_state_dict(state_to_load)

    def get_parameter_group_specs(self, config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        # need to do import here to avoid circular import (since llama imports from base here)
        from .utils import BLACKLIST_WEIGHT_MODULES

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, BLACKLIST_WEIGHT_MODULES):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
        # will appear in the no_decay and decay sets respectively after the above.
        # In addition, because named_parameters() doesn't return duplicates, it
        # will only return the first occurence, key'd by 'transformer.wte.weight', below.
        # so let's manually remove 'lm_head.weight' from decay set. This will include
        # this tensor into optimization via transformer.wte.weight only, and not decayed.
        # decay.remove("lm_head.weight") --- we handle this with config.untied_embeds
        if config.untied_embeds:
            decay.remove("lm_head.weight")
            no_decay.add("lm_head.weight")
        else:
            decay.remove("lm_head.weight")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        return [
            {
                "params": sorted(list(decay)),
                "lr": self.config.lr
                / (
                    self.config.n_embd / self.config.scale_base_model
                ),  # mup change here!
            },
            {"params": sorted(list(no_decay)), "weight_decay": 0.0},
        ]

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at sequence_length
            idx_cond = (
                idx
                if idx.size(1) <= self.config.sequence_length
                else idx[:, -self.config.sequence_length :]
            )
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond, get_logits=True)["logits"]
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    @torch.no_grad()
    def generate_from_string(self, in_str, max_new_tokens, temperature=1.0, top_k=None):
        idx = (
            torch.tensor(
                self.tokenizer.encode(in_str, allowed_special={"<|endoftext|>"})
            )
            .view(1, -1)
            .to(self.lm_head.weight.device)
        )
        out_idx = (
            self.generate(idx, max_new_tokens, temperature, top_k)
            .view(-1)
            .to("cpu")
            .numpy()
        )
        return self.tokenizer.decode(out_idx)
