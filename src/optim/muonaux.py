from typing import Callable, Optional, Dict, Any, List, Union, Tuple
import torch
from torch import nn
from torch.optim.optimizer import Optimizer, ParamsT
from .muon import Muon
from .muon_ema import MuonEMA
import warnings

'''
Wrapper optimizer for using Muon with an Auxilliary Optimizer. 
Currently supports Muon and MuonEMA for main optimizer
but any optimizer should work as an Auxillary optimizer.
Granted that the auxilliary optimizer is provided correctly.
Preferably provide parameters as a parameter_group.
but named_parameters (better) and regular parameters should work too.
'''
class MuonWithAux(Optimizer):
    def __init__(self, 
        params: ParamsT, 
        muon_type: str = "muon",
        muon_kwargs: Optional[dict] = None,
        aux_opt_class  = torch.optim.AdamW,
        aux_kwargs: Optional[dict[str,any]] = None, # whether to use defaults or not
    ):
        if muon_kwargs is None:
            muon_kwargs = {}
        if aux_kwargs is None:
            aux_kwargs = {}

        muon_params = []
        aux_params = []
        params = list(params)
        if isinstance(params[0],dict):
            # if dict then we want to verify that use_muon has been set
            for group in params:
                if "use_muon" not in group:
                    raise ValueError("Each param group dict must contain 'use_muon'")
                if group["use_muon"]: 
                    muon_params.append(group)
                else:
                    aux_params.append(group)
        elif isinstance(params[0], tuple) \
                and len(params[0]) == 2 \
                and isinstance(params[0][0], str) \
                and isinstance(params[0][1], nn.Parameter):
            # If named params we can find embedding layers  
            for n,p in params:
                if p.dim == 2 and "embed" not in n.lower():
                    muon_params.append(p)
                else:
                    aux_params.append(p)
        else: 
            # We cannot definitely know whether or not parameters
            # are embedding params, 
            warnings.warn("MuonWithAux cannot identify incompatible layers, can cause unexpected behaviour")
            for p in params:
                if p.ndim == 2:
                    muon_params.append(p)
                else: 
                    aux_params.append(p)
        if muon_type == "muonema":
            self.muon_opt = MuonEMA(muon_params, **muon_kwargs)
        else:
            self.muon_opt = Muon(muon_params,**muon_kwargs)
        self.aux_opt = aux_opt_class(aux_params,**aux_kwargs)
        
        # Combined param groups for base Optimizer bookkeeping
        param_groups: List[Dict[str, Any]] = []
        param_groups = self.muon_opt.param_groups.copy()
        param_groups += self.aux_opt.param_groups.copy()
        super().__init__(param_groups, defaults={})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.muon_opt.step()
        self.aux_opt.step()

        return loss
    
    @torch.no_grad()
    def zero_grad(self, set_to_none=False):
        self.muon_opt.zero_grad(set_to_none=set_to_none)
        self.aux_opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        """
        Returns a combined state dict from both internal optimizers.
        """
        state = {}
        if self.muon_opt:
            state["muon"] = self.muon_opt.state_dict()
        if self.aux_opt:
            state["aux"] = self.aux_opt.state_dict()
        return state

    def load_state_dict(self, state_dict):
        if "muon" in state_dict and self.muon_opt:
            self.muon_opt.load_state_dict(state_dict["muon"])
        if "aux" in state_dict and self.aux_opt:
            self.aux_opt.load_state_dict(state_dict["aux"])


# Taken from muon for the EPFL people
class CombinedScheduler:
    """
    CombinedScheduler implements a scheduler for the Muon optimizer: it leverages both Muon and AdamW learning rates, and applies the same sort of scheduler for both of them.

    Arguments:
        optimizer: Muon optimizer.
        cfg: arguments used for schedulers.
        muon_lr_key: defaults["lr"] is responsible for the Muon learning rate.
        adamw_lr_key: defaults["adamw_r"] is responsible for the AdamW learning rate.
    """

    def __init__(self, optimizer, cfg, muon_lr_key="lr", adamw_lr_key="adamw_lr"):
        self.schedulers = []
        scheduler_map = {
            "cos": torch.optim.lr_scheduler.OneCycleLR,
            "linear": torch.optim.lr_scheduler.OneCycleLR,
            "cos_inf": lambda opt, lr: torch.optim.lr_scheduler.LambdaLR(
                opt,
                cos_inf_schedule(
                    n_iterations=cfg.iterations,
                    n_warmup=cfg.warmup_steps,
                    n_inf=cfg.cos_inf_steps,
                    div_factor=1e2,
                    final_div_factor=0.1,
                ),
            ),
            "wsd": lambda opt, lr: torch.optim.lr_scheduler.LambdaLR(
                opt,
                wsd_schedule(
                    n_iterations=cfg.iterations,
                    n_warmup=cfg.warmup_steps,
                    fract_decay=cfg.wsd_fract_decay,
                    init_div_factor=1e2,
                    final_lr_factor=cfg.wsd_final_lr_scale,
                    decay_type=cfg.decay_type,
                ),
            ),
            "cos_wsd": lambda opt, lr: torch.optim.lr_scheduler.LambdaLR(
                opt,
                cosine_wsd_decay_schedule(
                    n_iterations=cfg.iterations,
                    n_warmup=cfg.warmup_steps,
                    anneal_end_factor=0.15,
                    fract_decay=cfg.wsd_fract_decay,
                    init_div_factor=1e2,
                    final_lr_factor=0.1,
                    decay_type=cfg.decay_type,
                ),
            ),
        }

        for group in optimizer.param_groups:
            lr_key = muon_lr_key if muon_lr_key in group else adamw_lr_key
            if lr_key in group:
                scheduler_cls = scheduler_map.get(cfg.scheduler, None)
                if scheduler_cls:
                    if cfg.scheduler in ["cos", "linear"]:
                        scheduler = scheduler_cls(
                            optimizer,
                            max_lr=[group.get(lr_key, getattr(cfg, lr_key.lower()))],
                            total_steps=cfg.iterations,
                            pct_start=cfg.warmup_steps / cfg.iterations,
                            anneal_strategy=cfg.scheduler,
                            cycle_momentum=False,
                            div_factor=1e2,
                            final_div_factor=1,
                        )
                    else:
                        scheduler = scheduler_cls(
                            optimizer, group.get(lr_key, getattr(cfg, lr_key.lower()))
                        )
                    self.schedulers.append(scheduler)

    def step(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def state_dict(self):
        state_dict = {}
        for i, scheduler in enumerate(self.schedulers):
            state_dict[f"scheduler_{i}"] = scheduler.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        for i, scheduler in enumerate(self.schedulers):
            scheduler.load_state_dict(state_dict[f"scheduler_{i}"])
