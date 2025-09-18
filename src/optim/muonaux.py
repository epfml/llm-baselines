from typing import Callable, Optional, Dict, Any, List, Union, Tuple
import torch
from torch import nn
from torch.optim import (
    Optimizer,
    ParamsT,
)
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
        muon_kwargs: Optional[dict] = None,
        aux_opt_class: Callable[Optimizer] = torch.optim.AdamW,
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
        if 'muon_type' in muon_params and muon_params['muon_type'] == "muonema":
            self.muon_opt = MuonEMA(muon_params, **muon_kwargs)
        else:
            self.muon_opt = Muon(muon_params,**muon_kwargs)
        self.aux_opt = aux_opt_class(aux_params,**aux_kwargs)
        
        # Combined param groups for base Optimizer bookkeeping
        param_groups: List[Dict[str, Any]] = []
        param_groups = self.muon_opt.param_groups 
        param_groups += self.aux_opt.param_groups
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


