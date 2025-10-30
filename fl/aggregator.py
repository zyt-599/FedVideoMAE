from typing import Dict, List
import torch


def fedavg(client_states: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not client_states:
        return {}
    agg: Dict[str, torch.Tensor] = {}
    # Process metrics and model parameters separately
    metrics = {}
    for state in client_states:
        # Process model parameters
        for k, v in state.items():
            if k == 'metrics':
                # Skip metrics field, process later
                continue
            if k not in agg:
                agg[k] = v.clone().float()
            else:
                agg[k] += v.float()
    # Average model parameters
    for k in agg:
        agg[k] /= float(len(client_states))
        agg[k] = agg[k].to(client_states[0][k].dtype)
    return agg

