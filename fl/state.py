from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class ClientConfig:
    client_id: str
    lr: float
    local_epochs: int
    weight_decay: float = 0.0
    clip_grad: float = 0.0


@dataclass
class FLState:
    round: int
    global_params: Dict[str, Any]
    selected_clients: List[str]

