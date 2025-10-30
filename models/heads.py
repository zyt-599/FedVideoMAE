import torch.nn as nn


class LinearHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, *, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        layers = []
        nl = max(1, int(num_layers))  # number of hidden layers
        prev = in_dim
        for _ in range(nl):
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(nn.GELU())
            if dropout and float(dropout) > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            prev = hidden_dim
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
