import torch


class HalmesResBlock(torch.nn.Module):
    def __init__(self, in_features: int, dropout: float = 0.15):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features),
            torch.nn.LayerNorm(in_features),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)