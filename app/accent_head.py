"""Small regression/classification head on top of pooled HuBERT embeddings for labeled accent training."""

from torch import nn


class AccentHead(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.reg_head = nn.Linear(128, 1)
        self.cls_head = nn.Linear(128, num_classes) if num_classes > 0 else None

    def forward(self, pooled):
        x = self.backbone(pooled)
        reg = self.reg_head(x).squeeze(-1)
        cls = self.cls_head(x) if self.cls_head is not None else None
        return reg, cls
