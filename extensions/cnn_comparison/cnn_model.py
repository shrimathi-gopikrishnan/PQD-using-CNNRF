"""1D-CNN architectures for the RF-vs-CNN comparison study.

Two scales:
  - PQDCNNLite     — intentionally tiny (~9 K params) for edge budgets
  - PQDCNNStandard — literature-realistic (~700 K params, similar to
                     Wang & Chen 2019, Khokhar et al. 2017)
"""

import torch
import torch.nn as nn


class PQDCNNLite(nn.Module):
    """Compact 1D-CNN (~9k params) targeting raw voltage windows."""

    def __init__(self, n_classes: int = 17):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 100)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.head(x)


class PQDCNNStandard(nn.Module):
    """Literature-realistic 1D-CNN for PQD classification.

    Sized similarly to published 1D-CNNs for power-quality classification
    (Wang & Chen 2019, Khokhar et al. 2017, Liu et al. 2020):
    4 conv blocks with growing channel width (64-128-256-256), large
    kernels in early layers to capture longer time structure, two FC
    layers in the head.

    ~700 K trainable parameters → ~3 MB on disk in float32.
    """

    def __init__(self, n_classes: int = 17):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=11, padding=5),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=7, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.block4 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.head(x)


class PQDCNNHeavy(nn.Module):
    """Heavyweight 1D-CNN — sized like the upper end of published deep
    PQD-classification networks (5+ conv blocks, channel widths up to 512).
    ~5 M parameters, ~100 M FLOPs / forward. Included to show where the
    CNN latency curve actually starts to approach the RF-pipeline cost.
    """

    def __init__(self, n_classes: int = 17):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=21, padding=10),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=17, padding=8),
            nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=13, padding=6),
            nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.block4 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=9, padding=4),
            nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.block5 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return self.head(x)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    for name, cls in [("PQDCNNLite", PQDCNNLite),
                       ("PQDCNNStandard", PQDCNNStandard),
                       ("PQDCNNHeavy", PQDCNNHeavy)]:
        m = cls()
        x = torch.randn(2, 1, 100); y = m(x)
        print(f"{name:18s}  params: {count_params(m):>9,d}   forward(2,1,100)->{tuple(y.shape)}")
