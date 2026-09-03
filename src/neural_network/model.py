"""
Definiția unei rețele neuronale simple pentru recunoașterea celor 15 clase de semne.
Acesta este un schelet de CNN folosit în Etapa 4 pentru a demonstra structura rețelei.
Pentru detecția finală în aplicație, folosim YOLO v11.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class TrafficSignNetwork(nn.Module):
    def __init__(self, classes: int = 15) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1), nn.BatchNorm2d(24), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(48, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(96, classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images).flatten(1))


def save_and_reload(path: Path) -> tuple[TrafficSignNetwork, torch.Size]:
    model = TrafficSignNetwork()
    output = model(torch.zeros(1, 3, 64, 64))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    restored = TrafficSignNetwork()
    restored.load_state_dict(torch.load(path, weights_only=True))
    return restored, output.shape


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    _, shape = save_and_reload(root / "models" / "traffic_sign_untrained.pt")
    print(f"Untrained model forward output: {tuple(shape)}")
