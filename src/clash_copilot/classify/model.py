"""Tiny CNN for card identity (the AmarSaini/KataCR-validated approach).

Torch is an optional dependency: install with `uv pip install -e ".[ml]"`.
Input: BGR uint8 crops at augment.SIZE; output: card name (or "empty") with
a softmax probability.
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn

from clash_copilot.classify.augment import SIZE


def _build_net(n_classes: int) -> nn.Module:
    def block(cin: int, cout: int) -> list[nn.Module]:
        return [nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]

    flat = 128 * (SIZE[1] // 8) * (SIZE[0] // 8)
    return nn.Sequential(
        *block(3, 32), *block(32, 64), *block(64, 128),
        nn.Flatten(),
        nn.Linear(flat, 256), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(256, n_classes),
    )


def _to_tensor(images: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(images).float().permute(0, 3, 1, 2) / 255.0


class CardClassifier:
    def __init__(self, net: nn.Module, class_names: list[str]):
        self.net = net
        self.class_names = class_names

    @classmethod
    def new(cls, class_names: list[str]) -> "CardClassifier":
        return cls(_build_net(len(class_names)), list(class_names))

    def fit(self, images: np.ndarray, labels: np.ndarray, epochs: int, lr: float = 1e-3,
            batch_size: int = 64) -> float:
        """Train on one array of samples; returns the last epoch's mean loss."""
        self.net.train()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        x, y = _to_tensor(images), torch.from_numpy(labels)
        last = 0.0
        for _ in range(epochs):
            order = torch.randperm(len(x))
            losses = []
            for start in range(0, len(x), batch_size):
                idx = order[start : start + batch_size]
                optimizer.zero_grad()
                loss = loss_fn(self.net(x[idx]), y[idx])
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            last = sum(losses) / len(losses)
        return last

    def predict(self, images: np.ndarray) -> list[tuple[str, float]]:
        self.net.eval()
        with torch.no_grad():
            probs = torch.softmax(self.net(_to_tensor(images)), dim=1)
        best = probs.argmax(dim=1)
        return [
            (self.class_names[int(i)], float(probs[row, i]))
            for row, i in enumerate(best)
        ]

    def save(self, path: str | Path) -> None:
        torch.save(
            {"state": self.net.state_dict(), "class_names": self.class_names}, path
        )

    @classmethod
    def load(cls, path: str | Path) -> "CardClassifier":
        payload = torch.load(path, map_location="cpu", weights_only=True)
        clf = cls.new(payload["class_names"])
        clf.net.load_state_dict(payload["state"])
        return clf
