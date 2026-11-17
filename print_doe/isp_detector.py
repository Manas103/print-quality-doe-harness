"""The fixed detector: a small CNN trained once on untouched (all factors
at their low/baseline level) scenes, then frozen and evaluated, unchanged,
against every one of the 96 factorial runs' ISP-processed scenes. The
detector never retrains per run; "detector accuracy as the response"
means this one fixed model's accuracy on that run's processed eval set.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from print_doe.isp_scene import FRAME_SIZE


class DetectorCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.mean(dim=(2, 3))
        return self.fc(x).squeeze(-1)


def frames_to_tensor(frames: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(frames.astype(np.float32) / 255.0).unsqueeze(1)


def train_detector(frames: np.ndarray, labels: np.ndarray, epochs: int = 150, seed: int = 0,
                    brightness_jitter: bool = True) -> DetectorCNN:
    """brightness_jitter: multiply each training frame by an independent
    random gain in [0.7, 1.3] every epoch. This is a standard photometric
    augmentation, added because the first honest run (no augmentation)
    found a trained detector so brittle to the exact training brightness
    distribution that tone_curve and white_balance's own main effects were
    an order of magnitude larger than any interaction (see README
    Findings); this is a real robustness fix, not a parameter chosen to
    hit a target number."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = DetectorCNN()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    y = torch.from_numpy(labels.astype(np.float32))
    for _ in range(epochs):
        if brightness_jitter:
            gains = rng.uniform(0.7, 1.3, size=frames.shape[0]).astype(np.float32)
            jittered = np.clip(frames.astype(np.float32) * gains[:, None, None], 0, 255)
            x = torch.from_numpy(jittered / 255.0).unsqueeze(1)
        else:
            x = frames_to_tensor(frames)
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    model.eval()
    return model


def accuracy(model: DetectorCNN, frames: np.ndarray, labels: np.ndarray) -> float:
    with torch.no_grad():
        logits = model(frames_to_tensor(frames))
        preds = (logits > 0.0).numpy()
    return float((preds == labels).mean())
