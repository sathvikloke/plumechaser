"""Compact CNN plume detector, replicating the Schuit et al. (2023) design.

Published description (section 2.2): convolutional front-end on the single
normalised CH4 channel, ReLU activations, one fully-connected layer with 40%
dropout before a sigmoid head, binary cross-entropy with double loss weight
on positive scenes. The exact filter counts were not published; we use a
standard compact configuration (8/16 filters) and report three-seed variance
with every trained model so reviewers can see stability rather than trust it.

Torch is an optional dependency: ``pip install plumechaser[ml]``.
"""

from __future__ import annotations


def build_model(input_size: int = 32):
    """Construct the CNN. Returns an ``nn.Module`` (lazy torch import)."""
    try:
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover - env guard
        raise ImportError("install the 'ml' extra: pip install plumechaser[ml]") from exc

    def block(cin: int, cout: int):
        return nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    flat = (input_size // 4) ** 2 * 16
    return nn.Sequential(
        block(1, 8),
        block(8, 16),
        nn.Flatten(),
        nn.Linear(flat, 32),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.4),
        nn.Linear(32, 1),
    )
