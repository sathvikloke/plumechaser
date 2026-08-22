"""Deterministic CNN training with flip/rot augmentation and early stopping.

Follows the frozen-plan requirements: seeded (numpy/torch/random), 2:1
positive loss weight, validation-loss early stopping keeping best weights,
and a three-seed retrain loop whose score spread is reported alongside the
model so no single lucky seed carries the result.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover
        pass


def _augment_batch(xb, yb, rng: np.random.Generator):
    """Random horizontal/vertical flips + 90-degree rotations per-sample."""
    import torch

    n = xb.shape[0]
    flips_h = torch.from_numpy(rng.integers(0, 2, n).astype(bool))
    flips_v = torch.from_numpy(rng.integers(0, 2, n).astype(bool))
    rots = rng.integers(0, 4, n)
    for i in range(n):
        t = xb[i]
        if flips_v[i]:
            t = torch.flip(t, dims=(1,))
        if flips_h[i]:
            t = torch.flip(t, dims=(2,))
        k = int(rots[i])
        if k:
            t = torch.rot90(t, k=k, dims=(1, 2))
        xb[i] = t
    return xb, yb


def train_cnn(
    x: np.ndarray,
    y: np.ndarray,
    out_dir: str | Path,
    *,
    seed: int = 0,
    epochs: int = 60,
    batch_size: int = 64,
    lr: float = 1e-3,
    val_fraction: float = 0.15,
    patience: int = 6,
) -> dict:
    """Train the plume CNN; writes model.pt + metrics.json into ``out_dir``."""
    from plumechaser.ml.model import build_model

    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:  # pragma: no cover - env guard
        raise ImportError("install the 'ml' extra: pip install plumechaser[ml]") from exc

    set_seeds(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    idx = np.random.default_rng(seed).permutation(len(x))
    n_val = max(1, int(len(x) * val_fraction))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(input_size=x.shape[-1]).to(dev)
    # pos_weight=2 replicates Schuit's double loss weight on plume scenes.
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.0))
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    tr_loader = DataLoader(
        TensorDataset(torch.from_numpy(x[tr_idx]), torch.from_numpy(y[tr_idx])),
        batch_size=batch_size,
        shuffle=False,  # we augment deterministically per-epoch instead
    )
    xv = torch.from_numpy(x[val_idx]).to(dev)
    yv = torch.from_numpy(y[val_idx]).to(dev)

    rng = np.random.default_rng(seed + 1)
    best_loss, best_state, best_epoch, bad = float("inf"), None, -1, 0
    history = []
    for epoch in range(epochs):
        model.train()
        for xtr, ytr in tr_loader:
            xtr, ytr = xtr.to(dev), ytr.to(dev).unsqueeze(1)
            xtr, ytr = _augment_batch(xtr, ytr, rng)
            opt.zero_grad()
            loss = criterion(model(xtr), ytr)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = float(criterion(model(xv), yv.unsqueeze(1)))
        history.append(vloss)
        if vloss < best_loss - 1e-4:
            best_loss, best_epoch, bad = vloss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), out / "model.pt")

    with torch.no_grad():
        logits = model(xv).squeeze(1).cpu().numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    truth = y[val_idx].astype(int)
    tp = int(((preds == 1) & (truth == 1)).sum())
    fp = int(((preds == 1) & (truth == 0)).sum())
    fn = int(((preds == 0) & (truth == 1)).sum())
    metrics = {
        "seed": seed,
        "best_epoch": best_epoch,
        "val_loss": best_loss,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "n_train": int(len(tr_idx)),
        "n_val": int(n_val),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def train_three_seeds(
    x: np.ndarray, y: np.ndarray, out_dir: str | Path, seeds: tuple[int, ...] = (0, 1, 2)
) -> list[dict]:
    """Frozen-plan requirement: report seed-variance instead of a single run."""
    results = [
        train_cnn(x, y, Path(out_dir) / f"seed{s}", seed=s) for s in seeds
    ]
    precisions = [r["precision"] for r in results if r["precision"] is not None]
    recalls = [r["recall"] for r in results if r["recall"] is not None]
    summary = {
        "per_seed": results,
        "precision_mean_std": ([float(np.mean(precisions)),
                                float(np.std(precisions))] if precisions else None),
        "recall_mean_std": ([float(np.mean(recalls)),
                             float(np.std(recalls))] if recalls else None),
    }
    (Path(out_dir) / "seed_summary.json").write_text(json.dumps(summary, indent=2))
    return results
