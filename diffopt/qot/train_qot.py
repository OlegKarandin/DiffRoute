"""Training script for SpanAttentionQoT model."""
from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import yaml

from diffopt.qot.dataset import SegmentQoTDataset
from diffopt.qot.model import SpanAttentionQoT


def load_config(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n = 0
    criterion = nn.MSELoss()
    for span_features, padding_mask, gsnr_db in loader:
        span_features = span_features.to(device)
        padding_mask = padding_mask.to(device)
        gsnr_db = gsnr_db.to(device)

        optimizer.zero_grad()
        pred = model(span_features, padding_mask)
        loss = criterion(pred, gsnr_db)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(gsnr_db)
        n += len(gsnr_db)
    return total_loss / n if n > 0 else 0.0


def eval_epoch(model, loader, device):
    model.eval()
    total_sq_err = 0.0
    n = 0
    with torch.no_grad():
        for span_features, padding_mask, gsnr_db in loader:
            span_features = span_features.to(device)
            padding_mask = padding_mask.to(device)
            gsnr_db = gsnr_db.to(device)

            pred = model(span_features, padding_mask)
            sq_err = ((pred - gsnr_db) ** 2).sum().item()
            total_sq_err += sq_err
            n += len(gsnr_db)
    rmse = math.sqrt(total_sq_err / n) if n > 0 else float("inf")
    return rmse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset_dir = Path(cfg["dataset_dir"])
    train_path = dataset_dir / "train.parquet"
    val_path = dataset_dir / "val.parquet"

    max_spans = cfg.get("max_spans_per_segment", 60)

    train_ds = SegmentQoTDataset(str(train_path), max_spans=max_spans)
    val_ds = SegmentQoTDataset(str(val_path), max_spans=max_spans)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"] * 2, shuffle=False, num_workers=0)

    model = SpanAttentionQoT(
        feature_dim=cfg.get("feature_dim", 5),
        model_dim=cfg.get("model_dim", 64),
        num_heads=cfg.get("num_heads", 4),
        num_layers=cfg.get("num_layers", 2),
        max_spans=max_spans,
        ff_dim=cfg.get("ff_dim", 128),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    epochs = cfg["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    checkpoint_dir = Path(cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(cfg["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / "train_log.csv"
    val_rmse_target = cfg.get("val_rmse_target", 0.5)
    best_rmse = float("inf")

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_mse", "val_rmse"])

        for epoch in range(1, epochs + 1):
            train_mse = train_epoch(model, train_loader, optimizer, device)
            val_rmse = eval_epoch(model, val_loader, device)
            scheduler.step()

            writer.writerow([epoch, f"{train_mse:.6f}", f"{val_rmse:.4f}"])
            f.flush()

            print(f"Epoch {epoch:3d}/{epochs} | train_mse={train_mse:.4f} | val_rmse={val_rmse:.4f} dB")

            if val_rmse < best_rmse:
                best_rmse = val_rmse
                ckpt_path = checkpoint_dir / "best_qot.pt"
                torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_rmse": val_rmse}, ckpt_path)
                print(f"  -> Saved checkpoint (val_rmse={val_rmse:.4f} dB)")

    print(f"\nTraining complete. Best val RMSE: {best_rmse:.4f} dB")
    if best_rmse < val_rmse_target:
        print(f"Target RMSE {val_rmse_target} dB achieved!")
    else:
        print(f"Target RMSE {val_rmse_target} dB NOT yet achieved.")


if __name__ == "__main__":
    main()
