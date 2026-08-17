"""PyTorch Dataset for segment-level QoT data."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from diffopt.qot.span_features import SPAN_FEATURE_DIM


class SegmentQoTDataset(Dataset):
    """Dataset of transparent segments with per-span features and GSNR labels.

    Each row in the parquet file represents one transparent segment.
    Span features are stored flattened: span_features_0 .. span_features_{max_spans*5-1}.

    Returns:
        span_features: FloatTensor (max_spans, 5)
        padding_mask: BoolTensor (max_spans,) - True for real spans, False for padding
        gsnr_db: scalar FloatTensor
    """

    # Same number as diffopt.qot.span_features.SPAN_FEATURE_DIM, for the same
    # reason: this is the per-span feature count the parquet schema's
    # span_features_0 .. span_features_{max_spans*5-1} columns are laid out
    # against. Re-exported (not redefined) so the two can't drift apart.
    FEATURE_DIM = SPAN_FEATURE_DIM

    def __init__(self, parquet_path: str, max_spans: int = 60):
        self.max_spans = max_spans
        df = pd.read_parquet(parquet_path)
        self.n_spans = df["n_spans"].to_numpy(dtype=np.int32)
        self.gsnr_db = df["gsnr_db"].to_numpy(dtype=np.float32)

        # Extract flattened span features
        feat_cols = [f"span_features_{i}" for i in range(max_spans * self.FEATURE_DIM)]
        self.span_features_flat = df[feat_cols].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.gsnr_db)

    def __getitem__(self, idx: int):
        flat = self.span_features_flat[idx]  # (max_spans * 5,)
        span_features = torch.from_numpy(flat.reshape(self.max_spans, self.FEATURE_DIM))

        n = int(self.n_spans[idx])
        padding_mask = torch.zeros(self.max_spans, dtype=torch.bool)
        padding_mask[:n] = True

        gsnr = torch.tensor(self.gsnr_db[idx], dtype=torch.float32)
        return span_features, padding_mask, gsnr
