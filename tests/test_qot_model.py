"""Tests for SpanAttentionQoT model."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from diffopt.qot.model import SpanAttentionQoT


@pytest.fixture
def model():
    return SpanAttentionQoT(
        feature_dim=5,
        model_dim=64,
        num_heads=4,
        num_layers=2,
        max_spans=60,
        ff_dim=128,
    )


def test_output_shape(model):
    """Output shape must be (batch,)."""
    batch_size = 8
    max_spans = 60
    span_features = torch.randn(batch_size, max_spans, 5)
    # First 5 spans real, rest padding
    padding_mask = torch.zeros(batch_size, max_spans, dtype=torch.bool)
    padding_mask[:, :5] = True

    out = model(span_features, padding_mask)
    assert out.shape == (batch_size,), f"Expected ({batch_size},), got {out.shape}"


def test_no_regen_inputs_in_forward():
    """Model forward signature must not include regenerator inputs."""
    import inspect
    sig = inspect.signature(SpanAttentionQoT.forward)
    params = list(sig.parameters.keys())
    # Only: self, span_features, padding_mask
    assert "regen" not in " ".join(params).lower()
    assert len(params) == 3, f"Expected 3 params (self, span_features, padding_mask), got: {params}"


def test_no_regen_in_model_params(model):
    """No parameter names should contain 'regen'."""
    for name, _ in model.named_parameters():
        assert "regen" not in name.lower(), f"Unexpected regen param: {name}"


def test_single_span_input(model):
    """Model should handle a single span correctly."""
    span_features = torch.randn(2, 60, 5)
    padding_mask = torch.zeros(2, 60, dtype=torch.bool)
    padding_mask[:, 0] = True  # Only first span is real

    out = model(span_features, padding_mask)
    assert out.shape == (2,)
    assert torch.isfinite(out).all()


def test_full_span_input(model):
    """Model should handle all 60 spans real."""
    span_features = torch.randn(4, 60, 5)
    padding_mask = torch.ones(4, 60, dtype=torch.bool)

    out = model(span_features, padding_mask)
    assert out.shape == (4,)
    assert torch.isfinite(out).all()


def test_mini_training_loss_decreases():
    """Loss should decrease over 10 epochs on 100 synthetic samples."""
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    # Generate synthetic data: GSNR ~ sum of first span feature (simplified)
    torch.manual_seed(42)
    n = 100
    max_spans = 60
    span_features = torch.randn(n, max_spans, 5)
    padding_mask = torch.zeros(n, max_spans, dtype=torch.bool)
    n_spans = torch.randint(1, 10, (n,))
    for i, ns in enumerate(n_spans):
        padding_mask[i, :ns] = True

    # Target GSNR: roughly based on features (just for convergence test)
    targets = span_features[:, :, 0].sum(dim=1) * 0.1 + 15.0

    dataset = TensorDataset(span_features, padding_mask, targets)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = SpanAttentionQoT(feature_dim=5, model_dim=32, num_heads=2, num_layers=1, max_spans=max_spans, ff_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    losses = []
    for epoch in range(10):
        model.train()
        epoch_loss = 0.0
        for sf, pm, gt in loader:
            optimizer.zero_grad()
            pred = model(sf, pm)
            loss = criterion(pred, gt)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss)

    # Loss should decrease from first epoch to last
    assert losses[-1] < losses[0], (
        f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
    )
