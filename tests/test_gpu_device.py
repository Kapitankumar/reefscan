"""GPU-only regression tests (skip on CPU machines / CI)."""

from __future__ import annotations

import pytest

import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available on this machine"
)


def test_ordinal_loss_buffers_follow_device():
    """Regression: OrdinalLoss.position buffers stayed on CPU while logits/targets
    were on CUDA -> 'Expected all tensors to be on the same device' crash on every
    GPU machine (coral uses ordinal_loss=true). The Trainer must move the whole
    criterion module to the device."""
    from src.classification.losses import build_loss

    criterion = build_loss(2, ordinal=True, ordinal_lam=0.5).to("cuda")
    logits = torch.randn(4, 2, device="cuda")
    targets = torch.randint(0, 2, (4,), device="cuda")
    loss = criterion(logits, targets)
    assert criterion.positions.device.type == "cuda"
    assert float(loss) == float(loss)  # finite


def test_trainer_criterion_lands_on_cuda(tmp_path):
    """End-to-end: Trainer.__init__ must place the criterion (incl. buffers) on CUDA."""
    from torch.utils.data import DataLoader, TensorDataset

    from src.classification.engine import TrainConfig, Trainer

    cfg = TrainConfig(epochs=1, strategy="feature_extraction", ordinal_loss=True)
    ds = TensorDataset(torch.randn(8, 3, 64, 64), torch.randint(0, 2, (8,)))
    loader = DataLoader(ds, batch_size=2)
    trainer = Trainer(
        "resnet34",
        2,
        loader,
        None,
        cfg,
        ["a", "b"],
        torch.device("cuda"),
        out_dir=tmp_path / "out",
    )
    assert trainer.criterion.positions.device.type == "cuda"
