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
    from src.classification.engine import TrainConfig, Trainer
    from src.classification.models import build_model

    cfg = TrainConfig(epochs=1, batch_size=2, image_size=64, strategy="feature_extraction")
    model = build_model("resnet34", 2, pretrained=False, freeze_backbone=True).to("cuda")
    trainer = Trainer(
        model,
        model_name="resnet34",
        class_names=["a", "b"],
        cfg=cfg,
        device=torch.device("cuda"),
        out_dir=tmp_path / "out",
    )
    assert trainer.criterion.positions.device.type == "cuda"
