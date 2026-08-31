import math

import torch

from src.data.dataset import ARCDatasetConfig, task_to_batch
from src.models.recursive_trm import DualStateRecursiveTransformer, TRMConfig
from src.training.scheduler import cosine_with_warmup, wsd_schedule
from src.training.trainer import ARCTrainer, TrainConfig


def _toy_task() -> dict:
    return {
        "id": "toy",
        "_task_id": 0,
        "train": [{"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 0]]}],
        "test": [{"input": [[1, 1], [0, 0]], "output": [[0, 0], [1, 1]]}],
    }


def _trainer(tmp_path) -> ARCTrainer:
    cfg = TRMConfig(max_grid=6, max_demos=2, dim=48, heads=6, z_layers=1, y_layers=1, outer_steps=2, inner_steps=1)
    model = DualStateRecursiveTransformer(cfg)
    train_cfg = TrainConfig(total_steps=4, warmup_steps=1, checkpoint_dir=str(tmp_path), amp=False)
    return ARCTrainer(model, train_cfg, device="cpu")


def test_train_step_reduces_finite_loss(tmp_path):
    trainer = _trainer(tmp_path)
    batch = task_to_batch(_toy_task(), ARCDatasetConfig(max_grid=6, max_demos=2, augment=False))
    losses = [trainer.train_step(batch)["loss"] for _ in range(3)]
    assert all(math.isfinite(loss) for loss in losses)
    assert losses[-1] < losses[0]


def test_checkpoint_round_trip(tmp_path):
    trainer = _trainer(tmp_path)
    batch = task_to_batch(_toy_task(), ARCDatasetConfig(max_grid=6, max_demos=2, augment=False))
    trainer.train_step(batch)
    path = trainer.save_checkpoint("ckpt.pt")
    reloaded = _trainer(tmp_path)
    reloaded.load_checkpoint(path)
    for (name, a), (_, b) in zip(trainer.model.named_parameters(), reloaded.model.named_parameters()):
        assert torch.equal(a, b), name


def test_ema_averaged_context_restores_weights(tmp_path):
    trainer = _trainer(tmp_path)
    batch = task_to_batch(_toy_task(), ARCDatasetConfig(max_grid=6, max_demos=2, augment=False))
    trainer.train_step(batch)
    before = {n: p.detach().clone() for n, p in trainer.model.named_parameters()}
    with trainer.ema.averaged(trainer.model):
        pass
    for name, param in trainer.model.named_parameters():
        assert torch.equal(param, before[name]), name


def test_schedulers_warm_up_and_decay():
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([param], lr=1.0)
    sched = cosine_with_warmup(opt, warmup_steps=2, total_steps=10)
    lrs = []
    for _ in range(10):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] < lrs[2]
    assert lrs[-1] < lrs[2]

    opt2 = torch.optim.SGD([param], lr=1.0)
    wsd = wsd_schedule(opt2, warmup_steps=2, stable_steps=3, decay_steps=5, min_lr_ratio=0.1)
    seen = []
    for _ in range(12):
        seen.append(opt2.param_groups[0]["lr"])
        opt2.step()
        wsd.step()
    assert seen[3] == 1.0
    assert seen[-1] >= 0.1 - 1e-6
    assert seen[-1] < 1.0
