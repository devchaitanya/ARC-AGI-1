import torch

from src.data.dataset import ARCDatasetConfig, ARCTaskDataset
from src.models.layers import count_parameters
from src.models.recursive_trm import DualStateRecursiveTransformer, TRMConfig
from src.training.loss import deep_supervision_loss


def _toy_task():
    return {
        "id": "toy",
        "_task_id": 0,
        "train": [{"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 0]]}],
        "test": [{"input": [[1, 1], [0, 0]], "output": [[0, 0], [1, 1]]}],
    }


def test_dataset_padding_shapes():
    ds = ARCTaskDataset([_toy_task()], ARCDatasetConfig(max_grid=6, max_demos=2))
    sample = ds[0]
    assert sample["demo_inputs"].shape == (2, 6, 6)
    assert sample["test_output_mask"].sum().item() == 4
    assert sample["output_shape"].tolist() == [2, 2]


def test_model_forward_and_parameter_budget():
    ds = ARCTaskDataset([_toy_task()], ARCDatasetConfig(max_grid=6, max_demos=2))
    batch = {k: v.unsqueeze(0) for k, v in ds[0].items()}
    cfg = TRMConfig(max_grid=6, max_demos=2, dim=48, heads=6, z_layers=1, y_layers=1, outer_steps=2, inner_steps=1)
    model = DualStateRecursiveTransformer(cfg)
    out = model(batch)
    assert out["logits"].shape == (1, 10, 6, 6)
    assert len(out["logits_by_step"]) == 2
    assert count_parameters(model) <= cfg.max_params
    loss = deep_supervision_loss(out["logits_by_step"], batch["test_output"], batch["test_output_mask"])
    assert torch.isfinite(loss)
