import json

from src.data.dataset import ARCDatasetConfig, collate_tasks, task_to_batch
from src.inference.ensemble import tta_predict_two_attempts, vote_top2
from src.inference.evaluate import predict_grids, predicted_shapes
from src.inference.submission import write_submission
from src.models.recursive_trm import DualStateRecursiveTransformer, TRMConfig


def _toy_task(task_id: int = 0) -> dict:
    return {
        "id": f"toy{task_id}",
        "_task_id": task_id,
        "train": [{"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 0]]}],
        "test": [{"input": [[1, 1], [0, 0]], "output": [[0, 0], [1, 1]]}],
    }


def _toy_model() -> DualStateRecursiveTransformer:
    cfg = TRMConfig(max_grid=6, max_demos=2, dim=48, heads=6, z_layers=1, y_layers=1, outer_steps=2, inner_steps=1)
    return DualStateRecursiveTransformer(cfg)


def _config() -> ARCDatasetConfig:
    return ARCDatasetConfig(max_grid=6, max_demos=2, augment=False)


def test_task_to_batch_and_collate_shapes():
    batch = task_to_batch(_toy_task(), _config())
    assert batch["test_input"].shape == (1, 6, 6)
    stacked = collate_tasks([_toy_task(0), _toy_task(1)], _config())
    assert stacked["test_input"].shape == (2, 6, 6)
    assert stacked["task_id"].tolist() == [0, 1]


def test_predict_grids_uses_shape_head():
    model = _toy_model()
    batch = task_to_batch(_toy_task(), _config())
    grids = predict_grids(model, batch, use_shape_head=True)
    h, w = predicted_shapes(model(batch), 6)[0]
    assert len(grids) == 1
    assert (len(grids[0]), len(grids[0][0])) == (h, w)

    ground_truth = predict_grids(model, batch, use_shape_head=False)[0]
    assert (len(ground_truth), len(ground_truth[0])) == (2, 2)


def test_vote_top2_prefers_majority():
    a, b = [[1, 1]], [[2, 2]]
    attempts = vote_top2([a, a, b])
    assert attempts[0] == a
    assert attempts[1] == b
    assert vote_top2([a]) == [a, a]


def test_tta_two_attempts_default_batch_builder():
    attempts = tta_predict_two_attempts(_toy_model(), _toy_task(), dataset_config=_config(), use_shape_head=False)
    assert len(attempts) == 2
    for grid in attempts:
        assert all(len(row) == len(grid[0]) for row in grid)
        assert all(0 <= cell <= 9 for row in grid for cell in row)


def test_write_submission_round_trip(tmp_path):
    path = write_submission({"abc": [[[1]], [[2]]]}, tmp_path / "submission.json")
    payload = json.loads(path.read_text())
    assert payload == {"abc": [{"attempt_1": [[1]], "attempt_2": [[2]]}]}
