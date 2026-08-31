from src.data.augmentations import D8_TRANSFORMS, apply_color_permutation, transform_task


def test_d8_inverse_round_trip_rectangular_grid():
    grid = [[1, 2, 3], [4, 5, 6]]
    for transform in D8_TRANSFORMS:
        assert transform.inverse(transform.apply(grid)) == grid


def test_color_permutation_keeps_background():
    grid = [[0, 1, 2], [3, 0, 9]]
    perm = {0: 0, 1: 9, 2: 8, 3: 7, 9: 1}
    assert apply_color_permutation(grid, perm) == [[0, 9, 8], [7, 0, 1]]


def test_transform_task_preserves_schema():
    task = {
        "id": "toy",
        "train": [{"input": [[1]], "output": [[2]]}],
        "test": [{"input": [[3]], "output": [[4]]}],
    }
    out = transform_task(task, D8_TRANSFORMS[0], {0: 0, 1: 2, 2: 1, 3: 4, 4: 3})
    assert out["id"] == "toy"
    assert out["train"][0]["input"] == [[2]]
    assert out["test"][0]["output"] == [[3]]
