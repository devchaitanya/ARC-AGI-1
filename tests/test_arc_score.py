import json

from src.inference.arc_score import compare, load_truth, mcnemar, score_run, wilson_ci


def _write_truth(tmp_path):
    d = tmp_path / "truth"
    d.mkdir()
    (d / "t1.json").write_text(json.dumps({
        "train": [], "test": [{"input": [[1]], "output": [[1, 2], [3, 4]]}]}))
    (d / "t2.json").write_text(json.dumps({
        "train": [], "test": [{"input": [[1]], "output": [[5, 5]]}]}))
    return d


def _write_preds(tmp_path, name, payload):
    d = tmp_path / name
    d.mkdir()
    (d / "eval_predictions.json").write_text(json.dumps(payload))
    return d / "eval_predictions.json"


def test_second_attempt_counts_and_shape_is_upper_bound(tmp_path):
    truth = load_truth(_write_truth(tmp_path))
    # t1: attempt_2 is right. t2: right shape, one wrong cell.
    p = _write_preds(tmp_path, "run_a", {
        "t1": [{"attempt_1": [[9, 9], [9, 9]], "attempt_2": [[1, 2], [3, 4]]}],
        "t2": [{"attempt_1": [[5, 0]], "attempt_2": [[5, 0]]}],
    })
    r = score_run(p, truth)
    assert r["solved"] == 1 and r["exact_match"] == 0.5
    assert r["shape_acc"] == 1.0          # both shapes right
    assert r["cell_acc"] == 0.75          # t1 4/4, t2 1/2 -> mean of 1.0 and 0.5
    assert r["shape_acc"] >= r["exact_match"]


def test_wrong_shape_scores_zero_and_is_excluded_from_cell_acc(tmp_path):
    truth = load_truth(_write_truth(tmp_path))
    p = _write_preds(tmp_path, "run_b", {
        "t1": [{"attempt_1": [[1, 2, 3]], "attempt_2": [[1, 2, 3]]}],   # wrong shape
        "t2": [{"attempt_1": [[5, 5]], "attempt_2": [[5, 5]]}],         # correct
    })
    r = score_run(p, truth)
    assert r["solved"] == 1
    assert r["shape_acc"] == 0.5
    assert r["cell_acc"] == 1.0           # only the correctly-shaped one counts


def test_mcnemar_is_symmetric_and_flat_when_swaps_balance():
    m = mcnemar({"a", "b", "c"}, {"a", "d", "e"})
    assert m["both"] == 1
    assert m["only_first"] == ["b", "c"] and m["only_second"] == ["d", "e"]
    assert m["p_value"] == 1.0            # 2 lost, 2 gained -> no evidence
    assert mcnemar({"a"}, {"a"})["p_value"] == 1.0


def test_wilson_ci_brackets_the_estimate():
    lo, hi = wilson_ci(10, 400)
    assert lo < 10 / 400 < hi
    assert lo > 0 and hi < 0.06
    assert wilson_ci(0, 0) == (0.0, 0.0)


def test_compare_renders_both_runs_and_pairing(tmp_path):
    truth = load_truth(_write_truth(tmp_path))
    a = score_run(_write_preds(tmp_path, "r1", {
        "t1": [{"attempt_1": [[1, 2], [3, 4]]}], "t2": [{"attempt_1": [[0, 0]]}]}), truth)
    b = score_run(_write_preds(tmp_path, "r2", {
        "t1": [{"attempt_1": [[0, 0], [0, 0]]}], "t2": [{"attempt_1": [[5, 5]]}]}), truth)
    out = compare([a, b])
    assert "r1" in out and "r2" in out
    assert "McNemar" in out
