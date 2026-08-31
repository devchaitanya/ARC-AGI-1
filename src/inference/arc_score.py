"""ARC-protocol scoring with enough resolution to compare runs.

Exact-match on 400 tasks cannot resolve differences smaller than a few points,
so a run is summarised with three metrics of increasing resolution:

  exact_match   ARC's own rule -- every test case in a task correct, two
                attempts allowed. This is the headline number and the only
                one comparable with published ARC results.
  shape_acc     Fraction of test cases where the predicted grid has the right
                dimensions. Wrong shape means exact-match is impossible, so
                this is an upper bound on exact_match.
  cell_acc      Mean fraction of cells correct, over correctly-shaped
                predictions only. Moves continuously, so it registers progress
                long before any extra task flips to solved.

Comparing two runs uses McNemar's exact test on the paired solved-sets rather
than comparing two independent proportions -- far more power at these rates.
"""

from __future__ import annotations

import json
from math import comb
from pathlib import Path


def load_truth(truth_dir: str | Path) -> dict[str, list[dict]]:
    """Map task id -> its list of test cases (each with 'input' and 'output')."""

    out = {}
    for f in sorted(Path(truth_dir).glob("*.json")):
        with f.open(encoding="utf-8") as fh:
            out[f.stem] = json.load(fh)["test"]
    return out


def _grid_shape(grid) -> tuple[int, int]:
    if not grid or not grid[0]:
        return (0, 0)
    return (len(grid), len(grid[0]))


def _cell_accuracy(pred, want) -> float:
    total = sum(len(r) for r in want)
    if total == 0:
        return 0.0
    hit = sum(p == w for pr, wr in zip(pred, want) for p, w in zip(pr, wr))
    return hit / total


def score_run(pred_path: str | Path, truth: dict[str, list[dict]]) -> dict:
    """Score one run's eval_predictions.json against ground truth."""

    with Path(pred_path).open(encoding="utf-8") as fh:
        preds = json.load(fh)
    solved: set[str] = set()
    n_tasks = n_cases = 0
    case_exact = shape_ok = 0
    cell_accs: list[float] = []

    for name, cases in preds.items():
        if name not in truth:
            continue
        gts = truth[name]
        n_tasks += 1
        task_ok = len(cases) == len(gts) and len(gts) > 0

        for case, gt in zip(cases, gts):
            n_cases += 1
            want = gt["output"]
            attempts = [case.get(f"attempt_{a}") for a in (1, 2)]
            attempts = [a for a in attempts if a]

            if any(a == want for a in attempts):
                case_exact += 1
            else:
                task_ok = False

            # ARC allows two attempts, so credit the better of them.
            shaped = [a for a in attempts if _grid_shape(a) == _grid_shape(want)]
            if shaped:
                shape_ok += 1
                cell_accs.append(max(_cell_accuracy(a, want) for a in shaped))

        if task_ok:
            solved.add(name)

    return {
        "run": Path(pred_path).parent.name,
        "n_tasks": n_tasks,
        "n_cases": n_cases,
        "exact_match": len(solved) / n_tasks if n_tasks else 0.0,
        "solved": len(solved),
        "case_exact_match": case_exact / n_cases if n_cases else 0.0,
        "shape_acc": shape_ok / n_cases if n_cases else 0.0,
        "cell_acc": sum(cell_accs) / len(cell_accs) if cell_accs else 0.0,
        "solved_set": solved,
    }


def wilson_ci(x: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval -- honest at the small counts ARC produces."""

    if n == 0:
        return (0.0, 0.0)
    from math import sqrt

    p = x / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar(a: set[str], b: set[str]) -> dict:
    """Exact McNemar on two paired solved-sets."""

    only_a, only_b = a - b, b - a
    n01, n10 = len(only_a), len(only_b)
    n = n01 + n10
    if n == 0:
        p = 1.0
    else:
        tail = sum(comb(n, k) for k in range(min(n01, n10) + 1))
        p = min(1.0, 2 * tail / (2 ** n))
    return {
        "both": len(a & b),
        "only_first": sorted(only_a),
        "only_second": sorted(only_b),
        "p_value": p,
    }


def compare(runs: list[dict]) -> str:
    """Render a comparison table plus pairwise McNemar against the first run."""

    lines = [
        f"{'run':<26} {'exact':>8} {'95% CI':>16} {'cases':>8} {'shape':>8} {'cell':>8}",
        "-" * 78,
    ]
    for r in runs:
        lo, hi = wilson_ci(r["solved"], r["n_tasks"])
        lines.append(
            f"{r['run']:<26} {r['exact_match']:>7.2%} "
            f"{f'[{lo:.2%}, {hi:.2%}]':>16} {r['case_exact_match']:>7.2%} "
            f"{r['shape_acc']:>7.2%} {r['cell_acc']:>7.2%}"
        )

    if len(runs) > 1:
        base = runs[0]
        lines += ["", f"Paired vs {base['run']} (McNemar exact):"]
        for r in runs[1:]:
            m = mcnemar(base["solved_set"], r["solved_set"])
            lines.append(
                f"  {r['run']:<26} both={m['both']:<3} "
                f"lost={len(m['only_first']):<3} gained={len(m['only_second']):<3} "
                f"p={m['p_value']:.3f}"
            )
    return "\n".join(lines)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", help="run directories containing eval_predictions.json")
    ap.add_argument("--truth", required=True, help="ARC evaluation data directory")
    args = ap.parse_args()

    truth = load_truth(args.truth)
    scored = []
    for d in args.runs:
        p = Path(d) / "eval_predictions.json"
        if not p.exists():
            print(f"skip {d}: no eval_predictions.json")
            continue
        scored.append(score_run(p, truth))
    print(compare(scored))


if __name__ == "__main__":
    main()
