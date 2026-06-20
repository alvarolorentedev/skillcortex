#!/usr/bin/env python3
"""Run and summarize the five-seed SkillLatticeCoder experiment."""

import argparse
import json
from pathlib import Path

from skill_lattice_coder.evaluation import evaluate
from skill_lattice_coder.metrics import (
    aggregate_results,
    classify_hypothesis,
    paired_execution_comparison,
)
from skill_lattice_coder.schemas import SKILLS
from skill_lattice_coder.train_generic import train_generic
from skill_lattice_coder.train_skill import train_skill
from skill_lattice_coder.utils import write_json

DEFAULT_SEEDS = (11, 22, 33, 44, 55)


def summarize(root: Path, seeds: list[int]) -> dict:
    rows = []
    for seed in seeds:
        path = root / f"seed-{seed}" / "results.jsonl"
        for line in path.read_text().splitlines():
            row = json.loads(line)
            row["seed"] = seed
            row["benchmark_group"] = (
                f"{seed}:{row.get('benchmark_group') or row['example_id']}"
            )
            rows.append(row)
    modes = aggregate_results(rows)
    routed = paired_execution_comparison(rows)
    oracle = paired_execution_comparison(rows, candidate="oracle-lattice")
    summary = {
        "seeds": seeds,
        "hypothesis": (
            classify_hypothesis(modes, routed) if routed["count"] else "inconclusive"
        ),
        "generic_vs_lattice_execution": routed,
        "generic_vs_oracle_lattice_execution": oracle,
        "modes": modes,
    }
    write_json(root / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--output", default="artifacts/experiments/five-seed")
    parser.add_argument("--dataset", default="data/eval.jsonl")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = Path(args.output)
    for seed in args.seeds:
        seed_root = root / f"seed-{seed}"
        adapter_root = seed_root / "adapters"
        for skill in SKILLS:
            train_skill(
                skill,
                seed=seed,
                adapter_root=adapter_root,
                dry_run=args.dry_run,
                force=args.force,
            )
        train_generic(
            seed=seed,
            adapter_root=adapter_root,
            dry_run=args.dry_run,
            force=args.force,
        )
        evaluate(
            args.dataset,
            output=seed_root,
            adapter_root=adapter_root,
            dry_run=args.dry_run,
        )
    print(json.dumps(summarize(root, args.seeds), indent=2))


if __name__ == "__main__":
    main()
