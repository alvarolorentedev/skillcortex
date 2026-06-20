import subprocess
import sys

from skill_lattice_coder.data import dataset_hash, load_jsonl
from skill_lattice_coder.metrics import extract_code
from skill_lattice_coder.utils import run_fixture


def test_benchmark_generator_is_reproducible(tmp_path):
    output = tmp_path / "eval.jsonl"
    subprocess.run(
        [sys.executable, "scripts/build_eval.py", str(output)],
        check=True,
    )
    assert dataset_hash(load_jsonl(output)) == dataset_hash(
        load_jsonl("data/eval.jsonl")
    )


def test_every_reference_target_passes_its_fixture():
    failures = [
        example.id
        for example in load_jsonl("data/eval.jsonl")
        if not run_fixture(example.execution, extract_code(example.target))[0]
    ]
    assert failures == []
