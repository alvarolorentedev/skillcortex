from skill_lattice_coder.cli import main


def test_all_cli_paths_support_dry_run(capsys, tmp_path):
    assert main(["train-skill", "python_skill", "--dry-run"]) == 0
    assert main(["train-generic", "--dry-run"]) == 0
    assert (
        main(
            [
                "infer",
                "--mode",
                "lattice",
                "--prompt",
                "Fix this Python traceback",
                "--dry-run",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "eval",
                "--dataset",
                "data/eval.jsonl",
                "--output",
                str(tmp_path),
                "--dry-run",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert '"selected_skills"' in output
