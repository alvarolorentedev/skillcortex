from skill_lattice_coder.cli import main
from test_analysis import _experiment


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


def test_analysis_cli_commands(capsys, tmp_path):
    experiment = _experiment(tmp_path)
    for command, filename in (
        ("analyze-router", "router_analysis.json"),
        ("analyze-python-regression", "python_regression_analysis.json"),
        ("analyze-composition", "composition_analysis.json"),
    ):
        assert main([command, "--experiment", str(experiment)]) == 0
        assert filename in capsys.readouterr().out
        assert (experiment / filename).exists()
