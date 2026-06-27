import io
import json
from contextlib import contextmanager

from skillcortex.cli import main
from skillcortex.runtime import OpenAICompatApp, SkillRuntime


def _compose_runtime(tmp_path):
    eval_summary = tmp_path / "eval-summary.json"
    eval_summary.write_text(json.dumps({"modes": {}, "tasks": {}}) + "\n")
    packages = {}
    for skill_id, name in (
        ("python_skill", "Python Skill"),
        ("debugging_skill", "Debugging Skill"),
    ):
        output = tmp_path / skill_id
        packages[skill_id] = output
        assert (
            main(
                [
                    "package-skill",
                    "--skill-id",
                    skill_id,
                    "--name",
                    name,
                    "--adapter-dir",
                    f"artifacts/adapters/{skill_id}",
                    "--output",
                    str(output),
                    "--train-dataset",
                    "data/train.jsonl",
                    "--eval-dataset",
                    "data/eval.jsonl",
                    "--eval-summary",
                    str(eval_summary),
                ]
            )
            == 0
        )
    runtime = tmp_path / "runtime"
    assert (
        main(
            [
                "compose-skills",
                "--skills",
                ",".join(str(path) for path in packages.values()),
                "--strategy",
                "routed",
                "--output",
                str(runtime),
            ]
        )
        == 0
    )
    return runtime


def _wsgi_request(app, method, path, payload=None):
    raw = json.dumps(payload).encode() if payload is not None else b""
    environ = {
        "REQUEST_METHOD": method,
        "PATH_INFO": path,
        "CONTENT_LENGTH": str(len(raw)),
        "wsgi.input": io.BytesIO(raw),
    }
    captured = {}

    def start_response(status, headers):
        captured["status"] = status
        captured["headers"] = headers

    body = b"".join(app(environ, start_response))
    return captured["status"], json.loads(body.decode())


def test_validate_runtime_bundle_accepts_composed_runtime(tmp_path):
    runtime = _compose_runtime(tmp_path)
    assert main(["validate-runtime", "--runtime", str(runtime)]) == 0


def test_validate_runtime_bundle_rejects_checksum_tamper(tmp_path):
    runtime = _compose_runtime(tmp_path)
    (runtime / "README.md").write_text("tampered\n")
    assert main(["validate-runtime", "--runtime", str(runtime)]) == 2


def test_runtime_infer_dry_run_uses_bundle_route_selection(tmp_path, capsys):
    runtime = _compose_runtime(tmp_path)
    capsys.readouterr()
    assert (
        main(
            [
                "infer",
                "--runtime",
                str(runtime),
                "--prompt",
                "Fix this Python traceback and failing test",
                "--dry-run",
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["selected_skills"] == ["debugging_skill", "python_skill"]
    assert output["route_type"] == "adapter"


def test_openai_compat_app_exposes_models_and_chat(tmp_path, monkeypatch):
    runtime_path = _compose_runtime(tmp_path)
    runtime = SkillRuntime.load(runtime_path)

    monkeypatch.setattr(
        runtime,
        "infer",
        lambda **kwargs: {
            "generation": "patched response",
            "prompt_tokens": 4,
            "generated_tokens": 2,
            "selected_skills": ["debugging_skill", "python_skill"],
            "route_type": "adapter",
            "reason": "patched",
        },
    )

    app = OpenAICompatApp(runtime)
    status, models = _wsgi_request(app, "GET", "/v1/models")
    assert status.startswith("200")
    assert models["data"][0]["id"] == "runtime"

    status, completion = _wsgi_request(
        app,
        "POST",
        "/v1/chat/completions",
        {
            "model": "runtime",
            "messages": [{"role": "user", "content": "Fix this traceback"}],
        },
    )
    assert status.startswith("200")
    assert completion["choices"][0]["message"]["content"] == "patched response"
    assert completion["usage"]["total_tokens"] == 6


def test_runtime_non_dry_run_calls_backend_seams(tmp_path, monkeypatch):
    runtime_path = _compose_runtime(tmp_path)
    runtime = SkillRuntime.load(runtime_path)
    calls = {}

    @contextmanager
    def fake_temporary_composed_adapter(paths, weights=None):
        calls["adapter_paths"] = [str(path) for path in paths]
        calls["weights"] = weights
        yield tmp_path / "composed-adapter"

    def fake_load_model(adapter=None, model_name=None):
        calls["load_model"] = {
            "adapter": str(adapter) if adapter else None,
            "model_name": model_name,
        }
        return "fake-model", "fake-tokenizer"

    def fake_generate_text(model, tokenizer, prompt=None, *, messages=None, max_tokens=None, temperature=None):
        calls["generate_text"] = {
            "model": model,
            "tokenizer": tokenizer,
            "prompt": prompt,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        return "runtime answer", 11, 7

    monkeypatch.setattr("skillcortex.runtime.temporary_composed_adapter", fake_temporary_composed_adapter)
    monkeypatch.setattr("skillcortex.runtime.load_model", fake_load_model)
    monkeypatch.setattr("skillcortex.runtime.generate_text", fake_generate_text)

    result = runtime.infer(
        prompt="Fix this Python traceback and failing test",
        max_tokens=33,
        temperature=0.2,
        dry_run=False,
    )

    assert result["status"] == "complete"
    assert result["selected_skills"] == ["debugging_skill", "python_skill"]
    assert calls["adapter_paths"] == [
        str(runtime.bundle.skills["debugging_skill"].adapter_path),
        str(runtime.bundle.skills["python_skill"].adapter_path),
    ]
    assert calls["load_model"]["model_name"] == runtime.bundle.runtime_model
    assert calls["generate_text"]["messages"] == [
        {"role": "user", "content": "Fix this Python traceback and failing test"}
    ]
    assert calls["generate_text"]["max_tokens"] == 33
    assert calls["generate_text"]["temperature"] == 0.2


def test_runtime_infer_supports_request_file(tmp_path, capsys):
    runtime = _compose_runtime(tmp_path)
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "system", "content": "You are a debugging assistant."},
                    {"role": "user", "content": "Fix this Python traceback and failing test"},
                ],
                "task_type": "debugging",
            }
        )
        + "\n"
    )
    capsys.readouterr()
    assert (
        main(
            [
                "infer",
                "--runtime",
                str(runtime),
                "--request-file",
                str(request),
                "--dry-run",
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["selected_skills"] == ["debugging_skill", "python_skill"]


def test_runtime_infer_rejects_malformed_request_file(tmp_path, capsys):
    runtime = _compose_runtime(tmp_path)
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"messages": [{"role": "tool", "content": 1}]}) + "\n")
    capsys.readouterr()
    assert (
        main(
            [
                "infer",
                "--runtime",
                str(runtime),
                "--request-file",
                str(request),
                "--dry-run",
            ]
        )
        == 2
    )
    assert "message role must be one of" in capsys.readouterr().err


def test_runtime_infer_requires_exactly_one_prompt_input(tmp_path, capsys):
    runtime = _compose_runtime(tmp_path)
    capsys.readouterr()
    assert main(["infer", "--runtime", str(runtime), "--dry-run"]) == 2
    assert "exactly one of --prompt or --request-file is required" in capsys.readouterr().err