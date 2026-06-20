from skill_lattice_coder.router import RuleRouter


def test_router_handles_single_overlapping_and_ambiguous_prompts():
    router = RuleRouter()
    assert router.route("Write a Python function").selected_skills == ["python_skill"]
    assert router.route(
        "Fix this Python traceback and failing test"
    ).selected_skills == [
        "debugging_skill",
        "python_skill",
    ]
    assert router.route(
        "Generate pytest tests for this Python function"
    ).selected_skills == [
        "test_generation_skill",
        "python_skill",
    ]
    assert router.route("Explain this algorithm").selected_skills == ["python_skill"]


def test_explicit_dataset_tags_override_prompt_rules():
    decision = RuleRouter().route(
        "Write Python tests",
        tags=["debugging_skill", "python_skill", "test_generation_skill"],
    )
    assert decision.selected_skills == ["debugging_skill", "python_skill"]
    assert "tags" in decision.reason.lower()
