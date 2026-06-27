TASK_TYPES = (
    "python_generation",
    "debugging",
    "test_generation",
)

SKILLS = ("python_skill", "debugging_skill", "test_generation_skill")
PROMOTED_SKILLS = ("alternating_skill",)
QUARANTINED_SKILLS = ()
KNOWN_SKILLS = (*SKILLS, *PROMOTED_SKILLS, *QUARANTINED_SKILLS)
MODES = ("base", "generic", "single-skill", "lattice", "oracle-lattice")
ROUTER_POLICIES = (
    "python_only_for_test_generation",
    "protected_skill_router",
    "protected_skill_router_without_failure_born",
    "skillcortex_router_v1",
    "legacy_rule_router",
    "weighted_task_composition",
    "reverse_weighted_task_composition",
    "protected_router_plus_alternating_skill",
)
