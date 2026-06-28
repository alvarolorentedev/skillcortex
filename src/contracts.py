TASK_TYPES = (
    "python_generation",
    "debugging",
    "test_generation",
)

PRESET_SLMS = ("python_slm", "debugging_slm", "test_generation_slm")
PROMOTED_SLMS = ("alternating_slm",)
QUARANTINED_SLMS = ()
KNOWN_SLMS = (*PRESET_SLMS, *PROMOTED_SLMS, *QUARANTINED_SLMS)
MODES = ("base", "generic", "single-slm", "lattice", "oracle-lattice")
ROUTER_POLICIES = (
    "python_only_for_test_generation",
    "protected_slm_router",
    "protected_slm_router_without_failure_born",
    "slmcortex_router_v1",
    "legacy_rule_router",
    "weighted_task_composition",
    "reverse_weighted_task_composition",
    "protected_router_plus_alternating_slm",
)
