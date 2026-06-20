from pathlib import Path

import yaml

from skill_lattice_coder.train_generic import build_generic_command
from skill_lattice_coder.train_skill import build_skill_command


def test_training_commands_use_expected_rank_and_mask_prompt(tmp_path):
    skill = build_skill_command(
        "debugging_skill", tmp_path / "data", tmp_path / "adapter"
    )
    generic = build_generic_command(tmp_path / "data", tmp_path / "generic")
    assert skill[1:4] == ["-m", "mlx_lm", "lora"]
    skill_config = yaml.safe_load(Path(skill[-1]).read_text())
    generic_config = yaml.safe_load(Path(generic[-1]).read_text())
    assert skill_config["mask_prompt"] is True
    assert skill_config["lora_parameters"]["rank"] == 8
    assert generic_config["lora_parameters"]["rank"] == 24
    assert Path(skill_config["adapter_path"]).name == "adapter"
