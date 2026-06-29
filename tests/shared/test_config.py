from slmcortex.shared.config import CONFIG_DIR, base_config


def test_base_config_can_use_profile_path(tmp_path, monkeypatch):
    profile = tmp_path / "prototype.yaml"
    profile.write_text("model: test-model\ntraining_enabled: true\n")
    monkeypatch.setenv("SLMCORTEX_BASE_CONFIG", str(profile))

    assert base_config()["model"] == "test-model"
    assert base_config()["training_enabled"] is True


def test_default_config_dir_uses_packaged_resources():
    assert CONFIG_DIR.name == "configs"
    assert CONFIG_DIR.parent.name == "slmcortex_resources"
