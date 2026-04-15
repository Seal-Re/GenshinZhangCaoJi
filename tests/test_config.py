from config.loader import load_config, Settings, GameConfig

def test_load_default_config():
    cfg = load_config()
    assert isinstance(cfg, Settings)
    assert cfg.llm.default == "claude-sonnet-4-6"

def test_minimap_region_is_four_ints():
    cfg = load_config()
    r = cfg.game.minimap_region
    assert len(r) == 4
    assert all(isinstance(v, int) for v in r)

def test_game_window_title():
    cfg = load_config()
    assert cfg.game.window_title == "原神"
