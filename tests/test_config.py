import pytest


@pytest.mark.unit
def test_demand_config_has_defaults():
    from fmcg_forecast.config import DemandConfig

    cfg = DemandConfig()
    assert cfg.input_window == 21
    assert cfg.forecast_horizon == 60
    assert cfg.epochs == 500
    assert cfg.batch_size == 128
    assert cfg.quantiles == [0.5]


@pytest.mark.unit
def test_sales_config_has_defaults():
    from fmcg_forecast.config import SalesConfig

    cfg = SalesConfig()
    assert cfg.input_window == 30
    assert cfg.forecast_horizon == 90
    assert cfg.epochs == 200


@pytest.mark.unit
def test_load_yaml_config(tmp_path):
    from fmcg_forecast.config import load_yaml_config

    yaml_file = tmp_path / "test.yaml"
    yaml_file.write_text("demand:\n  input_window: 14\n  epochs: 100\n")
    cfg = load_yaml_config(str(yaml_file))
    assert cfg["demand"]["input_window"] == 14
    assert cfg["demand"]["epochs"] == 100


@pytest.mark.unit
def test_model_config_has_defaults():
    from fmcg_forecast.config import ModelConfig

    cfg = ModelConfig()
    assert cfg.hidden_dim == 24
    assert cfg.num_layers == 4
    assert cfg.learning_rate == 0.0001
    assert cfg.dropout == 0.25
    assert cfg.product_embedding_dim == 16
    assert cfg.gudang_embedding_dim == 8
    assert cfg.early_stopping_patience == 25
