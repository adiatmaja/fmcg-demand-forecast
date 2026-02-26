import pytest


@pytest.mark.unit
def test_get_data_dirs_returns_expected_keys():
    from fmcg_forecast.utils.paths import get_data_dirs

    dirs = get_data_dirs()
    expected_keys = {
        "data",
        "logs",
        "demand_raw",
        "demand_preprocessed",
        "demand_forecast",
        "sales_raw",
        "sales_forecast",
        "sales_rec_buy",
        "synthetic",
    }
    assert set(dirs.keys()) == expected_keys


@pytest.mark.unit
def test_get_data_dirs_with_custom_root(tmp_path):
    from fmcg_forecast.utils.paths import get_data_dirs

    dirs = get_data_dirs(project_root=tmp_path)
    assert dirs["data"] == tmp_path / "data"
    assert dirs["synthetic"] == tmp_path / "data" / "synthetic"


@pytest.mark.unit
def test_ensure_dirs_creates_directories(tmp_path):
    from fmcg_forecast.utils.paths import ensure_dirs, get_data_dirs

    dirs = get_data_dirs(project_root=tmp_path)
    ensure_dirs(dirs)
    for path in dirs.values():
        assert path.exists()
