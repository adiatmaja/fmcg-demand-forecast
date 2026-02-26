import logging

import pytest


@pytest.mark.unit
def test_setup_logger_returns_configured_logger():
    from fmcg_forecast.utils.logging import setup_logger

    logger = setup_logger("test_logger", level=logging.DEBUG)
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger"
    assert logger.level == logging.DEBUG


@pytest.mark.unit
def test_setup_logger_has_console_handler():
    from fmcg_forecast.utils.logging import setup_logger

    logger = setup_logger("test_console", level=logging.INFO)
    handler_types = [type(h) for h in logger.handlers]
    assert logging.StreamHandler in handler_types


@pytest.mark.unit
def test_setup_logger_with_file_handler(tmp_path):
    from fmcg_forecast.utils.logging import setup_logger

    log_file = tmp_path / "test.log"
    logger = setup_logger("test_file", log_file=str(log_file))
    logger.info("test message")
    assert log_file.exists()
