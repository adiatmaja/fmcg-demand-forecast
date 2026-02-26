"""Centralized logging configuration."""
import logging
import sys


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: str | None = None,
) -> logging.Logger:
    """Create a configured logger with console and optional file output.

    Args:
        name: Logger name (typically __name__).
        level: Logging level.
        log_file: Optional path to log file.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger
