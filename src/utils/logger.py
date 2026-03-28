"""Logging configuration for Legal RAG Assistant."""

import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import settings

PROJECT_ROOT = Path(__file__).parent.parent.parent
LOG_DIR = PROJECT_ROOT / settings.log.dir
LOG_FILE = LOG_DIR / "legal-rag.log"

_DEFAULT_LEVEL = getattr(logging, settings.log.level.upper())


def setup_logger(name: str, level: int = _DEFAULT_LEVEL) -> logging.Logger:
    """Create a logger with file and console handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    LOG_DIR.mkdir(exist_ok=True)

    file_handler = TimedRotatingFileHandler(
        LOG_FILE,
        when="midnight",
        interval=1,
        backupCount=settings.log.backup_count,
    )
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_format)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(file_format)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


api_logger = setup_logger("api")
qdrant_logger = setup_logger("qdrant")
llm_logger = setup_logger("llm")
