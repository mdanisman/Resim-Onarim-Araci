from __future__ import annotations

"""Merkezi log altyapısı.

- Tek logger ("resim_onarim") kullanılır.
- RotatingFileHandler ile dosya boyutu yönetilir.
- Format, işlem kimliği, adım, dosya, yöntem, sonuç ve süre (ms) alanlarını içerir.
- LoggerAdapter ile eksik alanlar otomatik olarak "-" ile doldurulur.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Optional

LOGGER_NAME = "resim_onarim"
LOG_FILE_NAME = "resim_onarim.log"
REQUIRED_FIELDS = [
    "operation_id",
    "step",
    "file",
    "method",
    "result",
    "duration_ms",
]


def _ensure_logger(log_dir: Optional[Path] = None) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    log_path = (log_dir or Path.cwd()) / LOG_FILE_NAME
    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [op:%(operation_id)s] [step:%(step)s] "
        "[file:%(file)s] [method:%(method)s] [result:%(result)s] "
        "[duration:%(duration_ms)s] %(message)s"
    )

    file_handler = RotatingFileHandler(
        log_path, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False
    return logger


class ContextLogger(logging.LoggerAdapter):
    def __init__(self, logger: logging.Logger, default_extra: Dict[str, str]):
        merged_default = {field: "-" for field in REQUIRED_FIELDS}
        merged_default.update(default_extra)
        super().__init__(logger, merged_default)

    def process(self, msg, kwargs):
        extra = kwargs.get("extra", {})
        merged = {**self.extra, **(extra or {})}
        for field in REQUIRED_FIELDS:
            merged.setdefault(field, "-")
        kwargs["extra"] = merged
        return msg, kwargs


def create_logger(
    operation_id: Optional[str] = None,
    step: str = "init",
    log_dir: Optional[Path] = None,
    **extra: str,
) -> ContextLogger:
    base_extra: Dict[str, str] = {
        "operation_id": operation_id or "-",
        "step": step,
        "file": "-",
        "method": "-",
        "result": "-",
        "duration_ms": "-",
    }
    base_extra.update(extra)
    logger = _ensure_logger(log_dir)
    return ContextLogger(logger, base_extra)