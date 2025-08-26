"""Тонкая прокся к единому модулю логирования src.utils.logger.

Оставляем единый источник правды: src.utils.logger.
Этот модуль сохраняет совместимость с существующими импортами
(`from src.config.logging import configure_logging, get_logger`).
"""
from __future__ import annotations

from logging import Logger
from typing import Optional

from src.utils.logger import setup_logging as _setup_logging
from src.utils.logger import get_logger as _get_logger


def configure_logging(force: bool = False) -> None:
    """Инициализация логирования через общий модуль.

    Параметр force сохранён для совместимости, но не используется —
    `setup_logging` сам идемпотентен для большинства случаев.
    """
    _setup_logging()


def get_logger(name: Optional[str] = None) -> Logger:
    """Вернуть логгер из общего модуля логирования."""
    return _get_logger(name)


__all__ = ["configure_logging", "get_logger"]
