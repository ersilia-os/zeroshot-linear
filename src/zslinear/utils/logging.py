import sys
from typing import Optional

from rich.console import Console
from rich import box
from loguru import logger as _loguru

_loguru.remove()

_FORMAT = (
    "<green>{time:HH:mm:ss}</green> "
    "<level>{level: <8}</level> "
    "{message}"
)


class Logger:
    def __init__(self):
        self._loguru = _loguru
        self._console = Console(stderr=True, highlight=False)
        self._sink_id: Optional[int] = None
        self._verbose = False

    def set_verbosity(self, verbose: bool):
        self._verbose = verbose
        if verbose and self._sink_id is None:
            self._sink_id = self._loguru.add(
                sys.stderr,
                format=_FORMAT,
                colorize=True,
                level="DEBUG",
            )
        elif not verbose and self._sink_id is not None:
            try:
                self._loguru.remove(self._sink_id)
            except Exception:
                pass
            self._sink_id = None

    def debug(self, msg: str):
        self._loguru.debug(msg)

    def info(self, msg: str):
        self._loguru.info(msg)

    def warning(self, msg: str):
        self._loguru.warning(msg)

    def error(self, msg: str):
        self._loguru.error(msg)

    def success(self, msg: str):
        self._loguru.success(msg)

    def rule(self, title: str = "", style: str = "dim blue"):
        if not self._verbose:
            return
        if title:
            self._console.rule(f"[bold cyan]{title}[/]", style=style)
        else:
            self._console.rule(style=style)


logger = Logger()
