from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow

if TYPE_CHECKING:
    from pathlib import Path

    from config.settings import Settings


def run_gui(settings: Settings, initial_directory: Path | None = None) -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow(settings=settings, initial_directory=initial_directory)
    window.show()
    app.exec()
