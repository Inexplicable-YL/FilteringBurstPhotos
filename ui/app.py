from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import QApplication

from config.settings import Settings
from ui.main_window import MainWindow


def run_gui(settings: Settings, initial_directory: Optional[Path] = None) -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow(settings=settings, initial_directory=initial_directory)
    window.show()
    app.exec()
