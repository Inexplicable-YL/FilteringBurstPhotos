from __future__ import annotations

from config.settings import load_settings, save_settings
from ui.app import run_gui


def main() -> None:
    settings = load_settings()
    run_gui(settings, initial_directory=None)
    save_settings(settings)


if __name__ == "__main__":
    main()
