from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path("settings.json")


@dataclass
class Settings:
    scan_recursive: bool = True
    time_threshold_seconds: int = 3
    hash_threshold: int = 5
    min_group_size: int = 2
    duplicate_directory: str = "_duplicates"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Settings:
        return cls(**data)


def load_settings(path: Path = DEFAULT_CONFIG_PATH) -> Settings:
    if not path.exists():
        return Settings()
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return Settings.from_dict(data)


def save_settings(settings: Settings, path: Path = DEFAULT_CONFIG_PATH) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(settings.to_dict(), handle, indent=2)
