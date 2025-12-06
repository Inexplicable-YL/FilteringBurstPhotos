from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

DEFAULT_CONFIG_PATH = Path("settings.json")


class Settings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    scan_recursive: bool = True
    time_threshold_seconds: int = 3
    hash_threshold: int = 5
    min_group_size: int = 2

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Settings:
        return cls.model_validate(data)


def load_settings(path: Path = DEFAULT_CONFIG_PATH) -> Settings:
    if not path.exists():
        return Settings()
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return Settings.from_dict(data)


def save_settings(settings: Settings, path: Path = DEFAULT_CONFIG_PATH) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(settings.to_dict(), handle, indent=2)
