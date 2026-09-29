"""Engine configuration. Plain env-var reads so there's no extra dependency."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    # Where the flywheel DB and generated artifacts live.
    data_dir: Path = Path(os.getenv("JUWON_DATA_DIR", Path(__file__).resolve().parent.parent / ".data"))
    # STAC endpoint for free Sentinel imagery (Element84 Earth Search).
    stac_url: str = os.getenv("JUWON_STAC_URL", "https://earth-search.aws.element84.com/v1")
    # Max acceptable cloud cover (%) when searching optical scenes.
    max_cloud_cover: float = float(os.getenv("JUWON_MAX_CLOUD", "20"))
    # Default provider when the caller doesn't specify: "fake" or "sentinel".
    default_provider: str = os.getenv("JUWON_PROVIDER", "sentinel")

    @property
    def db_path(self) -> Path:
        return self.data_dir / "flywheel.db"

    @property
    def artifacts_dir(self) -> Path:
        return self.data_dir / "artifacts"


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
