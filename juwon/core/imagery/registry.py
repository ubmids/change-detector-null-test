"""Provider factory. Remote providers are wrapped in the disk cache by
default; heavy deps (rasterio) stay lazy so the core never hard-requires them.
"""
from __future__ import annotations

from ...config import settings
from .base import ImageryProvider

PROVIDERS = ["fake", "sentinel", "sentinel1"]


def make_provider(name: str, *, cache: bool | None = None) -> ImageryProvider:
    if name == "fake":
        from .fake import FakeImageryProvider

        return FakeImageryProvider()  # synthetic — nothing worth caching

    if name == "sentinel":
        from .sentinel import SentinelProvider

        inner: ImageryProvider = SentinelProvider()
    elif name == "sentinel1":
        from .sentinel1 import Sentinel1Provider

        inner = Sentinel1Provider()
    else:
        raise KeyError(f"unknown provider {name!r}; available: {PROVIDERS}")

    if cache is False:
        return inner
    from .cache import CachingProvider

    return CachingProvider(inner, settings.data_dir / "cache")
