"""Canonical identities and published documentation locations of indexed packages."""

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

# Search and browse filter; "all" admits both API languages.
Language = Literal["all", "python", "cpp"]


class Package(StrEnum):
    """An indexed package, identified by its distribution name."""

    MLX = "mlx"
    DATA = "mlx-data"
    LM = "mlx-lm"

    @property
    def label(self) -> str:
        """Return the display name that page titles and responses use."""
        return _PACKAGES[self].label

    @property
    def base_url(self) -> str:
        """Return the root URL of the package's published docs or source repository."""
        return _PACKAGES[self].base_url

    @property
    def sphinx(self) -> bool:
        """Return whether the package publishes a crawlable Sphinx site."""
        return _PACKAGES[self].sphinx

    @property
    def titled_version(self) -> bool:
        """Return whether the site's page titles name the documented release."""
        return _PACKAGES[self].titled_version

    def accepts_url(self, url: str) -> bool:
        """Return whether url is a plain page path on this package's Sphinx site.

        Query strings, fragments, percent-encoding, and dot segments are rejected.
        """
        path = url.removeprefix(self.base_url)
        return (
            self.sphinx
            and url.startswith(self.base_url)
            and re.fullmatch(r"[\w./-]+", path) is not None
            and all(part not in {".", ".."} for part in path.split("/"))
        )


@dataclass(frozen=True)
class PackageMetadata:
    """Display name and source location for one Package."""

    label: str
    base_url: str
    sphinx: bool
    titled_version: bool


_PACKAGES = {
    Package.MLX: PackageMetadata(
        "MLX", "https://ml-explore.github.io/mlx/build/html/", True, True
    ),
    Package.DATA: PackageMetadata(
        "MLX Data",
        "https://ml-explore.github.io/mlx-data/build/html/",
        True,
        # The site's config still says 0.0.2, but it was rebuilt for 0.2.0 on
        # 2025-09-26, so its titles name no usable release.
        False,
    ),
    Package.LM: PackageMetadata(
        "MLX LM", "https://github.com/ml-explore/mlx-lm/", False, False
    ),
}


class EvidenceKind(StrEnum):
    """The kind of source a passage quotes; search and fetch label it."""

    DOCUMENTATION = "published documentation"
    GUIDE = "guide text"
    API = "extracted API reference"
    CLI = "captured CLI output"
