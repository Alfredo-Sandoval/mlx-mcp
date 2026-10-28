"""The single release manifest for MLX LM build inputs and evidence locations."""

import json
import re
from dataclasses import dataclass, fields
from pathlib import Path, PurePosixPath
from typing import cast


@dataclass(frozen=True)
class LmSource:
    version: str
    commit: str
    guides: tuple[str, ...]
    wheel_filename: str
    wheel_url: str
    capture_filename: str
    capture_tag: str
    capture_repository: str
    capture_path: str

    def __post_init__(self) -> None:
        if not all(
            (
                self.version,
                self.wheel_filename,
                self.wheel_url,
                self.capture_filename,
                self.capture_repository,
                self.capture_path,
            )
        ):
            raise ValueError("MLX LM manifest requires nonempty strings")
        if not re.fullmatch(r"[a-f0-9]{40}", self.commit):
            raise ValueError("MLX LM manifest has an invalid revision")
        if not re.fullmatch(r"v\d+\.\d+\.\d+", self.capture_tag):
            raise ValueError("MLX LM capture requires a release tag")
        if not self.guides or len(set(self.guides)) != len(self.guides):
            raise ValueError("MLX LM guide paths must be nonempty and unique")
        for path in (*self.guides, self.capture_filename, self.capture_path):
            if not re.fullmatch(r"[\w./-]+", path) or (
                PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts
            ):
                raise ValueError("MLX LM manifest contains an invalid relative path")
        if self.wheel_filename != f"mlx_lm-{self.version}-py3-none-any.whl" or not (
            self.wheel_url.startswith("https://files.pythonhosted.org/packages/")
            and self.wheel_url.endswith("/" + self.wheel_filename)
        ):
            raise ValueError("MLX LM wheel URL and version disagree")
        if not re.fullmatch(
            r"https://github.com/[\w-]+/[\w-]+", self.capture_repository
        ):
            raise ValueError("MLX LM capture requires a GitHub repository URL")

    @property
    def raw_base(self) -> str:
        return f"https://raw.githubusercontent.com/ml-explore/mlx-lm/{self.commit}/"

    @property
    def blob_base(self) -> str:
        return f"https://github.com/ml-explore/mlx-lm/blob/{self.commit}/"

    @property
    def capture_url(self) -> str:
        """Cite the release tag without requiring a commit to contain its own ID."""
        return f"{self.capture_repository}/blob/{self.capture_tag}/{self.capture_path}"

    @property
    def capture_file(self) -> Path:
        return Path(__file__).with_name("source_data") / self.capture_filename

    @property
    def package_url(self) -> str:
        return f"https://pypi.org/project/mlx-lm/{self.version}/"


def load_source() -> LmSource:
    """Load and validate the shipped source_data/mlx_lm.json manifest."""
    decoded: object = json.loads(
        Path(__file__).with_name("source_data").joinpath("mlx_lm.json").read_text()
    )
    if not isinstance(decoded, dict):
        raise ValueError("Unsupported MLX LM source manifest schema")  # noqa: TRY004
    manifest = dict(cast(dict[str, object], decoded))
    if manifest.pop("schema_version", None) != 1:
        raise ValueError("Unsupported MLX LM source manifest schema")
    guides = manifest.pop("guides", None)
    items = cast(list[object], guides) if isinstance(guides, list) else []
    paths = tuple(item for item in items if isinstance(item, str))
    if not isinstance(guides, list) or len(paths) != len(items):
        raise ValueError("MLX LM guides must be a JSON array of paths")
    names = [field.name for field in fields(LmSource) if field.name != "guides"]
    if set(manifest) != set(names):
        raise ValueError("MLX LM manifest fields do not match the source schema")
    values = {name: manifest[name] for name in names}
    text = {name: value for name, value in values.items() if isinstance(value, str)}
    if len(text) != len(values):
        raise ValueError("MLX LM manifest requires nonempty strings")
    return LmSource(guides=paths, **text)


LM_SOURCE = load_source()
