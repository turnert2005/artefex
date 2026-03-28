"""Configuration system - loads settings from .artefex.toml or pyproject.toml."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None


@dataclass
class ArtefexConfig:
    """Artefex configuration loaded from config files."""

    # Analysis
    detectors: list[str] = field(default_factory=list)  # empty = all
    min_confidence: float = 0.15
    min_severity: float = 0.0

    # Restoration
    use_neural: bool = True
    output_format: str = ""  # empty = same as input
    output_quality: int = 95

    # Web
    web_host: str = "127.0.0.1"
    web_port: int = 8787

    # Watch
    watch_interval: float = 2.0
    watch_auto_restore: bool = False

    # Output
    default_output_dir: str = ""
    verbose: bool = False


def load_config(start_dir: Optional[Path] = None) -> ArtefexConfig:
    """Load config by searching for .artefex.toml or [tool.artefex] in pyproject.toml.

    Search order:
    1. .artefex.toml in start_dir and parents
    2. pyproject.toml [tool.artefex] in start_dir and parents
    3. ~/.artefex.toml (global config)
    4. Defaults
    """
    if tomllib is None:
        return ArtefexConfig()

    start = start_dir or Path.cwd()

    # Search up from start_dir
    for directory in [start] + list(start.parents):
        # Check .artefex.toml
        config_path = directory / ".artefex.toml"
        if config_path.exists():
            return _load_from_file(config_path)

        # Check pyproject.toml
        pyproject_path = directory / "pyproject.toml"
        if pyproject_path.exists():
            config = _load_from_pyproject(pyproject_path)
            if config is not None:
                return config

    # Check global config
    global_config = Path.home() / ".artefex.toml"
    if global_config.exists():
        return _load_from_file(global_config)

    return ArtefexConfig()


def _load_from_file(path: Path) -> ArtefexConfig:
    """Load config from a .artefex.toml file."""
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return _parse_config(data)


def _load_from_pyproject(path: Path) -> Optional[ArtefexConfig]:
    """Load config from [tool.artefex] in pyproject.toml."""
    with open(path, "rb") as f:
        data = tomllib.load(f)

    artefex_data = data.get("tool", {}).get("artefex")
    if artefex_data is None:
        return None

    return _parse_config(artefex_data)


def _parse_config(data: dict) -> ArtefexConfig:
    """Parse a config dict into an ArtefexConfig."""
    config = ArtefexConfig()

    analysis = data.get("analysis", {})
    config.detectors = analysis.get("detectors", config.detectors)
    config.min_confidence = analysis.get("min_confidence", config.min_confidence)
    config.min_severity = analysis.get("min_severity", config.min_severity)

    restore = data.get("restore", {})
    config.use_neural = restore.get("use_neural", config.use_neural)
    config.output_format = restore.get("output_format", config.output_format)
    config.output_quality = restore.get("output_quality", config.output_quality)

    web = data.get("web", {})
    config.web_host = web.get("host", config.web_host)
    config.web_port = web.get("port", config.web_port)

    watch = data.get("watch", {})
    config.watch_interval = watch.get("interval", config.watch_interval)
    config.watch_auto_restore = watch.get("auto_restore", config.watch_auto_restore)

    output = data.get("output", {})
    config.default_output_dir = output.get("dir", config.default_output_dir)
    config.verbose = output.get("verbose", config.verbose)

    return config
