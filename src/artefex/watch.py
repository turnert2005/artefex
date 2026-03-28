"""Directory watcher - monitors a folder and auto-analyzes new images."""

import time
from pathlib import Path
from typing import Optional

from artefex.analyze import DegradationAnalyzer
from artefex.restore import RestorationPipeline

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif"}


class DirectoryWatcher:
    """Watches a directory for new images and processes them."""

    def __init__(
        self,
        watch_dir: Path,
        output_dir: Optional[Path] = None,
        auto_restore: bool = False,
        use_neural: bool = True,
    ):
        self.watch_dir = watch_dir
        self.output_dir = output_dir or watch_dir / "artefex_output"
        self.auto_restore = auto_restore
        self.analyzer = DegradationAnalyzer()
        self.pipeline = RestorationPipeline(use_neural=use_neural) if auto_restore else None
        self._seen: set[str] = set()

    def _scan(self) -> list[Path]:
        """Find new image files that haven't been processed yet."""
        new_files = []
        for f in self.watch_dir.iterdir():
            if f.suffix.lower() in IMAGE_EXTENSIONS and str(f) not in self._seen:
                # Skip files in our output directory
                if self.output_dir in f.parents:
                    continue
                new_files.append(f)
        return new_files

    def _process(self, file: Path, on_result=None):
        """Analyze (and optionally restore) a single file."""
        self._seen.add(str(file))

        result = self.analyzer.analyze(file)

        info = {
            "file": file.name,
            "degradations": len(result.degradations),
            "overall_severity": result.overall_severity,
            "top_issue": result.degradations[0].name if result.degradations else None,
            "restored": False,
        }

        if self.auto_restore and result.degradations:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            out_path = self.output_dir / f"{file.stem}_restored{file.suffix}"
            self.pipeline.restore(file, result, out_path)
            info["restored"] = True
            info["restored_path"] = str(out_path)

        if on_result:
            on_result(info)

        return info

    def run(self, interval: float = 2.0, on_result=None, on_scan=None):
        """Start watching the directory. Blocks until interrupted.

        Args:
            interval: Seconds between scans.
            on_result: Callback(info_dict) for each processed file.
            on_scan: Callback() on each scan cycle.
        """
        # Initial scan to mark existing files as seen
        for f in self.watch_dir.iterdir():
            if f.suffix.lower() in IMAGE_EXTENSIONS:
                self._seen.add(str(f))

        while True:
            if on_scan:
                on_scan()

            new_files = self._scan()
            for f in new_files:
                self._process(f, on_result=on_result)

            time.sleep(interval)
