"""Archive pre-remediation CTI 01-13 outputs without deleting them."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ARCHIVE_NAME = "pre-remediation-2026-07-31"
PROJECTS = [f"{number:02d}" for number in range(1, 14)]
SYNTHETIC_DATA_PROJECTS = {"06", "07", "08", "09", "10", "12", "13"}


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_tree(source: Path, destination: Path, records: list[dict[str, str]]) -> None:
    """Move a file or directory into the archive and record its files."""
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Archive target already exists: {destination}")
    for path in source.rglob("*") if source.is_dir() else [source]:
        if path.is_file():
            records.append({
                "original": str(path),
                "archived": str(destination / path.relative_to(source)) if source.is_dir() else str(destination),
                "sha256": sha256_file(path),
            })
    shutil.move(str(source), str(destination))


def archive_project(root: Path, project: str) -> dict[str, object]:
    """Archive model/results and generated synthetic data for one project."""
    matches = list(root.glob(f"CTI LAB/{project}-*"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one project directory for {project}, found {matches}")
    project_dir = matches[0]
    archive_dir = project_dir / "legacy" / ARCHIVE_NAME
    records: list[dict[str, str]] = []

    results = project_dir / "results"
    archive_tree(results, archive_dir / "results", records)
    if project == "01":
        archive_tree(project_dir / "spam_text_model.joblib", archive_dir / "spam_text_model.joblib", records)
    if project in SYNTHETIC_DATA_PROJECTS:
        archive_tree(project_dir / "data", archive_dir / "data", records)

    archive_dir.mkdir(parents=True, exist_ok=True)
    warning = (
        "These artifacts were generated before the CTI ML validity remediation. "
        "They are preserved for comparison only and must not be presented as "
        "generalization evidence."
    )
    (archive_dir / "ARCHIVE_WARNING.md").write_text(f"# Legacy CTI artifacts\n\n{warning}\n", encoding="utf-8")
    manifest = {
        "archived_at": datetime.now(timezone.utc).isoformat(),
        "project": project_dir.name,
        "warning": warning,
        "files": records,
    }
    (archive_dir / "archive_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    """Archive selected CTI project artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--project", action="append", choices=PROJECTS)
    args = parser.parse_args()
    projects = args.project or PROJECTS
    for project in projects:
        manifest = archive_project(args.root, project)
        print(f"Archived {manifest['project']}: {len(manifest['files'])} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
