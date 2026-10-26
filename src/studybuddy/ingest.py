import hashlib
import json
from pathlib import Path

MAX_FAILURES = 3


def snapshot(folder: Path) -> dict[str, tuple[int, float]]:
    if not folder.exists():
        return {}
    return {
        p.name: (p.stat().st_size, p.stat().st_mtime)
        for p in sorted(folder.glob("*.pdf"))
    }


def stable_names(prev: dict, cur: dict) -> set[str]:
    return {
        name for name, stat in cur.items() if prev.get(name) == stat
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def plan_rebuild(
    folder: Path, manifest: dict, names: set[str] | None = None
) -> tuple[list[Path], dict]:
    files: dict[str, dict] = {}
    changed = False
    selected: list[Path] = []
    for path in sorted(folder.glob("*.pdf")):
        if names is not None and path.name not in names:
            files[path.name] = manifest.get("files", {}).get(path.name, {})
            continue
        entry = dict(manifest.get("files", {}).get(path.name, {}))
        sha = sha256_file(path)
        if entry.get("sha") != sha:
            entry = {"sha": sha, "failures": 0, "quarantined": False, "error": None}
            changed = True
        files[path.name] = entry
        selected.append(path)

    removed = set(manifest.get("files", {})) - set(files)
    if removed or manifest.get("collection") is None:
        changed = True

    next_manifest = dict(manifest)
    next_manifest["files"] = files
    if not changed:
        return [], next_manifest
    return [p for p in selected if not files[p.name].get("quarantined")], next_manifest


def record_failure(manifest: dict, name: str, error: str) -> None:
    entry = manifest.setdefault("files", {}).setdefault(
        name, {"sha": None, "failures": 0, "quarantined": False, "error": None}
    )
    entry["failures"] = entry.get("failures", 0) + 1
    entry["error"] = error
    if entry["failures"] >= MAX_FAILURES:
        entry["quarantined"] = True


def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {"collection": None, "files": {}}
    return json.loads(path.read_text())


def save_manifest(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))
