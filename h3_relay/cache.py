"""Managed H3 Relay cache and preview publication helpers."""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

import folder_paths


CACHE_SCHEME = "cache://"
TEMP_SCHEME = "temp://"
REVISION_RE = re.compile(r"clip_(\d{4})\.([0-9a-f]{32})(?:\.|$)")
HEX_REVISION_RE = re.compile(r"\b[0-9a-f]{32}\b")


def cache_root() -> str:
    if hasattr(folder_paths, "get_system_user_directory"):
        root = folder_paths.get_system_user_directory("h3_relay_cache")
    else:
        root = os.path.join(
            folder_paths.get_user_directory(), "__h3_relay_cache"
        )
    return os.path.abspath(root)


def cache_path(*parts: str) -> str:
    root = cache_root()
    path = os.path.abspath(os.path.join(root, *[str(part) for part in parts]))
    if os.path.commonpath([root, path]) != root:
        raise ValueError("H3 Relay cache path escapes its managed root.")
    return path


def _root_relative(path: str, root: str) -> str | None:
    resolved = os.path.abspath(path)
    try:
        if os.path.commonpath([root, resolved]) != root:
            return None
    except ValueError:
        return None
    return os.path.relpath(resolved, root)


def artifact_uri(path: str) -> str:
    resolved = os.path.abspath(path)
    relative = _root_relative(resolved, cache_root())
    if relative is not None:
        return CACHE_SCHEME + relative.replace(os.sep, "/")
    relative = _root_relative(
        resolved, os.path.abspath(folder_paths.get_temp_directory())
    )
    if relative is not None:
        return TEMP_SCHEME + relative.replace(os.sep, "/")
    output = os.path.abspath(folder_paths.get_output_directory())
    relative = _root_relative(resolved, output)
    if relative is None:
        raise ValueError("H3 Relay artifact is outside cache, temp, and output roots.")
    return relative


def resolve_artifact(value: str) -> str:
    raw = str(value)
    if raw.startswith(CACHE_SCHEME):
        return cache_path(*raw[len(CACHE_SCHEME):].split("/"))
    if raw.startswith(TEMP_SCHEME):
        root = os.path.abspath(folder_paths.get_temp_directory())
        path = os.path.abspath(os.path.join(
            root, *raw[len(TEMP_SCHEME):].split("/")
        ))
        if os.path.commonpath([root, path]) != root:
            raise ValueError("H3 Relay temp artifact escapes the temp root.")
        return path
    if os.path.isabs(raw):
        resolved = os.path.abspath(raw)
        allowed = [
            cache_root(),
            os.path.abspath(folder_paths.get_temp_directory()),
            os.path.abspath(folder_paths.get_output_directory()),
        ]
        if not any(_root_relative(resolved, root) is not None for root in allowed):
            raise ValueError("H3 Relay absolute artifact path is unmanaged.")
        return resolved
    output = os.path.abspath(folder_paths.get_output_directory())
    resolved = os.path.abspath(os.path.join(output, raw))
    if os.path.commonpath([output, resolved]) != output:
        raise ValueError("H3 Relay legacy output artifact escapes output root.")
    return resolved


def _preview_path(cache_file: str) -> str:
    relative = _root_relative(cache_file, cache_root())
    if relative is None:
        raise ValueError("Preview source is not in the managed cache.")
    root = os.path.abspath(folder_paths.get_temp_directory())
    path = os.path.abspath(os.path.join(root, "h3_relay", "cache", relative))
    if os.path.commonpath([root, path]) != root:
        raise ValueError("H3 Relay preview path escapes the temp root.")
    return path


def preview_proxy(path: str) -> tuple[str, str]:
    resolved = os.path.abspath(path)
    output = os.path.abspath(folder_paths.get_output_directory())
    if _root_relative(resolved, output) is not None:
        return resolved, "output"
    temp = os.path.abspath(folder_paths.get_temp_directory())
    if _root_relative(resolved, temp) is not None:
        return resolved, "temp"
    if _root_relative(resolved, cache_root()) is None:
        raise ValueError("H3 Relay preview source is outside managed roots.")
    proxy = _preview_path(resolved)
    os.makedirs(os.path.dirname(proxy), exist_ok=True)
    try:
        if (
            os.path.isfile(proxy)
            and os.path.getsize(proxy) == os.path.getsize(resolved)
            and os.stat(proxy).st_ino == os.stat(resolved).st_ino
        ):
            return proxy, "temp"
    except OSError:
        pass
    try:
        if os.path.lexists(proxy):
            os.unlink(proxy)
        os.link(resolved, proxy)
    except OSError:
        shutil.copy2(resolved, proxy)
    return proxy, "temp"


def video_output_item(path: str) -> dict[str, str]:
    proxy, kind = preview_proxy(path)
    root = (
        folder_paths.get_output_directory()
        if kind == "output" else folder_paths.get_temp_directory()
    )
    relative = os.path.relpath(proxy, os.path.abspath(root))
    return {
        "filename": os.path.basename(relative),
        "subfolder": os.path.dirname(relative),
        "type": kind,
    }


def _logical_revision_group(path: Path, shot: str) -> str:
    relative = path.relative_to(Path(cache_root()))
    parts = relative.parts
    if len(parts) >= 3 and parts[0] == "h3_relay":
        return "/".join(parts[:3]) + "/" + shot
    if len(parts) >= 2 and parts[0] == "h3_chains":
        stage = "h3"
        if "enhanced_ltx" in parts:
            stage = "enhanced_ltx"
        elif "enhanced" in parts:
            stage = "enhanced"
        return "/".join(parts[:2]) + "/" + stage + "/" + shot
    return "/".join(parts[:-1]) + "/" + shot


def _revision_groups() -> dict[tuple[str, str], list[Path]]:
    root = Path(cache_root())
    groups: dict[tuple[str, str], list[Path]] = {}
    if not root.is_dir():
        return groups
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        match = REVISION_RE.search(path.name)
        if match is None:
            continue
        shot, revision = match.groups()
        key = (_logical_revision_group(path, shot), revision)
        groups.setdefault(key, []).append(path)
    return groups


def _protected_revisions() -> set[str]:
    protected: set[str] = set()
    root = Path(cache_root())
    if not root.is_dir():
        return protected
    for path in root.rglob("*.json"):
        if REVISION_RE.search(path.name):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        protected.update(HEX_REVISION_RE.findall(text))
    return protected


def cache_stats() -> dict[str, int]:
    root = Path(cache_root())
    files = 0
    size = 0
    if root.is_dir():
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            try:
                size += path.stat().st_size
                files += 1
            except OSError:
                continue
    return {
        "bytes": size,
        "files": files,
        "revision_groups": len(_revision_groups()),
    }


def _unlink_cache_file(path: Path) -> int:
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    try:
        proxy = Path(_preview_path(str(path)))
        if proxy.exists() or proxy.is_symlink():
            proxy.unlink()
    except (OSError, ValueError):
        pass
    try:
        path.unlink()
        return size
    except OSError:
        return 0


def prune_superseded(keep_per_shot: int = 2) -> dict[str, int]:
    keep = max(1, int(keep_per_shot))
    protected = _protected_revisions()
    grouped: dict[str, list[tuple[str, list[Path], float]]] = {}
    for (logical, revision), paths in _revision_groups().items():
        modified = max(
            (path.stat().st_mtime for path in paths if path.exists()),
            default=0.0,
        )
        grouped.setdefault(logical, []).append((revision, paths, modified))
    removed_files = 0
    removed_bytes = 0
    removed_revisions = 0
    for revisions in grouped.values():
        revisions.sort(key=lambda item: item[2], reverse=True)
        retained = {item[0] for item in revisions[:keep]} | protected
        for revision, paths, _ in revisions:
            if revision in retained:
                continue
            removed_revisions += 1
            for path in paths:
                removed = _unlink_cache_file(path)
                if removed:
                    removed_files += 1
                    removed_bytes += removed
    return {
        "removed_bytes": removed_bytes,
        "removed_files": removed_files,
        "removed_revisions": removed_revisions,
    }


def maybe_prune_run(keep_per_shot: int = 2) -> None:
    prune_superseded(keep_per_shot)


def format_bytes(value: int) -> str:
    size = float(max(0, int(value)))
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return "%.2f %s" % (size, unit)
        size /= 1024.0
    return "%.2f TiB" % size


__all__ = [
    "artifact_uri",
    "cache_path",
    "cache_root",
    "cache_stats",
    "format_bytes",
    "maybe_prune_run",
    "preview_proxy",
    "prune_superseded",
    "resolve_artifact",
    "video_output_item",
]
