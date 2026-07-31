from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

MARKDOWN_ROOTS = (
    Path("README.md"),
    Path("docs"),
    Path("src"),
    Path(".agents"),
)

_MARKDOWN_LINK = re.compile(
    r"!?\[[^\]]*\]\(\s*"
    r"(?P<target><[^>]+>|[^\s)]+)"
    r"(?:\s+(?:\"[^\"]*\"|'[^']*'|\([^)]*\)))?\s*\)"
)
_EXTERNAL_PREFIXES = ("http:", "https:", "mailto:")


def _markdown_files() -> tuple[Path, ...]:
    files: list[Path] = []
    for root in MARKDOWN_ROOTS:
        if root.is_file():
            files.append(root)
        elif root.is_dir():
            files.extend(root.rglob("*.md"))
    return tuple(sorted(files))


def _local_target(raw_target: str) -> str | None:
    target = raw_target[1:-1] if raw_target.startswith("<") else raw_target
    if target.startswith((*_EXTERNAL_PREFIXES, "#")):
        return None
    return unquote(target.partition("#")[0])


def test_local_markdown_links_resolve() -> None:
    missing: list[str] = []
    for source in _markdown_files():
        contents = source.read_text(encoding="utf-8")
        for match in _MARKDOWN_LINK.finditer(contents):
            raw_target = match.group("target")
            target = _local_target(raw_target)
            if not target:
                continue
            if not (source.parent / target).resolve().exists():
                missing.append(f"{source.as_posix()} -> {raw_target}")

    assert sorted(missing) == []
