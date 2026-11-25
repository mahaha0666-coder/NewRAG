"""Resolve local Python imports for narrow snapshots and retirement audits."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

__all__ = ["closure", "dependencies", "modules", "strategy_fingerprint"]


def strategy_fingerprint(entrypoint: Path, repo: Path, parameters: dict, inputs: dict) -> str:
    """Ignore comments, file names, and scheduler changes when detecting exact retries."""
    syntax = sorted(ast.dump(ast.parse(path.read_text(encoding="utf-8")), include_attributes=False)
                    for path in closure([entrypoint], repo))
    payload = {"syntax": syntax, "parameters": parameters, "inputs": inputs}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def modules(repo: Path) -> dict[str, Path]:
    result = {}
    for directory in ("src", "scripts", "research", "tests"):
        for path in (repo / directory).rglob("*.py"):
            relative = path.relative_to(repo).with_suffix("")
            parts = relative.parts[:-1] if relative.name == "__init__" else relative.parts
            result[".".join(parts)] = path
    return result


def dependencies(path: Path, index: dict[str, Path], repo: Path) -> set[Path]:
    relative = path.relative_to(repo).with_suffix("")
    package = list(relative.parts[:-1])
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = package[:len(package) - node.level + 1] if node.level else []
            base = ".".join([*prefix, *((node.module or "").split(".") if node.module else [])])
            names.add(base)
            names.update(f"{base}.{alias.name}" for alias in node.names if alias.name != "*")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in index:
                names.add(node.value)
            elif node.value.endswith(".py"):
                candidate = node.value.removesuffix(".py").replace("/", ".")
                if candidate in index:
                    names.add(candidate)
    result = {index[name] for name in names if name in index}
    for depth in range(1, len(relative.parts)):
        parent = ".".join(relative.parts[:depth])
        if parent in index and index[parent].name == "__init__.py":
            result.add(index[parent])
    return result


def closure(roots: list[Path], repo: Path) -> set[Path]:
    index = modules(repo)
    seen, pending = set(), list(roots)
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        if not path.is_file():
            raise FileNotFoundError(path)
        seen.add(path)
        pending.extend(dependencies(path, index, repo) - seen)
    return seen
