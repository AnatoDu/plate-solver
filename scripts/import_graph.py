#!/usr/bin/env python3
"""import_graph.py — mermaid-граф внутренних импортов пакета.

Печатает блок ``flowchart TD`` по фактическим ``from .X import`` /
``import plate_solver.X`` в src/plate_solver/*.py. Используется для
docs/ARCHITECTURE.md (вставка руками при изменении структуры).
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "plate_solver"


def edges() -> list[tuple[str, str]]:
    out = set()
    for py in sorted(SRC.glob("*.py")):
        mod = py.stem
        if mod == "__init__":
            continue
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
                dep = node.module.split(".")[0]
                if (SRC / f"{dep}.py").exists() and dep != mod:
                    out.add((mod, dep))
    return sorted(out)


def main() -> int:
    print("```mermaid")
    print("flowchart TD")
    for a, b in edges():
        print(f"    {a} --> {b}")
    print("```")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
