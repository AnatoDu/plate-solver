"""Ворота депроцессизации (F1.6): документация — язык продукта.

Стоп-словарь процессных привязок (внутренние рукописи, фазы разработки)
не должен встречаться в отслеживаемых текстовых файлах. Белый список —
явным перечнем: исторические примечания CHANGELOG и сам словарь. Научная библиография на
опубликованные внешние источники сохраняется — она формулируется без
стоп-слов («в публикации: …»), что и проверяется этим тестом.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: процессные привязки; \b — юникодные границы слов (не байтовые, как grep)
STOP = re.compile(
    r"диссерта|глав[аеы]\b|стать[яеи]\b|доклад|фаз[аы]\b|трек\b|"
    r"ЮСНВ|ВМСС|Вестник|P[0-9]\.[0-9]|положение на защиту"
)

#: до миграции эталонного отчёта (F5.3) golden-семейство заморожено как есть
WHITELIST = {
    "CHANGELOG.md",                  # исторические примечания релизов
    "tests/test_doc_policy.py",      # сам словарь
}

EXTS = (".py", ".md", ".toml", ".json", ".ipynb", ".cfg", ".txt",
        ".yml", ".yaml", ".mmd", ".cff")


def _tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT,
                         capture_output=True, text=True)
    if out.returncode != 0:                      # вне git (например, sdist)
        pytest.skip("git недоступен — политика проверяется в репозитории")
    return out.stdout.split()


def test_no_process_stopwords_in_tracked_docs():
    bad: list[str] = []
    for rel in _tracked_files():
        if rel in WHITELIST or not rel.endswith(EXTS):
            continue
        text = (ROOT / rel).read_text(encoding="utf-8", errors="replace")
        for i, line in enumerate(text.splitlines(), 1):
            m = STOP.search(line)
            if m:
                bad.append(f"{rel}:{i}: [{m.group(0)}] {line.strip()[:80]}")
    assert not bad, "стоп-словарь процесса найден:\n" + "\n".join(bad)


def test_private_dir_is_never_tracked():
    """Контроль F0.5: домашние материалы не попадают в индекс."""
    out = subprocess.run(["git", "ls-files", "private/"], cwd=ROOT,
                         capture_output=True, text=True)
    if out.returncode != 0:
        pytest.skip("git недоступен")
    assert out.stdout.strip() == ""
