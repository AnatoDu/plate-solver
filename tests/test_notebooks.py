"""Смок ноутбуков: case-файлы валидны, пути переносимы, выводы не протухли.

В быстром наборе ноутбуки НЕ исполняются (лишние dev-зависимости): гейт —
JSON корректен, каждый упомянутый case-файл проходит валидатор ``Problem``, а
пути к case-файлам записаны ОТНОСИТЕЛЬНО каталога ноутбуков (иначе ядро,
запущенное в ``notebooks/``, не находит файл — так было у 09 до v0.8.0).

Ворота АКТУАЛЬНОСТИ выходов (маркер ``big``, нужен ``nbclient``): ноутбуки
поставляются ИСПОЛНЕННЫМИ, и после изменения чисел решателя их выводы обязаны
быть перегенерированы. Аудит 0.8.0 нашёл в 07/08/09 выводы, посчитанные ещё на
0.7.0 (n_contact 84 против 120, r_max 4.13e-2 против 2.62e-2, печатная строка
«отличается на 6.5 %» против фактических 40.6 %) — при том что README называет
цепочку 01–09 обучающей. Проверка исполняет ноутбук в копии и сверяет
ЧИСЛОВЫЕ строки его вывода с сохранёнными.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from plate_solver.problem import Problem

_ROOT = Path(__file__).resolve().parents[1]
_NBS = sorted((_ROOT / "notebooks").glob("*.ipynb"))


@pytest.mark.parametrize("nb_path", _NBS, ids=lambda p: p.stem)
def test_notebook_case_files_validate(nb_path):
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    text = "".join("".join(c.get("source", [])) for c in nb["cells"])
    cases = set(re.findall(r"cases/[\w/]+\.toml", text))
    for case in cases:
        problem = Problem.from_toml(_ROOT / case)     # CaseError ⇒ красный тест
        assert problem.geometry.kind
    if nb_path.stem != "01_circle_api":               # 01 — чистый API без case
        assert cases, f"{nb_path.name}: ожидалась ссылка на case-файл"
    # пути в КОДЕ обязаны быть относительно каталога ноутбуков
    code = "".join("".join(c.get("source", [])) for c in nb["cells"]
                   if c.get("cell_type") == "code")
    bad = [m for m in re.findall(r'"([^"]*cases/[\w/]+\.toml)"', code)
           if not m.startswith("../")]
    assert not bad, (f"{nb_path.name}: путь к case-файлу не относителен каталогу "
                     f"ноутбуков: {bad} — ядро, запущенное в notebooks/, его не найдёт")


@pytest.mark.big
@pytest.mark.parametrize("nb_path", _NBS, ids=lambda p: p.stem)
def test_notebook_outputs_are_current(nb_path, tmp_path):
    """ВОРОТА АКТУАЛЬНОСТИ: сохранённые выводы совпадают со свежим прогоном.

    Сверяются ЧИСЛА в текстовых выводах (картинки и объекты игнорируются;
    сравнение — по нормализованной строке). Расхождение означает, что числа
    решателя сдвинулись, а поставляемые ноутбуки остались от прошлой версии.
    """
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")

    stored = nbformat.read(nb_path, as_version=4)
    fresh = nbformat.read(nb_path, as_version=4)
    nbclient.NotebookClient(
        fresh, timeout=1800, kernel_name="python3",
        resources={"metadata": {"path": str(nb_path.parent)}}).execute()

    # ЛЕТУЧИЕ фрагменты (пути временных файлов ядра, адреса объектов) от прогона
    # к прогону меняются и к числам решателя отношения не имеют
    volatile = re.compile(r"\S*(?:ipykernel_|/var/folders/|/tmp/|0x[0-9a-f]{6,})\S*")
    number = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")

    def numbers(nb):
        out = []
        for cell in nb["cells"]:
            for o in cell.get("outputs", []):
                if o.get("output_type") == "stream":
                    text = volatile.sub("", "".join(o.get("text", [])))
                    out.append(number.findall(text))
        return out

    a, b = numbers(stored), numbers(fresh)
    assert len(a) == len(b), f"{nb_path.name}: изменилось число текстовых выводов"
    for old, new in zip(a, b, strict=True):
        assert len(old) == len(new), (
            f"{nb_path.name}: изменился состав вывода — перезапустите ноутбук\n"
            f"было: {old[:12]}\nстало: {new[:12]}")
        for o, n in zip(old, new, strict=True):
            # сравнение ЧИСЛЕННОЕ с относительным допуском: последний знак
            # печатного вывода зависит от BLAS и порядка суммирования, а ворота
            # ловят СМЫСЛОВОЕ устаревание (в аудите 0.8.0 это были 40…60 %)
            assert float(n) == pytest.approx(float(o), rel=1e-5, abs=1e-12), (
                f"{nb_path.name}: числа сохранённого вывода устарели — "
                f"перезапустите ноутбук ({o} → {n})")
