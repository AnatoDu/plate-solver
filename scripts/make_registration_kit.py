#!/usr/bin/env python3
"""make_registration_kit.py — комплект материалов госрегистрации ПрЭВМ.

Собирает в НЕОТСЛЕЖИВАЕМЫЙ каталог private/registration/:

* ``referat.md`` — заготовка реферата по полям Роспатента (название,
  назначение, функциональные возможности из docs/FEATURES.md, язык,
  объём исходного текста, требования к окружению);
* ``listing.md`` — листинг исходного текста: титульный лист + склейка
  src/plate_solver/*.py в фиксированном порядке с колонтитулами и
  постраничной нумерацией (60 строк/страница; PDF — печатью из любого
  просмотрщика: новых зависимостей пакет не приобретает);
* ``checksums.txt`` — SHA-256 всех файлов листинга.

Комплект НЕ коммитится (private/ в .gitignore). Подача — по чек-листу
private/REGISTRATION.md.
"""

from __future__ import annotations

import hashlib
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "plate_solver"
OUT = ROOT / "private" / "registration"

#: фиксированный порядок листинга: слоями, сверху вниз (ARCHITECTURE.md)
ORDER = [
    "__init__.py", "config.py", "problem.py", "cli.py", "dispatch.py",
    "references.py",
    "contact.py", "ktn.py",
    "plate.py", "poisson.py", "clamped.py", "radial.py",
    "geometry.py", "rfunctions.py", "basis.py", "quadrature.py",
    "assembler.py",
    "analytic.py", "analytic_auto.py", "ladder.py", "verify_fem.py",
    "penalty.py",
    "green1d.py", "mor1d.py", "stamp.py", "stamp_ritz.py",
    "strip_contact.py", "problems.py",
    "viz.py",
]

LINES_PER_PAGE = 60


def referat() -> str:
    sys.path.insert(0, str(ROOT / "src"))
    import plate_solver

    n_files = len(ORDER)
    n_lines = sum(len((SRC / f).read_text(encoding="utf-8").splitlines())
                  for f in ORDER)
    n_kb = sum((SRC / f).stat().st_size for f in ORDER) / 1024
    return f"""# Реферат программы для ЭВМ (заготовка полей Роспатента)

**Название программы для ЭВМ.** plate-solver — комплекс программ для
расчёта изгиба и одностороннего контактного взаимодействия упругих
пластин произвольного очертания.

**Назначение и область применения.** Математическое моделирование
напряжённо-деформированного состояния тонких упругих пластин
произвольной формы в плане (включая невыпуклые области, входящие углы и
вырезы) при изгибе и одностороннем контакте с жёстким основанием, штампом
или второй пластиной. Область применения: научные исследования по
механике деформируемого твёрдого тела, инженерные расчёты пластинчатых
элементов конструкций, верификация численных методов.

**Функциональные возможности.** Задание геометрии R-функциями
(система R0 В. Л. Рвачёва): круг, прямоугольник, L-форма, кольцо,
составные области (мини-язык compose). Изгиб по классической теории
(мягкий шарнир — расщепление бигармонического уравнения; защемление —
прямой метод Ритца; смешанные краевые условия и свободный край — полная
билинейная форма) и по уточнённой теории типа Кармана–Тимошенко–Нагди
(поперечный сдвиг и обжатие). Односторонний контакт методом обобщённой
реакции: жёсткое основание, неплоский штамп (поле зазора), силовое
управление штампом, контакт двух пластин. Напряжения и прогибы лицевых
поверхностей. Постановка задач декларативными case-файлами (TOML) с
валидатором и диагностикой; интерфейс командной строки (plate-solve,
plate-verify, plate-ladder); автоматическая верификация по эталонам
(аналитика с самосертификацией, изготовленные решения, независимый МКЭ,
1D-сверка); графический вывод. Полный перечень возможностей —
автоматическая матрица docs/FEATURES.md.

**Язык программирования.** Python (версии 3.11–3.12); свободные
библиотеки NumPy, SciPy, SymPy, Matplotlib (опционально scikit-fem).

**Объём программы для ЭВМ.** {n_files} модулей исходного текста,
{n_lines} строк, {n_kb:.0f} КБ (версия {plate_solver.__version__},
дата сборки комплекта {date.today().isoformat()}).

**Требуемые технические средства.** ЭВМ с ОС Windows / Linux / macOS,
Python ≥ 3.11, ОЗУ от 4 ГБ (тяжёлые верификационные расчёты — от 8 ГБ).

**Правообладатель / автор.** Дуркин А. А. (реквизиты — по форме подачи).
"""


def listing() -> tuple[str, list[Path]]:
    sys.path.insert(0, str(ROOT / "src"))
    import plate_solver

    files = [SRC / f for f in ORDER]
    title = f"""{'=' * 78}

    plate-solver — комплекс программ для расчёта изгиба и одностороннего
    контактного взаимодействия упругих пластин произвольного очертания

    ЛИСТИНГ ИСХОДНОГО ТЕКСТА

    Автор: Дуркин А. А.
    Версия: {plate_solver.__version__}
    Дата: {date.today().isoformat()}
    Язык: Python
    Модулей: {len(files)}

{'=' * 78}
"""
    out = [title]
    page_no = 1
    for f in files:
        lines = f.read_text(encoding="utf-8").splitlines()
        for i in range(0, len(lines), LINES_PER_PAGE):
            page_no += 1
            chunk = lines[i:i + LINES_PER_PAGE]
            header = (f"— стр. {page_no} — {f.relative_to(ROOT)} "
                      f"(строки {i + 1}–{i + len(chunk)}) —")
            out.append("\n" + header.center(78, "·") + "\n")
            out.extend(chunk)
    out.append(f"\n{'=' * 78}\nКонец листинга. Всего страниц: {page_no}.\n")
    return "\n".join(out), files


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "referat.md").write_text(referat(), encoding="utf-8")
    text, files = listing()
    (OUT / "listing.md").write_text(text, encoding="utf-8")
    sums = []
    for f in files:
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        sums.append(f"{h}  {f.relative_to(ROOT)}")
    h_listing = hashlib.sha256((OUT / "listing.md").read_bytes()).hexdigest()
    sums.append(f"{h_listing}  private/registration/listing.md")
    (OUT / "checksums.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")
    print(f"комплект: {OUT} (referat.md, listing.md, checksums.txt)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
