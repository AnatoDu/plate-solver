r"""Тест-ворота верификационной лестницы изгиба (RFM + Ритц, без МОР).

Лестница по нарастанию сложности; каждая ступень сверяется с точным решением или
эталоном (формулы/таблицы из отчёта по точным решениям, ν=0.3). Тесты вызывают
сами расчётные функции ``run_ladder_*`` — то есть проверяют ДЕЛИВЕРАБЛ.
"""

from __future__ import annotations

import pytest
from run_ladder_1d import run_ladder_1d
from run_ladder_circle import run_ladder_circle
from run_ladder_mms import run_ladder_mms
from run_ladder_rect_clamped import run_ladder_rect_clamped
from run_ladder_rect_hinge import run_ladder_rect_hinge

# Лестница гоняет серии решений с квадратурами до Q=1024 — тяжёлая (память/время).
pytestmark = pytest.mark.big


# --------------------------------------------------------------------------- #
#  Ступень 1 — 1D полоса (полиномиальное решение ⇒ машинная точность)
# --------------------------------------------------------------------------- #
def test_ladder_1d():
    """ВОРОТА: отн. погрешность w_max < 1e-6 для шарнира и защемления (оба p≥2)."""
    d = run_ladder_1d()
    for support in ("hinge", "clamped"):
        for p, r in d[support]["rows"].items():
            assert r["err"] < 1e-6, (support, p, r["err"])


# --------------------------------------------------------------------------- #
#  Ступень 2 — круг
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def circle():
    return run_ladder_circle()


def test_ladder_circle_clamped(circle):
    """ВОРОТА: защемление p=10 < 0.2 % к точной (4.1), и погрешность убывает по Q."""
    assert circle["clamped_rows"][10]["err"] < 2e-3
    q = list(circle["q_rows"].values())
    assert q[0] > q[1] > q[2] > q[3] > q[4]            # нет модельного «пола»
    assert q[-1] < 2e-3


def test_ladder_circle_hinge_soft(circle):
    """ВОРОТА: мягкий шарнир совпадает с «мягким» эталоном < 0.2 % при p=10."""
    assert circle["hinge_rows"][10]["err"] < 2e-3
    # модельный разрыв с Кирхгофом задокументирован как ~26 % (парадокс)
    assert circle["soft_model_gap_pct"] == pytest.approx(26.42, abs=0.1)


# --------------------------------------------------------------------------- #
#  Ступень 3 — MMS
# --------------------------------------------------------------------------- #
def test_ladder_mms():
    """ВОРОТА: MMS-прямоугольник L² < 1e-6 (машинно); круг квадратурно-ограничен (~1/Q)."""
    d = run_ladder_mms()
    assert max(d["rect_rows"].values()) < 1e-6        # сборка/квадратура точны
    disk = list(d["disk_rows"].values())
    assert disk[0] > disk[-1]                          # падает с Q (геометрия R-функции)
    assert disk[-1] < 5e-3                             # ступенчатая маска границы


# --------------------------------------------------------------------------- #
#  Ступень 4 — прямоугольник, шарнир (SSSS)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def rect_hinge():
    return run_ladder_rect_hinge()


def test_ladder_rect_hinge_sin(rect_hinge):
    """ВОРОТА (4a): синус — отн. погрешность w_max < 1e-4 (точное решение)."""
    assert rect_hinge["sin_rows"][10]["err"] < 1e-4


def test_ladder_rect_hinge_unif(rect_hinge):
    """ВОРОТА (4b): равномерная — w_max совпадает с 0.00406 до 3–4 знаков; M ~ 0.0479."""
    assert rect_hinge["unif_rows"][10]["err_w"] < 1e-3
    assert rect_hinge["unif_rows"][12]["err_M"] < 5e-3
    # ряд Навье воспроизводит табличные константы (контроль самого эталона)
    assert abs(rect_hinge["w_navier"] - rect_hinge["w_table"]) / rect_hinge["w_table"] < 1e-3


# --------------------------------------------------------------------------- #
#  Ступень 5 — прямоугольник, защемление (CCCC)
# --------------------------------------------------------------------------- #
def test_ladder_rect_clamped():
    """ВОРОТА: w_max совпадает с 0.00126 до 3 знаков; моменты центра/края — к таблице."""
    d = run_ladder_rect_clamped()
    assert d["rows"][10]["err_w"] < 1e-3
    assert d["rows"][10]["err_Mc"] < 5e-3
    assert d["rows"][12]["err_Me"] < 5e-3              # момент края — «медленная» величина
