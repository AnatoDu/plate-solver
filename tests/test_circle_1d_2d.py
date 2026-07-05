r"""Тест-ворота Части 2 — круг как ОДНОМЕРНАЯ задача по радиусу.

Осесимметричная пластина решена 1D-Ритцем по радиусу и сверена с радиальным
сечением 2D-решения (RFM+Ритц) и с точной формулой: верификация 1D ↔ 2D ↔ аналитика.
"""

from __future__ import annotations

import numpy as np
import pytest
from run_circle_1d_2d import run_circle_1d_2d

# 2D-часть сверки считается на квадратуре Q=1024 — тяжёлая (память/время).
pytestmark = pytest.mark.big


@pytest.fixture(scope="module")
def circ():
    return run_circle_1d_2d()


def test_circle_1d_exact(circ):
    """ВОРОТА: 1D-решение защемления совпадает с точной формулой < 0.1 %."""
    assert circ["clamped"]["l2_1d_ex"] < 0.1
    # 1D точнее 2D (точное решение лежит в радиальной структуре) — машинно
    assert circ["clamped"]["l2_1d_ex"] < circ["clamped"]["l2_2d_ex"]


def test_circle_1d_vs_2d(circ):
    """ВОРОТА: 1D и 2D-сечение совпадают того же порядка, что их отклонения от точного."""
    c = circ["clamped"]
    assert c["l2_1d_2d"] < 0.5
    assert c["l2_1d_2d"] == pytest.approx(c["l2_2d_ex"], abs=0.05)   # 1D↔2D ≈ 2D↔точное


def test_circle_profiles_agree_pointwise(circ):
    """Профили монотонно совпадают по всему радиусу (а не только в среднем)."""
    c = circ["clamped"]
    w0 = c["exact"][0]
    assert np.max(np.abs(c["w1d"] - c["w2d"])) / w0 < 5e-3        # поточечно мало
    # оба профиля монотонно убывают от центра к краю
    assert np.all(np.diff(c["w1d"]) <= 1e-12)
    assert np.all(np.diff(c["w2d"]) <= 1e-12)


def test_circle_hinge_1d_vs_2d(circ):
    """Мягкий шарнир: 1D (две радиальные Пуассоны) ↔ 2D ↔ мягкий эталон сходятся."""
    s = circ["hinge"]
    assert s["l2_1d_ex"] < 0.1                                    # 1D машинно к мягкому
    assert s["l2_1d_2d"] < 0.5                                    # 1D↔2D того же порядка
