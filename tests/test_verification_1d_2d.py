"""ВЕРИФИКАЦИЯ 1D ↔ 2D — несокращаемое ядро диссертации.

Сравнивает одномерное аналитическое решение защемлённой круглой пластины
(эталон) с двумерным численным решением той же задачи на круге, заданном
R-функцией. Пока 2D-решатель — заготовка, тест аккуратно ПРОПУСКАЕТСЯ
(skip) с понятной причиной. Как только ``solver.solve_clamped_circular``
будет реализован, тест автоматически станет рабочим критерием корректности
всего метода (положение на защиту № 4).

Менять допуск TOL и сетку N по мере уточнения метода; ослаблять критерий,
чтобы «пройти», запрещено — это сертификат доверия ко всему счёту.
"""

import numpy as np
import pytest

from plate_solver import PlateMaterial, analytic
from plate_solver.solver import solve_clamped_circular

# Параметры эталонной задачи
A = 1.0          # радиус
Q = 1.0e4        # равномерная нагрузка
N = 64           # дискретизация 2D-метода
TOL = 1e-3       # относительный допуск согласования 1D ↔ 2D


def _sample_2d_on_radius(w2d, a, npts=21):
    """Снять профиль 2D-решения вдоль радиуса (y=0, x от 0 до a).

    Интерфейс 2D-решения ещё не зафиксирован: поддерживаем либо callable
    w(x, y), либо объект с методом .at(x, y). Уточнить в главе 2.
    """
    xs = np.linspace(0.0, a, npts)
    if callable(w2d):
        return xs, np.array([float(w2d(x, 0.0)) for x in xs])
    if hasattr(w2d, "at"):
        return xs, np.array([float(w2d.at(x, 0.0)) for x in xs])
    raise AssertionError(
        "Неизвестный формат 2D-решения: ожидался callable w(x,y) или объект с .at()."
    )


def test_verification_1d_vs_2d_clamped_circular(steel_plate: PlateMaterial):
    # 2D-решение (когда метод не реализован — корректно пропускаем тест)
    try:
        w2d = solve_clamped_circular(A, Q, steel_plate, n=N)
    except NotImplementedError as exc:
        pytest.skip(f"2D-решатель ещё не реализован: {exc}")

    xs, w_num = _sample_2d_on_radius(w2d, A)
    w_ref = analytic.clamped_uniform(xs, A, Q, steel_plate.D)

    denom = float(np.max(np.abs(w_ref))) or 1.0
    rel_err = float(np.max(np.abs(w_num - w_ref))) / denom

    assert rel_err < TOL, (
        f"Расхождение 1D↔2D = {rel_err:.2e} превышает допуск {TOL:.1e}. "
        "Не ослабляйте критерий — ищите ошибку в методе/краевых условиях."
    )
