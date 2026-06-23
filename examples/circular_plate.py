"""Минимальный рабочий пример: прогиб защемлённой круглой пластины (аналитика).

Запуск:  python examples/circular_plate.py
Работает сразу (аналитическое решение). 2D-расчёт появится по мере наполнения
solver/ — тогда пример можно дополнить сравнением 1D ↔ 2D.
"""

import numpy as np

from plate_solver import PlateMaterial, analytic, geometry


def main() -> None:
    mat = PlateMaterial(E=2.0e11, nu=0.3, h=0.01)  # сталь, СИ
    a, q = 1.0, 1.0e4

    # граница области (круг) через R-функцию — «способ задания границы»
    omega = lambda x, y: geometry.circle(x, y, radius=a)  # noqa: E731
    assert omega(0.0, 0.0) > 0  # центр внутри области

    r = np.linspace(0.0, a, 6)
    w = analytic.clamped_uniform(r, a, q, mat.D)

    print(f"Жёсткость D = {mat.D:.3e} Н·м")
    print(f"Макс. прогиб (центр): {analytic.clamped_uniform_wmax(a, q, mat.D):.3e} м")
    print("Профиль w(r):")
    for ri, wi in zip(r, w):
        print(f"  r={ri:.3f} м   w={wi:.3e} м")


if __name__ == "__main__":
    main()
