"""Контактная задача: балка-полоса на жёстком основании (МОР vs. точное решение).

Постановка: балка-полоса длиной L=100 (шарнир слева, плоскость симметрии
справа) под равномерной нагрузкой q₀=4 контактирует с жёстким основанием
высотой Δ=1, расположенным на участке [45, 100].

Запуск:  python examples/contact_strip.py

Результат: два графика —
    • прогиб w(x): МОР (красная сплошная) и точное решение Maple (синяя пунктир),
    • контактная реакция r(x).
"""

import matplotlib.pyplot as plt

from plate_solver.mor1d import ContactStrip1D, solve_mor_1d
from plate_solver.strip_contact import W_MAPLE, X_MAPLE


def main() -> None:
    problem = ContactStrip1D(
        E=2.1e6,
        h=1.0,
        nu=0.3,
        L=100.0,
        q0=4.0,
        gap=1.0,
        foundation_start=45.0,
        n=100,
        beta=0.02,
        max_iter=3_000_000,
    )

    print("Решение МОР 1D…")
    x, w, r = solve_mor_1d(problem)
    print("Готово.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(x, w, "r-", label="МОР (численное)")
    ax1.plot(X_MAPLE, W_MAPLE, "b--", label="Точное решение (Maple)")
    ax1.axhline(problem.gap, color="gray", linestyle=":", linewidth=0.8, label=f"Δ = {problem.gap}")
    ax1.axvline(problem.foundation_start, color="green", linestyle=":", linewidth=0.8,
                label=f"x₀ = {problem.foundation_start}")
    ax1.set_title("Прогиб пластины")
    ax1.set_xlabel("x, см")
    ax1.set_ylabel("w, см")
    ax1.legend(fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(x, r, "r-")
    ax2.axvline(problem.foundation_start, color="green", linestyle=":", linewidth=0.8)
    ax2.set_title("Контактная реакция")
    ax2.set_xlabel("x, см")
    ax2.set_ylabel("r, см")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
