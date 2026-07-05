"""golden_config.py — единый «золотой» конфиг главы 4 доклада.

Все числа и рисунки главы выходят из ОДНОЙ серии с этим набором параметров.
Круг (табл. 4.1) считается при h=1.0 (см. свойство D); вся L-серия (w_free,
верификация, контакт, КТН — табл. 4.2/4.3) — при единой толщине h_ktn, чтобы
зазор Δ = gap_factor·w_free был ОДИН И ТОТ ЖЕ в контакте и в КТН.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GoldenConfig:
    # физика
    nu: float = 0.3
    q0: float = 4.0
    a: float = 1.0
    # L-форма
    L_side: float = 1.0
    L_cut: float = 0.5
    # дискретизация
    p: int = 10
    p_sweep: tuple = (2, 4, 6, 8, 10)
    Q_circle: int = 1024
    Q_lshape: int = 120
    grid_n: int = 80
    # контакт (МОР)
    beta: float = 1.2
    gap_factor: float = 0.5      # Δ = gap_factor * w_free
    mor_iter: int = 8000
    mor_tol: float = 1e-8
    # КТН (и толщина всей L-серии — ради единого Δ)
    h_ktn: float = 0.06
    # независимый МКЭ (scikit-fem) для верификации L-формы
    fem_mesh_m: int = 16
    fem_refine: int = 3
    # вывод
    out_md: str = "golden_results.md"
    fig_dir: str = "figures"

    # модуль Юнга и толщина круга — фиксированы
    E: float = 2.1e6
    h_circle: float = 1.0

    @property
    def D(self) -> float:
        """Цилиндрическая жёсткость круга (h=1.0)."""
        return self.E * self.h_circle**3 / (12 * (1 - self.nu**2))

    @property
    def D_lshape(self) -> float:
        """Цилиндрическая жёсткость L-серии (h=h_ktn)."""
        return self.E * self.h_ktn**3 / (12 * (1 - self.nu**2))
