r"""viz.py — графики для доклада: поверхность w, реакция r, зона контакта.

Наглядный вывод результатов (figures/ статьи и слайды доклада). Поля
восстанавливаются на регулярной сетке по bbox области и маскируются условием
``ω > 0`` (вне Ω — NaN), реакция ``r`` берётся из ``ContactResult`` (узлы
квадратуры → сетка уже сэмплированы в contact.py, NOTES.md §3).

matplotlib импортируется ВНУТРИ функций (необязателен для расчётов; в headless-
прогонах графику не дёргаем). Каждая функция возвращает ``Figure``; при заданном
``save`` сохраняет в файл, при ``show=True`` — показывает.
"""

from __future__ import annotations

import numpy as np

from .config import Config
from .contact import ContactResult
from .plate import PlateBending


# --------------------------------------------------------------------------- #
#  Вспомогательное
# --------------------------------------------------------------------------- #
def _domain_grid(domain, grid_n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Регулярная сетка по bbox + маска ``inside`` (ω>0) и поле ``ω``."""
    x0, x1, y0, y1 = domain.bbox
    Xg, Yg = np.meshgrid(np.linspace(x0, x1, grid_n), np.linspace(y0, y1, grid_n))
    omega = domain.omega(Xg, Yg)
    return Xg, Yg, omega > 0.0, omega


def _finish(fig, save: str | None, show: bool):
    if save is not None:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    if show:
        import matplotlib.pyplot as plt

        plt.show()
    return fig


def _outline(ax, Xg, Yg, omega):
    """Очертание ∂Ω как линия уровня ω = 0."""
    ax.contour(Xg, Yg, omega, levels=[0.0], colors="k", linewidths=1.0)


# --------------------------------------------------------------------------- #
#  Прогиб
# --------------------------------------------------------------------------- #
def plot_deflection_surface(
    config: Config, plate: PlateBending, cw, *, save: str | None = None, show: bool = False,
    cmap: str = "viridis",
):
    """3D-поверхность прогиба ``w(x, y)`` по области Ω (вне Ω — разрыв сетки)."""
    import matplotlib.pyplot as plt

    Xg, Yg, inside, _ = _domain_grid(plate.domain, config.grid_n)
    W = np.full(Xg.shape, np.nan)
    W[inside] = plate.deflection(cw, Xg[inside], Yg[inside])

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(Xg, Yg, W, cmap=cmap, linewidth=0, antialiased=True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("w")
    ax.set_title("Прогиб пластины w(x, y)")
    return _finish(fig, save, show)


def plot_deflection_contour(
    config: Config, plate: PlateBending, cw, *, save: str | None = None, show: bool = False,
    cmap: str = "viridis", ax=None,
):
    """Плоская карта прогиба ``w(x, y)`` с изолиниями и очертанием ∂Ω."""
    import matplotlib.pyplot as plt

    Xg, Yg, inside, omega = _domain_grid(plate.domain, config.grid_n)
    W = np.full(Xg.shape, np.nan)
    W[inside] = plate.deflection(cw, Xg[inside], Yg[inside])

    own = ax is None
    fig = (ax.figure if not own else plt.figure(figsize=(6, 5)))
    if own:
        ax = fig.add_subplot(111)
    pcm = ax.contourf(Xg, Yg, np.ma.masked_invalid(W), levels=20, cmap=cmap)
    fig.colorbar(pcm, ax=ax, label="w")
    _outline(ax, Xg, Yg, omega)
    ax.set_aspect("equal")
    ax.set_title("Прогиб w(x, y)")
    return _finish(fig, save, show)


# --------------------------------------------------------------------------- #
#  Реакция и зона контакта
# --------------------------------------------------------------------------- #
def plot_reaction(
    config: Config, result: ContactResult, *, save: str | None = None, show: bool = False,
    cmap: str = "magma", ax=None,
):
    """Поле контактной реакции ``r(x, y)``, граница зоны контакта и пик."""
    import matplotlib.pyplot as plt

    own = ax is None
    fig = (ax.figure if not own else plt.figure(figsize=(6, 5)))
    if own:
        ax = fig.add_subplot(111)

    pcm = ax.pcolormesh(result.Xg, result.Yg, np.ma.masked_invalid(result.r_grid),
                        cmap=cmap, shading="auto")
    fig.colorbar(pcm, ax=ax, label="r")
    if result.contact_zone.any():
        ax.contour(result.Xg, result.Yg, result.contact_zone.astype(float),
                   levels=[0.5], colors="cyan", linewidths=1.2)
    px, py = result.peak_xy
    ax.plot(px, py, "w*", markersize=13, markeredgecolor="k", label="пик r")
    _, _, _, omega = _domain_grid(result.plate.domain, config.grid_n)
    _outline(ax, result.Xg, result.Yg, omega)
    ax.set_aspect("equal")
    ax.set_title("Контактная реакция r(x, y)")
    ax.legend(loc="upper right", fontsize=8)
    return _finish(fig, save, show)


def plot_contact_zone(
    config: Config, result: ContactResult, *, save: str | None = None, show: bool = False,
    ax=None,
):
    """Зона контакта (маска ``r > 0``) поверх очертания области ∂Ω."""
    import matplotlib.pyplot as plt

    own = ax is None
    fig = (ax.figure if not own else plt.figure(figsize=(6, 5)))
    if own:
        ax = fig.add_subplot(111)

    _, _, _, omega = _domain_grid(result.plate.domain, config.grid_n)
    ax.contourf(result.Xg, result.Yg, result.contact_zone.astype(float),
                levels=[0.5, 1.5], colors=["tab:red"], alpha=0.6)
    _outline(ax, result.Xg, result.Yg, omega)
    px, py = result.peak_xy
    ax.plot(px, py, "k*", markersize=12)
    ax.set_aspect("equal")
    ax.set_title("Зона контакта (r > 0)")
    return _finish(fig, save, show)


def plot_convergence(
    result: ContactResult, *, save: str | None = None, show: bool = False, ax=None,
):
    """График сходимости МОР: невязка ‖Δr‖ по итерациям (полулог)."""
    import matplotlib.pyplot as plt

    own = ax is None
    fig = (ax.figure if not own else plt.figure(figsize=(6, 4)))
    if own:
        ax = fig.add_subplot(111)
    ax.semilogy(np.arange(1, result.residual_history.size + 1), result.residual_history)
    ax.set_xlabel("итерация")
    ax.set_ylabel("‖r_k − r_{k-1}‖")
    ax.set_title(f"Сходимость МОР ({result.iters} итер.)")
    ax.grid(True, which="both", alpha=0.3)
    return _finish(fig, save, show)


# --------------------------------------------------------------------------- #
#  Сводный планшет
# --------------------------------------------------------------------------- #
def plot_contact_summary(
    config: Config, result: ContactResult, *, save: str | None = None, show: bool = False,
):
    """Планшет 2×2: прогиб, реакция, зона контакта, сходимость (слайд доклада)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plot_deflection_contour(config, result.plate, result.cw, ax=axes[0, 0])
    plot_reaction(config, result, ax=axes[0, 1])
    plot_contact_zone(config, result, ax=axes[1, 0])
    plot_convergence(result, ax=axes[1, 1])
    fig.tight_layout()
    return _finish(fig, save, show)


__all__ = [
    "plot_deflection_surface",
    "plot_deflection_contour",
    "plot_reaction",
    "plot_contact_zone",
    "plot_convergence",
    "plot_contact_summary",
]
