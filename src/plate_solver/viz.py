r"""viz.py — графики результатов: поверхность w, реакция r, зона контакта.

Наглядный вывод результатов (публикационные фигуры). Поля
восстанавливаются на регулярной сетке по bbox области и маскируются условием
``ω > 0`` (вне Ω — NaN), реакция ``r`` берётся из ``ContactResult`` (узлы
квадратуры → сетка уже сэмплированы в contact.py, NOTES.md §3).

matplotlib импортируется ВНУТРИ функций (необязателен для расчётов; в headless-
прогонах графику не дёргаем). Каждая функция возвращает ``Figure``; при заданном
``save`` сохраняет в файл, при ``show=True`` — показывает.
"""

from __future__ import annotations

from pathlib import Path

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
    """Планшет 2×2: прогиб, реакция, зона контакта, сходимость."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plot_deflection_contour(config, result.plate, result.cw, ax=axes[0, 0])
    plot_reaction(config, result, ax=axes[0, 1])
    plot_contact_zone(config, result, ax=axes[1, 0])
    plot_convergence(result, ax=axes[1, 1])
    fig.tight_layout()
    return _finish(fig, save, show)


def plot_pair_summary(
    result, *, save: str | None = None, show: bool = False,
):
    """Планшет 2×2 для контакта ДВУХ пластин: w₁, w₂, r, сходимость.

    ``result`` — :class:`plate_solver.contact.TwoPlateResult`; поля берутся
    с фоновой сетки (NaN вне соответствующей области). Реакция ``r`` —
    пара взаимодействия: первая пластина получает −r, вторая +r.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, fld, ttl in ((axes[0, 0], result.w_grid, "Прогиб w₁ (верхняя пластина)"),
                         (axes[0, 1], result.w2_grid, "Прогиб w₂ (нижняя пластина)")):
        pcm = ax.contourf(result.Xg, result.Yg, np.ma.masked_invalid(fld),
                          levels=20, cmap="viridis")
        fig.colorbar(pcm, ax=ax, label="w")
        ax.set_aspect("equal")
        ax.set_title(ttl)
    ax = axes[1, 0]
    pcm = ax.pcolormesh(result.Xg, result.Yg, np.ma.masked_invalid(result.r_grid),
                        cmap="magma", shading="auto")
    fig.colorbar(pcm, ax=ax, label="r")
    if result.contact_zone.any():
        ax.contour(result.Xg, result.Yg, result.contact_zone.astype(float),
                   levels=[0.5], colors="cyan", linewidths=1.2)
    ax.set_aspect("equal")
    ax.set_title("Реакция взаимодействия r (пара ±r)")
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
    "plot_pair_summary",
    "plot_modes",
    "replot",
    "surface3d",
    "stress_maps",
]

# --------------------------------------------------------------------------- #
#  Перерисовка из fields.npz: фигуры без пересчёта
# --------------------------------------------------------------------------- #
def replot(result_dir, formats=("png",), dpi: int = 300,
           surface: str = "mid") -> list:
    """Перерисовать фигуры из ``<dir>/fields.npz`` (схема полей 1, 2 или 3).

    Пересчёт не выполняется: всё берётся из снимка полей. Создаются
    w-поверхность (``surface`` = ``mid`` | ``top`` | ``bottom`` — срединная
    или лицевые, NOTES §21; лицевые доступны со схемы 2), карты лицевых
    напряжений (σx±, σy±, τxy±), моментов, а со схемы 3 — мембранных усилий
    ``N`` (нелинейные теории), эквивалентного ``σ_vm`` и напряжений ВТОРОЙ
    пластины пары (суффикс «2»); при контакте — карта реакции с зоной.
    Возвращает список путей.
    """
    import json as _json

    import matplotlib.pyplot as plt

    result_dir = Path(result_dir)
    data = np.load(result_dir / "fields.npz", allow_pickle=False)
    if int(data["fields_schema"]) not in (1, 2, 3):
        raise ValueError(f"Неизвестная версия схемы полей: {int(data['fields_schema'])}")
    X, Y = np.meshgrid(data["x"], data["y"])
    out: list = []
    try:
        theory = _json.loads(str(data["problem_json"]))["model"]["theory"]
    except (KeyError, ValueError, TypeError):
        theory = "classic"

    def _save(fig, name):
        for fmt in formats:
            path = result_dir / f"{name}.{fmt}"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            out.append(path)
        plt.close(fig)

    # 1) поверхность прогиба (выбор поверхности)
    key, label = {"mid": ("w", "w"), "top": ("w_top", "w (верхняя лицевая)"),
                  "bottom": ("w_bot", "w (нижняя лицевая)")}.get(surface, (None, None))
    if key is None:
        raise ValueError(f"surface: получено {surface!r}, "
                         "ожидалось mid | top | bottom")
    if key not in data.files:                     # схема 1 — только срединная
        key, label = "w", "w"
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, data[key], cmap="viridis", linewidth=0, antialiased=True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("w")
    ax.set_title(f"Прогиб {label}(x, y) (из fields.npz)")
    _save(fig, "w_surface")

    # 2) карты лицевых напряжений: диверг. палитра, симметричная норма
    comps = [("sx_top", "σx, верх"), ("sx_bot", "σx, низ"),
             ("sy_top", "σy, верх"), ("sy_bot", "σy, низ"),
             ("txy_top", "τxy, верх"), ("txy_bot", "τxy, низ")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    vmax = max(float(np.nanmax(np.abs(data[k]))) for k, _ in comps)
    for ax, (key, title) in zip(axes.ravel(), comps, strict=True):
        pcm = ax.pcolormesh(X, Y, np.ma.masked_invalid(data[key]),
                            cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
        ax.set_aspect("equal")
        ax.set_title(title)
    fig.colorbar(pcm, ax=axes, label="σ", shrink=0.8)
    fig.suptitle("Напряжения на лицевых поверхностях (NOTES §19)")
    _save(fig, "stress_faces")

    # 3) контакт: реакция + зона + профиль σy⁻ вдоль сечения через зону (D1)
    if "r" in data.files:
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0))
        pcm = ax.pcolormesh(X, Y, np.ma.masked_invalid(data["r"]),
                            cmap="magma", shading="auto")
        zone = data["zone"]
        if zone.any():
            ax.contour(X, Y, zone.astype(float), levels=[0.5],
                       colors="cyan", linewidths=1.2)
            j = int(round(np.mean(np.nonzero(zone.any(axis=1))[0])))
        else:
            j = zone.shape[0] // 2
        yline = float(data["y"][j])
        ax.axhline(yline, color="w", ls="--", lw=0.8)
        ax.set_aspect("equal")
        # при КТН зона контакта определяется прогибом НИЖНЕЙ лицевой (§21)
        ttl = "Реакция r(x, y) и зона контакта"
        if theory in ("ktn", "ktn_linear", "ktn_full"):
            ttl += " (зона — по нижней лицевой)"
        ax.set_title(ttl)
        fig.colorbar(pcm, ax=ax, label="r")
        prof = data["sy_bot"][j, :]
        keep = np.isfinite(prof)
        ax2.plot(data["x"][keep], prof[keep], "o-", ms=3,
                 label=f"σy, низ (y = {yline:.3f})")
        in_zone = zone[j, :] & keep
        if in_zone.any():
            ax2.plot(data["x"][in_zone], data["sy_bot"][j, :][in_zone], "rs",
                     ms=5, label="зона контакта (+ν/(1−ν)·r)")
        ax2.set_xlabel("x")
        ax2.set_ylabel("σy на нижней лицевой")
        ax2.grid(alpha=0.3)
        ax2.legend(fontsize=8)
        ax2.set_title("Профиль σy⁻ через зону (влияние обжатия)")
        _save(fig, "reaction")

    # 4) карта обжатия dh = w_bot − w_срединная (лицевая кинематика КТН, §21);
    #    для classic/karman тождественно ноль ⇒ фигура не строится.
    if "dh" in data.files and float(np.nanmax(np.abs(data["dh"]))) > 0.0:
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        vmax = float(np.nanmax(np.abs(data["dh"])))
        pcm = ax.pcolormesh(X, Y, np.ma.masked_invalid(data["dh"]),
                            cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Обжатие dh = w_нижн.лицевая − w_срединная (§21)")
        fig.colorbar(pcm, ax=ax, label="dh")
        _save(fig, "compression")

    # 5) карты изгибных моментов Mx, My, Mxy (matplotlib; ранее только в VTK)
    if "Mx" in data.files:
        mcomps = [("Mx", "Mx"), ("My", "My"), ("Mxy", "Mxy")]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
        vmax = max((float(np.nanmax(np.abs(data[k]))) for k, _ in mcomps
                    if k in data.files), default=0.0) or 1.0
        for ax, (key, title) in zip(axes.ravel(), mcomps, strict=True):
            pcm = ax.pcolormesh(X, Y, np.ma.masked_invalid(data[key]),
                                cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
            ax.set_aspect("equal")
            ax.set_title(title)
        fig.colorbar(pcm, ax=axes, label="M", shrink=0.8)
        fig.suptitle("Изгибные моменты Mx, My, Mxy")
        _save(fig, "moments")

    # 6) мембранные усилия Nx, Ny, Nxy (нелинейные теории; схема 3, v0.6.6)
    if "Nx" in data.files:
        ncomps = [("Nx", "Nx"), ("Ny", "Ny"), ("Nxy", "Nxy")]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
        vmax = max((float(np.nanmax(np.abs(data[k]))) for k, _ in ncomps),
                   default=0.0) or 1.0
        for ax, (key, title) in zip(axes.ravel(), ncomps, strict=True):
            pcm = ax.pcolormesh(X, Y, np.ma.masked_invalid(data[key]),
                                cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
            ax.set_aspect("equal")
            ax.set_title(title)
        fig.colorbar(pcm, ax=axes, label="N", shrink=0.8)
        fig.suptitle("Мембранные усилия Nx, Ny, Nxy (натяжение срединной)")
        _save(fig, "membrane")

    # 6б) перерезывающие силы Qx, Qy (равновесие моментов; схема 3, v0.6.6)
    if "Qx" in data.files:
        qcomps = [("Qx", "Qx"), ("Qy", "Qy")]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
        vmax = max(float(np.nanmax(np.abs(data[k]))) for k, _ in qcomps) or 1.0
        for ax, (key, title) in zip(axes.ravel(), qcomps, strict=True):
            pcm = ax.pcolormesh(X, Y, np.ma.masked_invalid(data[key]),
                                cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
            ax.set_aspect("equal")
            ax.set_title(title)
        fig.colorbar(pcm, ax=axes, label="Q", shrink=0.8)
        fig.suptitle("Перерезывающие силы Qx, Qy (из равновесия моментов)")
        _save(fig, "shear")

    # 7) эквивалентное напряжение фон Мизеса на лицевых (схема 3, v0.6.6)
    if "svm_top" in data.files:
        panels = [("svm_top", "σ_vm, верх"), ("svm_bot", "σ_vm, низ")]
        if "svm_top2" in data.files:                    # пара: + вторая пластина
            panels += [("svm_top2", "σ_vm, верх (пластина 2)"),
                       ("svm_bot2", "σ_vm, низ (пластина 2)")]
        ncols = 2
        nrows = (len(panels) + 1) // 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4.6 * nrows),
                                 squeeze=False)
        vmax = max(float(np.nanmax(np.abs(data[k]))) for k, _ in panels) or 1.0
        for ax, (key, title) in zip(axes.ravel(), panels, strict=False):
            pcm = ax.pcolormesh(X, Y, np.ma.masked_invalid(data[key]),
                                cmap="magma", vmin=0.0, vmax=vmax, shading="auto")
            ax.set_aspect("equal")
            ax.set_title(title)
        for ax in axes.ravel()[len(panels):]:
            ax.axis("off")
        fig.colorbar(pcm, ax=axes, label="σ_vm", shrink=0.8)
        fig.suptitle("Эквивалентное напряжение фон Мизеса (лицевые поверхности)")
        _save(fig, "von_mises")

    # 8) напряжения ВТОРОЙ пластины пары (данные сохранялись со схемы 2,
    #    фигура — v0.6.6): та же 2×3, что stress_faces
    if "sx_top2" in data.files:
        comps2 = [("sx_top2", "σx, верх"), ("sx_bot2", "σx, низ"),
                  ("sy_top2", "σy, верх"), ("sy_bot2", "σy, низ"),
                  ("txy_top2", "τxy, верх"), ("txy_bot2", "τxy, низ")]
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        vmax = max(float(np.nanmax(np.abs(data[k]))) for k, _ in comps2) or 1.0
        for ax, (key, title) in zip(axes.ravel(), comps2, strict=True):
            pcm = ax.pcolormesh(X, Y, np.ma.masked_invalid(data[key]),
                                cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
            ax.set_aspect("equal")
            ax.set_title(title)
        fig.colorbar(pcm, ax=axes, label="σ", shrink=0.8)
        fig.suptitle("Напряжения на лицевых ВТОРОЙ пластины пары (канон §19)")
        _save(fig, "stress_faces2")
    return out

def section_profile(result_dir, key: str, p0, p1, n: int = 200):
    r"""Профиль поля ``key`` вдоль отрезка ``p0 → p1`` из ``fields.npz`` (v0.6.6).

    Билинейная интерполяция сеточного поля на ``n`` точках отрезка; вне Ω —
    ``NaN``. Возвращает ``(s, values)``: дуговая координата (0 в ``p0``) и
    значения. Ключи — любые сеточные из fields.npz (см. CASE_SCHEMA#output):
    ``w``, ``Mx``, ``Qx``, ``sx_bot``, ``svm_top``, ``r`` …
    """
    from scipy.interpolate import RegularGridInterpolator

    result_dir = Path(result_dir)
    data = np.load(result_dir / "fields.npz", allow_pickle=False)
    if key not in data.files:
        raise KeyError(f"section_profile: ключа {key!r} нет в fields.npz "
                       f"(есть: {sorted(data.files)})")
    field = np.asarray(data[key], float)
    x, y = np.asarray(data["x"], float), np.asarray(data["y"], float)
    if field.shape != (y.size, x.size):
        raise ValueError(f"section_profile: {key!r} — не сеточное поле "
                         f"(форма {field.shape})")
    interp = RegularGridInterpolator((y, x), field, bounds_error=False,
                                     fill_value=np.nan)
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    t = np.linspace(0.0, 1.0, int(n))
    pts = p0[None, :] + t[:, None] * (p1 - p0)[None, :]
    vals = interp(pts[:, ::-1])                     # (y, x)-порядок интерполятора
    s = t * float(np.hypot(*(p1 - p0)))
    return s, vals


def overlay_profiles(result_dirs, key: str, p0, p1, *, labels=None, n: int = 200,
                     save=None, dpi: int = 300):
    r"""Наложение профилей одного поля из НЕСКОЛЬКИХ результатов (v0.6.6).

    Сравнение теорий/кейсов вдоль одного сечения: по кривой на каталог
    результата (``fields.npz``). Возвращает ``(fig, ax)``; ``save`` — путь
    файла фигуры (расширение задаёт формат).
    """
    import matplotlib.pyplot as plt

    dirs = list(result_dirs)
    labels = list(labels) if labels is not None else [Path(d).name for d in dirs]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for d, lab in zip(dirs, labels, strict=True):
        s, vals = section_profile(d, key, p0, p1, n=n)
        ax.plot(s, vals, "-", lw=1.6, label=lab)
    ax.set_xlabel("s (вдоль сечения)")
    ax.set_ylabel(key)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title(f"Профиль {key}: ({p0[0]:g},{p0[1]:g}) → ({p1[0]:g},{p1[1]:g})")
    if save is not None:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
    return fig, ax


def surface3d(X, Y, W, *, elev: float = 28.0, azim: float = -60.0,
              cmap: str = "viridis", title: str = "w(x, y)",
              save: str | None = None, show: bool = False, dpi: int = 300):
    """Публикационная 3D-поверхность (D1): ракурс (elev, azim), NaN-стрижка вне ω."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(np.asarray(X, float), np.asarray(Y, float),
                    np.asarray(W, float), cmap=cmap, linewidth=0,
                    antialiased=True)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("w")
    ax.set_title(title)
    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
    return _finish(fig, None, show)


def plot_modes(eigen, n: int = 6, *, grid_n: int = 80, save: str | None = None,
               show: bool = False, dpi: int = 300):
    """Сетка первых ``n`` собственных форм (контуры); значения — в заголовках.

    ``eigen`` — :class:`~plate_solver.eigenmodes.EigenPair` (устойчивость или
    колебания). Формы нормированы на max|·|=1 (:meth:`EigenPair.mode_on_grid`).
    """
    import matplotlib.pyplot as plt

    n = min(int(n), len(eigen.values))
    ncol = min(3, n)
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 4.0 * nrow), squeeze=False)
    label = "λ_cr" if eigen.kind == "buckling" else "ω"
    for k, ax in enumerate(axes.ravel()):
        if k < n:
            Xg, Yg, W = eigen.mode_on_grid(k, grid_n)
            ax.contourf(Xg, Yg, np.ma.masked_invalid(W), levels=20, cmap="RdBu_r")
            ax.set_aspect("equal")
            ax.set_title(f"Форма {k + 1}: {label} = {eigen.values[k]:.4g}")
        else:
            ax.axis("off")
    kind = "устойчивость" if eigen.kind == "buckling" else "колебания"
    fig.suptitle(f"Собственные формы ({kind})")
    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
    return _finish(fig, None, show)


def stress_maps(X, Y, stresses: dict, *, components=None,
                save: str | None = None, show: bool = False, dpi: int = 300):
    """Карты лицевых напряжений (D1): дивергентная палитра, симметричная норма.

    ``stresses`` — словарь из :func:`plate_solver.ktn.stresses_faces`;
    ``components`` — список ключей (None ⇒ вся шестёрка сеткой 2×3).
    """
    import matplotlib.pyplot as plt

    titles = {"sx_top": "σx, верх", "sx_bot": "σx, низ",
              "sy_top": "σy, верх", "sy_bot": "σy, низ",
              "txy_top": "τxy, верх", "txy_bot": "τxy, низ"}
    keys = list(components) if components else list(titles)
    ncol = 3 if len(keys) > 2 else len(keys)
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4.4 * nrow),
                             squeeze=False)
    vmax = max(float(np.nanmax(np.abs(stresses[k]))) for k in keys) or 1.0
    pcm = None
    for ax, key in zip(axes.ravel(), keys, strict=False):
        pcm = ax.pcolormesh(X, Y, np.ma.masked_invalid(stresses[key]),
                            cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
        ax.set_aspect("equal")
        ax.set_title(titles.get(key, key))
    for ax in axes.ravel()[len(keys):]:
        ax.axis("off")
    fig.colorbar(pcm, ax=axes, label="σ", shrink=0.85)
    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
    return _finish(fig, None, show)

