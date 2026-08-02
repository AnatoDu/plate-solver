r"""export.py — результирующие усилия как поля + экспорт в VTK (§9.4 ТЗ v0.6.0).

Постобработка/экспорт результата расчёта (:class:`~plate_solver.dispatch.Result`)
для независимой сверки (ParaView) и как материал под рисунки:

* :func:`forces_on_grid` — результирующие усилия на фоновой сетке единым
  словарём: изгибные моменты ``Mx, My, Mxy`` (всегда), мембранные усилия
  ``Nx, Ny, Nxy`` (для нелинейных теорий, если решатель их отдаёт), обобщённая
  реакция ``r`` (для контактных задач);
* :func:`to_vtk` — запись сеточных полей в LEGACY-формат VTK (STRUCTURED_POINTS,
  ASCII) БЕЗ новых зависимостей: ParaView читает напрямую, вне-Ω помечено ``NaN``.

Экспорт чисто постобработочный: решение не меняет, лишь перекладывает поля.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def forces_on_grid(result) -> dict[str, np.ndarray]:
    r"""Результирующие усилия на фоновой сетке (§9.4) единым словарём.

    Всегда: изгибные моменты ``Mx, My, Mxy`` (:meth:`Result.moments_on_grid`).
    Дополнительно (если есть): мембранные усилия ``Nx, Ny, Nxy`` (нелинейные
    теории) и обобщённая реакция ``r`` (контакт). Все поля — ``NaN`` вне Ω.
    """
    Mx, My, Mxy = result.moments_on_grid()
    out: dict[str, np.ndarray] = {"Mx": Mx, "My": My, "Mxy": Mxy}
    # Мембранные усилия N (нелинейные теории): полный набор коэффициентов
    # (cu, cv, cw) лежит в ``result._karman_ref`` (изгиб — KarmanResult;
    # нелинейный контакт — снимок с восстановленной мембраной, v0.6.6);
    # восстановление поля — membrane_forces_at(cu, cv, cw, X, Y).
    plate = getattr(result, "_plate_ref", None)
    kr = getattr(result, "_karman_ref", None)
    mem = getattr(plate, "membrane_forces_at", None)
    cu = getattr(kr, "cu", None)
    cv = getattr(kr, "cv", None)
    cw = getattr(kr, "cw", None)
    if mem is not None and cu is not None and cv is not None and cw is not None:
        Xg, Yg = result.Xg, result.Yg
        inside = _inside_mask(result)
        nx, ny, nxy = mem(cu, cv, cw, Xg[inside], Yg[inside])
        Nx = np.full(Xg.shape, np.nan)
        Ny = np.full(Xg.shape, np.nan)
        Nxy = np.full(Xg.shape, np.nan)
        Nx[inside], Ny[inside], Nxy[inside] = nx, ny, nxy
        out["Nx"], out["Ny"], out["Nxy"] = Nx, Ny, Nxy
    # обобщённая реакция контакта — в спутнике result.contact
    contact = getattr(result, "contact", None)
    r_grid = getattr(contact, "r_grid", None) if contact is not None else None
    if r_grid is not None:
        out["r"] = np.asarray(r_grid, float)
    return out


def shear_forces_on_grid(result) -> dict[str, np.ndarray]:
    r"""Перерезывающие силы ``Qx, Qy`` на фоновой сетке из РАВНОВЕСИЯ моментов.

    .. math:: Q_x = \partial M_x/\partial x + \partial M_{xy}/\partial y,\qquad
              Q_y = \partial M_{xy}/\partial x + \partial M_y/\partial y.

    Производные — конечные разности на фоновой сетке (``np.gradient``,
    порядок O(Δ²)); поля моментов спектрально гладкие, поэтому точность
    задаёт сетка ``grid_n`` (увеличивается ключом ``--grid``/``regrid``).
    Верификация — точное осесимметричное равновесие ``Q_r = q·r/2`` при
    равномерной нагрузке (не зависит от КУ; tests/test_fields_export.py).
    Вне Ω (и в приграничном поясе шириной в шаг сетки, где разности
    пересекают границу) — ``NaN``.
    """
    Mx, My, Mxy = result.moments_on_grid()
    x = np.asarray(result.Xg[0, :], float)
    y = np.asarray(result.Yg[:, 0], float)
    dMx_dx = np.gradient(Mx, x, axis=1)
    dMxy_dy = np.gradient(Mxy, y, axis=0)
    dMxy_dx = np.gradient(Mxy, x, axis=1)
    dMy_dy = np.gradient(My, y, axis=0)
    Qx = dMx_dx + dMxy_dy
    Qy = dMxy_dx + dMy_dy
    return {"Qx": Qx, "Qy": Qy}


def _inside_mask(result) -> np.ndarray:
    """Маска узлов сетки внутри Ω (по знаку ω области решателя)."""
    plate = getattr(result, "_plate_ref", None)
    dom = getattr(plate, "domain", None)
    if dom is not None:
        return dom.omega(result.Xg, result.Yg) > 0.0
    return ~np.isnan(result.w_grid)                      # запас: по определённости w


def to_vtk(result, path) -> Path:
    r"""Записать сеточные поля в LEGACY-VTK (STRUCTURED_POINTS, ASCII) для ParaView.

    Скалярные поля точек: ``w`` (прогиб) и все усилия из :func:`forces_on_grid`
    (``Mx, My, Mxy``, при наличии ``Nx, Ny, Nxy, r``). Сетка равномерная
    (``ORIGIN``/``SPACING`` из ``Xg, Yg``), порядок точек — x быстрее y (канон
    VTK). Вне-Ω значения — ``NaN`` (ParaView маскирует). Возвращает путь файла.
    """
    Xg, Yg = np.asarray(result.Xg), np.asarray(result.Yg)
    ny, nx = Xg.shape
    x0, y0 = float(Xg[0, 0]), float(Yg[0, 0])
    dx = float(Xg[0, 1] - Xg[0, 0]) if nx > 1 else 1.0
    dy = float(Yg[1, 0] - Yg[0, 0]) if ny > 1 else 1.0

    fields: dict[str, np.ndarray] = {"w": np.asarray(result.w_grid, float)}
    fields.update(forces_on_grid(result))

    lines = ["# vtk DataFile Version 3.0", "plate-solver export", "ASCII",
             "DATASET STRUCTURED_POINTS",
             f"DIMENSIONS {nx} {ny} 1",
             f"ORIGIN {x0:.10g} {y0:.10g} 0",
             f"SPACING {dx:.10g} {dy:.10g} 1",
             f"POINT_DATA {nx * ny}"]
    for name, field in fields.items():
        arr = np.asarray(field, float)
        if arr.shape != (ny, nx):                        # экспортируем только сеточные поля
            continue
        lines.append(f"SCALARS {name} double 1")
        lines.append("LOOKUP_TABLE default")
        # порядок STRUCTURED_POINTS: x быстрее (внутри строки), затем y ⇒ row-major
        lines.extend(f"{v:.10g}" for v in arr.ravel(order="C"))

    path = Path(path)
    if path.suffix != ".vtk":
        path = path.with_suffix(".vtk")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


__all__ = ["forces_on_grid", "shear_forces_on_grid", "to_vtk"]
