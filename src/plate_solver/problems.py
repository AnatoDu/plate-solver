"""Постановки краевых задач изгиба и одностороннего контакта.

Структуры данных, связывающие воедино геометрию (ω), краевые условия,
нагрузку и материал. Сами решатели — в подпакетах solver/ и contact/.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from .ktn import PlateMaterial

# Скалярное поле на плоскости: f(x, y) -> значение (numpy-совместимо).
Field = Callable[["float", "float"], "float"]


class EdgeCondition(Enum):
    """Тип закрепления края пластины."""

    CLAMPED = "clamped"            # защемление
    SIMPLY_SUPPORTED = "simply"    # шарнирное опирание
    FREE = "free"                  # свободный край
    REINFORCED = "reinforced"      # подкреплённый край (полудеформационные величины)


@dataclass
class BendingProblem:
    """Задача изгиба пластины произвольного очертания.

    Attributes
    ----------
    omega : R-функция области, ω(x, y) > 0 внутри, = 0 на границе.
    load : распределённая поперечная нагрузка q(x, y).
    material : материал пластины.
    edge : условие на внешнем контуре.
    """

    omega: Field
    load: Field
    material: PlateMaterial
    edge: EdgeCondition = EdgeCondition.CLAMPED


@dataclass
class ContactProblem:
    """Контактная задача со свободной (заранее неизвестной) границей.

    К задаче изгиба добавляется препятствие (основание/штамп/вторая пластина),
    заданное зазором gap(x, y), и условие односторонней связи
    (комплементарность): прогиб <= зазор, реакция >= 0, их произведение = 0.
    Решается методом обобщённой реакции (см. contact/mor.py).
    """

    bending: BendingProblem
    gap: Field
