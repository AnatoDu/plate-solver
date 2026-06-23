"""Подпакет model — определяющие соотношения уточнённой теории (КТН)."""

from .ktn import PlateMaterial, flexural_rigidity, refined_operator

__all__ = ["PlateMaterial", "flexural_rigidity", "refined_operator"]
