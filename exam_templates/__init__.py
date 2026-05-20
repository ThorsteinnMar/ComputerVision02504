"""Exam-specific and predicted computation templates."""

from .exam2023_templates import *
from .exam2024_templates import *
from .predicted_question_templates import *

__all__ = [name for name in globals() if not name.startswith("_")]
