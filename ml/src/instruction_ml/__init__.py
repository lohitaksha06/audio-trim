"""Instruction-understanding ML project (standalone; no Audelle imports)."""

from .schema import (
    AMOUNT_OPS,
    DEFAULT_FULL_MIX_OPS,
    OPERATIONS,
    OPERATION_KEYS,
    TARGETS,
    TARGET_KEYS,
    UNITS,
    UNITS_BY_OPERATION,
    Label,
    SchemaError,
    primary_op,
    validate_label,
)

__all__ = [
    "AMOUNT_OPS",
    "DEFAULT_FULL_MIX_OPS",
    "OPERATIONS",
    "OPERATION_KEYS",
    "TARGETS",
    "TARGET_KEYS",
    "UNITS",
    "UNITS_BY_OPERATION",
    "Label",
    "SchemaError",
    "primary_op",
    "validate_label",
]

__version__ = "0.1.0"
