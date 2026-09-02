"""Balance accounting: dimensioned ledgers, audits and reconciliation.

The native :class:`BalanceLedger` records amounts, transfers and residuals in
explicit units so a conservation check is a statement about a named quantity
(mass, moles, energy, charge) rather than a bare number. :func:`reconcile_balances`
matches transfers between ledgers, and :func:`balance_residual` summarizes a
solver's own initial/final mass and boundary accounting.

These tools report; they never adjust a field to make a balance close.
This module only re-exports existing objects and implements no numerics.
"""

from ._core import (
    BalanceAudit,
    BalanceDimension,
    BalanceLedger,
    BalanceReconciliation,
    BalanceTerm,
    BalanceTransfer,
    BalanceTransferDirection,
    BalanceUnit,
    DimensionBalanceAudit,
    MatchedBalanceTransfer,
    balance_base_unit,
    balance_dimension_name,
    balance_unit_dimension,
    balance_unit_symbol,
    convert_balance_value,
    reconcile_balances,
)
from .reproducibility import balance_residual

__all__ = [
    "BalanceAudit",
    "BalanceDimension",
    "BalanceLedger",
    "BalanceReconciliation",
    "BalanceTerm",
    "BalanceTransfer",
    "BalanceTransferDirection",
    "BalanceUnit",
    "DimensionBalanceAudit",
    "MatchedBalanceTransfer",
    "balance_base_unit",
    "balance_dimension_name",
    "balance_residual",
    "balance_unit_dimension",
    "balance_unit_symbol",
    "convert_balance_value",
    "reconcile_balances",
]
