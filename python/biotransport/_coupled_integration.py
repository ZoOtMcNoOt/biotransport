"""Piecewise integration and volume-scaled external amount accounting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import bmat, csc_matrix

if TYPE_CHECKING:
    from .coupled import CompiledModel


def integrate(
    model: CompiledModel,
    final: float,
    times: np.ndarray,
    method: str,
    rtol: float,
    tolerances: np.ndarray,
    max_step: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int]]:
    """Advance concentrations and signed external transfers in the same solve.

    Each ledger state is cumulative moles divided by its connected control
    volume, so the integrator sees concentration units and the same absolute
    tolerance as that control volume. This avoids silently losing small-volume
    transfers by applying a concentration tolerance to a quantity in moles.
    """
    n = model._initial.size
    indices = model._external_indices
    q = len(indices)
    knots = [t for t in model.breakpoints if t < final]
    times = np.unique(np.concatenate((times, knots)))
    fields = np.empty((times.size, n))
    exchanges = np.zeros((times.size, q))
    fields[0] = model._initial
    current = np.concatenate((model._initial, np.zeros(q)))
    atol = np.concatenate((tolerances, tolerances[indices]))
    counts = np.zeros(3, dtype=int)
    if not final:
        return times, fields, exchanges, (0, 0, 0)

    ledger_jacobian = csc_matrix(
        (-model._external_loss, (np.arange(q), indices)), shape=(q, n)
    )
    ledger_zeros = csc_matrix((q, q))

    def augment(matrix):
        return (
            bmat([[matrix, None], [ledger_jacobian, ledger_zeros]], format="csc")
            if q
            else matrix
        )

    def jacobian(time, state):
        return augment(model.jacobian(time, state[:n]))

    jac = (
        (jacobian if q else model.jacobian)
        if model._nonlinear
        else augment(model._linear)
    )
    start = 0.0
    for stop in [*knots, final]:
        # Freeze the one-sided boundary law for this interval. At a jump the
        # finishing segment sees the old value, the next sees the new value.
        initial_bath = np.asarray(
            [schedule._value(start) for schedule in model._external_schedules]
        )
        final_bath = np.asarray(
            [schedule._value(stop, left=True) for schedule in model._external_schedules]
        )

        def rhs(time, state):
            fraction = (time - start) / (stop - start)
            bath = (1.0 - fraction) * initial_bath + fraction * final_bath
            forcing = model._external_gain * bath
            concentrations = state[:n]
            derivative = model._local_rhs(time, concentrations)
            if q:
                np.add.at(derivative, indices, forcing)
                ledger_rate = forcing - model._external_loss * concentrations[indices]
                derivative = np.concatenate((derivative, ledger_rate))
            if not np.all(np.isfinite(derivative)):
                raise ValueError("external transport rate is nonfinite")
            return derivative

        selected = np.flatnonzero((times > start) & (times <= stop))
        result = solve_ivp(
            rhs if q else model._local_rhs,
            (start, stop),
            current,
            method=method,
            t_eval=times[selected],
            jac=jac,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
        )
        if not result.success:
            raise RuntimeError(
                f"coupled integration failed on [{start:g}, {stop:g}]: {result.message}"
            )
        if result.t.size == 0 or result.t[-1] != stop:
            raise RuntimeError(
                "coupled integration did not reach the scheduled endpoint"
            )
        current = result.y[:, -1].copy()
        fields[selected] = result.y[:n].T
        exchanges[selected] = result.y[n:].T * model._external_volumes
        counts += (result.nfev, result.njev, result.nlu)
        start = stop
    if not np.all(np.isfinite(exchanges)):
        raise RuntimeError(
            "external amount overflows; rescale concentrations or volumes"
        )
    return times, fields, exchanges, (int(counts[0]), int(counts[1]), int(counts[2]))
