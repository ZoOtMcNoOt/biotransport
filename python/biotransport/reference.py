"""Python reference and legacy numerics, kept apart from the native path.

Everything here is implemented in Python and carries its own
:class:`~biotransport.PythonNumericalContract` (see
:func:`biotransport.get_python_numerical_contract`). These surfaces exist as
readable references, teaching material, or legacy time-stepping loops; they do
not claim the canonical solver's performance or its verified scope, and their
evidence applies only to the equations, dimensions and boundary conditions
each contract states.

- :mod:`biotransport.adaptive`: error-controlled explicit stepping for 1D
  uniform diffusion with Dirichlet ends.
- :mod:`biotransport.time_integrators`: ``integrate(method=...)`` with the
  native Euler path or the Heun/RK4 reference integrators.
- :mod:`biotransport.pulsatile`: time-varying boundary drivers and a Python
  diffusion loop that applies them.
- :mod:`biotransport.newton_raphson`: Newton iteration for nonlinear
  steady-state reaction-diffusion problems.

This module only re-exports those objects; it implements no numerics.
"""

from .adaptive import (
    AdaptiveResult,
    AdaptiveTimeStepper,
    AdaptiveTimeStepperConfig,
    solve_adaptive,
)
from .newton_raphson import (
    ConvergenceCriterion,
    NewtonEvaluationError,
    NewtonLinearSolveError,
    NewtonLineSearchError,
    NewtonRaphsonSolver,
    NewtonResult,
    NewtonSolverError,
    NonlinearDiffusionSolver,
    bistable,
    exponential_decay,
    hill_kinetics,
    michaelis_menten,
)
from .pulsatile import (
    ArterialPressureBC,
    CardiacOutputBC,
    CompositeBC,
    ConstantBC,
    CustomBC,
    DrugInfusionBC,
    PulsatileBC,
    PulsatileResult,
    RampBC,
    RespiratoryBC,
    SinusoidalBC,
    SquareWaveBC,
    StepBC,
    VenousPressureBC,
    heart_rate_to_period,
    period_to_heart_rate,
    sample_waveform,
    solve_pulsatile,
)
from .time_integrators import (
    HeunIntegrator,
    IntegrationResult,
    RK4Integrator,
    euler_step,
    heun_step,
    integrate,
    rk4_step,
)

__all__ = [
    # adaptive
    "AdaptiveResult",
    "AdaptiveTimeStepper",
    "AdaptiveTimeStepperConfig",
    "solve_adaptive",
    # time integrators
    "HeunIntegrator",
    "IntegrationResult",
    "RK4Integrator",
    "euler_step",
    "heun_step",
    "integrate",
    "rk4_step",
    # pulsatile boundary drivers
    "ArterialPressureBC",
    "CardiacOutputBC",
    "CompositeBC",
    "ConstantBC",
    "CustomBC",
    "DrugInfusionBC",
    "PulsatileBC",
    "PulsatileResult",
    "RampBC",
    "RespiratoryBC",
    "SinusoidalBC",
    "SquareWaveBC",
    "StepBC",
    "VenousPressureBC",
    "heart_rate_to_period",
    "period_to_heart_rate",
    "sample_waveform",
    "solve_pulsatile",
    # Newton-Raphson
    "ConvergenceCriterion",
    "NewtonEvaluationError",
    "NewtonLinearSolveError",
    "NewtonLineSearchError",
    "NewtonRaphsonSolver",
    "NewtonResult",
    "NewtonSolverError",
    "NonlinearDiffusionSolver",
    "bistable",
    "exponential_decay",
    "hill_kinetics",
    "michaelis_menten",
]
