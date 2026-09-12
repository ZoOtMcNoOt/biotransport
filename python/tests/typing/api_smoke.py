"""Static consumer smoke test for the canonical typed API."""

import biotransport as bt


mesh = bt.StructuredMesh(10, 0.0, 1.0)
problem = bt.TransportProblem(mesh).diffusivity(0.1).initial_condition(1.0)
options = bt.SolveOptions.until(0.01)
native_result: bt.TransportResult = bt.solve_transport(problem, options)
native_plan: bt.TransportPlan = bt.plan_transport(problem, options)
planned_steps: int = native_plan.planned_steps
selected_step: float = native_plan.selected_time_step

# bt.solve() wraps the native result in a Solution, which keeps the native
# attribute names and adds geometry, saved frames, plotting and comparison.
python_result: bt.Solution = bt.solve(problem, end_time=0.01)
diagnostics = python_result.diagnostics
assert diagnostics is not None  # only None for a steady solve
steps: int = diagnostics.steps
field = native_result.concentration

# The richer surface, typed.
typed_mesh = bt.mesh_1d(10, 0.0, 1.0)
typed_problem = (
    bt.Problem(typed_mesh)
    .diffusivity(0.1)
    .initial(1.0)
    .dirichlet("left", 0.0)
    .sealed("right")
)
description: str = typed_problem.describe()
largest_step: float = typed_problem.stable_time_step()
history: bt.Solution = bt.solve(typed_problem, end_time=0.01, save_every=0.005)
saved_times: tuple[float, ...] = history.times
final_field = history.c
summary_text: str = history.summary()
error_report: bt.ErrorReport = history.compare(lambda x: 0.0 * x)
worst: float = error_report.max_abs
steady_solution: bt.Solution = bt.solve_steady(typed_problem)
is_steady: bool = steady_solution.steady
checkpoint_result: bt.CheckpointResult = bt.run_checkpoints(
    mesh,
    [0.01, 0.02],
    0.1,
    initial_condition=1.0,
    time_step=0.001,
)
checkpoint_steps: int = checkpoint_result.total_steps
checkpoint_field = checkpoint_result[0.02]
adaptive_config = bt.AdaptiveTimeStepperConfig(max_factor=1.5)
adaptive_max_factor: float = adaptive_config.max_factor


def grid_scalar_result(n: int) -> float:
    return 1.0 + 1.0 / n**2


grid_convergence: bt.ConvergenceResult = bt.run_convergence_study(
    grid_scalar_result, (10, 20, 40), verbose=False
)


def temporal_result_without_error(dt: float) -> tuple[float, None]:
    return 1.0 + dt**2, None


temporal_convergence: bt.ConvergenceResult = bt.temporal_convergence_study(
    temporal_result_without_error, (0.1, 0.05, 0.025), verbose=False
)

python_contract: bt.PythonNumericalContract = bt.get_python_numerical_contract("solve")
python_backend: bt.PythonBackend = python_contract.backend

flow_mesh = bt.StructuredMesh(4, 4, 0.0, 1.0, 0.0, 1.0)
flow = bt.NavierStokesSolver(flow_mesh, 1.0, 0.1)
flow_result = flow.solve_steps(1)
divergence: float = flow_result.divergence
