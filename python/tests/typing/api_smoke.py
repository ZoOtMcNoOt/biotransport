"""Static consumer smoke test for the canonical typed API."""

import biotransport as bt


mesh = bt.StructuredMesh(10, 0.0, 1.0)
problem = bt.TransportProblem(mesh).diffusivity(0.1).initial_condition(1.0)
options = bt.SolveOptions.until(0.01)
native_result: bt.TransportResult = bt.solve_transport(problem, options)
python_result: bt.TransportResult = bt.solve(problem, end_time=0.01)
steps: int = python_result.diagnostics.steps
field = native_result.concentration
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
