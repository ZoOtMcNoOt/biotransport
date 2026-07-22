"""Static consumer smoke test for the canonical typed API."""

import biotransport as bt


mesh = bt.StructuredMesh(10, 0.0, 1.0)
problem = bt.TransportProblem(mesh).diffusivity(0.1).initial_condition(1.0)
options = bt.SolveOptions.until(0.01)
native_result: bt.TransportResult = bt.solve_transport(problem, options)
python_result: bt.TransportResult = bt.solve(problem, end_time=0.01)
steps: int = python_result.diagnostics.steps
field = native_result.concentration

flow_mesh = bt.StructuredMesh(4, 4, 0.0, 1.0, 0.0, 1.0)
flow = bt.NavierStokesSolver(flow_mesh, 1.0, 0.1)
flow_result = flow.solve_steps(1)
divergence: float = flow_result.divergence
