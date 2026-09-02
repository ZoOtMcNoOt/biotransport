Upgrading to 0.2.0
==================

Version 0.2.0 reorganizes the Python API around one vocabulary. Every retired
spelling keeps working for one deprecation window (removed in 0.4.0) and emits
a :class:`biotransport.BioTransportDeprecationWarning` naming its replacement;
the policy is in :download:`DEPRECATION_POLICY.md <../notes/DEPRECATION_POLICY.md>`.
Run your code with ``python -W error::biotransport.BioTransportDeprecationWarning``
to find every call that needs to change.

Nothing numerical moved. Every native solver produces bitwise-identical fields
for the same inputs, guarded by the golden fixtures in ``python/tests/golden``.

Time and results
----------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - 0.1 spelling
     - 0.2 spelling
   * - ``bt.run(problem, t_end)``
     - ``bt.solve(problem, end_time=...)``
   * - ``bt.solve(problem, t=..., dt=...)``
     - ``bt.solve(problem, end_time=..., time_step=...)``
   * - ``bt.run_checkpoints(mesh, times, D, ...)`` / ``CheckpointResult``
     - ``bt.solve(problem, save_times=[...]).snapshots`` (:class:`~biotransport.Snapshots`)
   * - ``TransportResult.solution``
     - ``TransportResult.concentration`` (or ``result.field`` on :class:`~biotransport.Result`)
   * - ``bt.ExplicitFD().run(problem, t)`` / ``RunResult`` / ``SolverStats``
     - ``bt.solve(problem, end_time=t)`` / :class:`~biotransport.Result` / ``result.diagnostics``
   * - ``solver.solve_until(t, maximum_dt=dt)``
     - ``solver.solve_until(t, time_step=dt)`` on every native stepping solver
   * - ``bt.integrate(problem, t_end)`` (implicit RK4)
     - ``bt.reference.integrate(problem, t_end, method="euler" | "heun" | "rk4")``

Boundaries
----------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - 0.1 spelling
     - 0.2 spelling
   * - ``problem.neumann(side, flux=...)``
     - ``problem.neumann(side, normal_derivative=...)`` (the value was always ``dc/dn``)
   * - ``NernstPlanckSolver.set_neumann_boundary(side, flux)``
     - ``set_outward_flux_boundary(side, outward_molar_flux)`` (a physical flux, not a derivative)
   * - ``solver.set_dirichlet_boundary(...)`` and friends
     - still available; the fluent ``dirichlet`` / ``neumann`` / ``robin`` / ``boundary`` / ``outward_flux`` verbs return the solver and refuse unsupported kinds
   * - spelling each side of a mesh
     - ``for side in bt.sides(mesh): ...``

Helpers, plotting and namespaces
--------------------------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - 0.1 spelling
     - 0.2 spelling
   * - ``bt.gaussian(mesh, ...)`` returning a ``list``
     - returns a flat ``float64`` array (also ``step``, ``uniform``, ``circle``, ``sinusoidal``, ``layered_1d``, ``SpatialField.build()``); call ``.tolist()`` if a list is required
   * - ``SpatialField.build_array()``
     - ``SpatialField.build()``
   * - ``bt.plot_1d_solution`` / ``plot_2d_solution`` / ``plot_2d_surface`` / ``plot_field`` / ``plot_1d`` / ``plot_2d``
     - ``bt.plot(mesh, values, kind=..., save_to=...)`` or ``result.plot()``
   * - ``bt.plot(mesh, solution=...)``
     - ``bt.plot(mesh, values)``
   * - ``bt.AdaptiveTimeStepper``, ``bt.solve_adaptive``, ``bt.RK4Integrator``, ``bt.solve_pulsatile``, ``bt.NewtonRaphsonSolver``, ``bt.NonlinearDiffusionSolver`` and their helpers
     - the same objects under :mod:`biotransport.reference`
   * - ``bt.write_vtk_series_with_metadata``
     - ``bt.write_vtk_series`` (or ``result.write_vtk``)
   * - ``bt.DiffusionProblem`` / ``LinearReactionDiffusionProblem`` / ``AdvectionDiffusionProblem``
     - ``bt.Problem``

``biotransport.__all__`` now lists the canonical path and the namespaces only.
Specialized native classes such as ``bt.DiffusionSolver`` remain attributes of
the root and do not warn; ``from biotransport import *`` and the public-surface
snapshot are the only things that see the smaller list.
