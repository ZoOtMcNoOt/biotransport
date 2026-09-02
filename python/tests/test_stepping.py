"""The shared solve_until lifecycle and boundary verbs on native stepping solvers."""

from __future__ import annotations

import numpy as np
import pytest

import biotransport as bt
from biotransport import BioTransportDeprecationWarning, Result
from biotransport.contracts import get_contract, get_python_numerical_contract
from biotransport.stepping import StepDiagnostics, registered_stepping_classes

MESH_1D = bt.mesh_1d(12)
MESH_2D = bt.mesh_2d(6, 5)
MESH_3D = bt.StructuredMesh3D(4, 3, 3, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)


def _bump_1d(mesh=MESH_1D):
    return np.asarray(bt.gaussian(mesh, center=0.5, width=0.15), dtype=float)


def _factory(cls):
    """Return (solver, explicit_time_step) for one registered class."""
    name = cls.__name__
    if name == "DiffusionSolver":
        s = cls(MESH_1D, 0.01)
        s.set_initial_condition(_bump_1d())
        return s, None
    if name == "ReactionDiffusionSolver":
        s = cls(MESH_1D, 0.01, lambda c, x, y, t: -0.1 * c)
        s.set_initial_condition(_bump_1d())
        return s, 1e-3
    if name == "LinearReactionDiffusionSolver":
        s = cls(MESH_1D, 0.01, 0.1)
        s.set_initial_condition(_bump_1d())
        return s, 1e-3
    if name == "LogisticReactionDiffusionSolver":
        s = cls(MESH_1D, 0.01, 0.5, 1.0)
        s.set_initial_condition(0.5 * _bump_1d())
        return s, 1e-3
    if name == "MichaelisMentenReactionDiffusionSolver":
        s = cls(MESH_1D, 0.01, 0.2, 0.5)
        s.set_initial_condition(_bump_1d())
        return s, 1e-3
    if name == "MaskedMichaelisMentenReactionDiffusionSolver":
        mask = [1] + [0] * (MESH_1D.num_nodes() - 1)
        s = cls(MESH_1D, 0.01, 0.2, 0.5, mask, 1.0)
        s.set_initial_condition(_bump_1d())
        return s, 1e-3
    if name == "ConstantSourceReactionDiffusionSolver":
        s = cls(MESH_1D, 0.01, 0.1)
        s.set_initial_condition(_bump_1d())
        return s, 1e-3
    if name == "AdvectionDiffusionSolver":
        s = cls(MESH_1D, 0.01, 0.1, 0.0, bt.AdvectionScheme.UPWIND)
        s.set_initial_condition(_bump_1d())
        return s, 1e-3
    if name == "DiffusionSolver3D":
        s = cls(MESH_3D, 0.01)
        s.set_initial_condition(np.full(MESH_3D.num_nodes(), 1.0))
        return s, None
    if name == "LinearReactionDiffusionSolver3D":
        s = cls(MESH_3D, 0.01, 0.1)
        s.set_initial_condition(np.full(MESH_3D.num_nodes(), 1.0))
        return s, None
    if name == "CrankNicolsonDiffusion":
        s = cls(MESH_1D, 0.01)
        s.set_initial_condition(_bump_1d())
        return s, 2e-3
    if name == "ADIDiffusion2D":
        s = cls(MESH_2D, 0.01)
        s.set_initial_condition(np.full(MESH_2D.num_nodes(), 1.0))
        return s, 2e-3
    if name == "ADIDiffusion3D":
        s = cls(MESH_3D, 0.01)
        s.set_initial_condition(np.full(MESH_3D.num_nodes(), 1.0))
        return s, 2e-3
    if name == "ImplicitDiffusion2D":
        if not bt.sparse_matrix_available():
            pytest.skip("Eigen sparse backend unavailable")
        s = cls(MESH_2D, 0.01)
        s.set_initial_condition(np.full(MESH_2D.num_nodes(), 1.0))
        return s, 2e-3
    if name == "ImplicitDiffusion3D":
        if not bt.sparse_matrix_available():
            pytest.skip("Eigen sparse backend unavailable")
        s = cls(MESH_3D, 0.01)
        s.set_initial_condition(np.full(MESH_3D.num_nodes(), 1.0))
        return s, 2e-3
    if name == "NonuniformDiffusion1D":
        mesh = bt.NonuniformMesh1D([0.0, 0.1, 0.25, 0.5, 1.0])
        s = cls(mesh, 0.01)
        s.set_initial_condition([1.0, 0.8, 0.5, 0.2, 0.0])
        return s, None
    if name == "MultiSpeciesSolver":
        s = cls(MESH_1D, [0.01, 0.01])
        s.set_reaction_model(bt.LotkaVolterraReaction(0.5, 0.02, 0.3, 0.01, 100.0))
        s.set_uniform_initial_condition(0, 10.0)
        s.set_uniform_initial_condition(1, 5.0)
        return s, None
    if name == "NernstPlanckSolver":
        s = cls(MESH_1D, bt.IonSpecies("Na+", 1, 1.33e-9))
        s.set_initial_condition(np.full(MESH_1D.num_nodes(), 1.0))
        s.set_uniform_field(10.0)
        return s, None
    if name == "MultiIonSolver":
        s = cls(
            MESH_1D,
            [bt.IonSpecies("Na+", 1, 1.33e-9), bt.IonSpecies("Cl-", -1, 2.03e-9)],
        )
        s.set_initial_condition(0, np.full(MESH_1D.num_nodes(), 1.0))
        s.set_initial_condition(1, np.full(MESH_1D.num_nodes(), 1.0))
        s.set_uniform_field(10.0)
        return s, None
    raise AssertionError(f"no factory for {name}")


CLASSES = registered_stepping_classes()
IDS = [cls.__name__ for cls in CLASSES]


def test_registry_covers_every_transient_native_solver() -> None:
    names = set(IDS)
    assert len(names) == len(CLASSES) == 19
    for cls in CLASSES:
        assert hasattr(cls, "solve_until") and hasattr(cls, "dirichlet")
        assert hasattr(cls, "time") and hasattr(cls, "mesh")
    contract = get_python_numerical_contract("solve_until")
    assert contract.contract_id == "python.native_adapter.stepping"


def test_long_runs_tolerate_the_native_clock_accumulation_but_not_wrong_steps() -> None:
    """Thousands of native ``time += dt`` additions drift by more than 64 ulp."""
    mesh = bt.StructuredMesh(50, 0.0, 1.0e-3)
    solver = bt.DiffusionSolver(mesh, 1.0e-9)
    solver.set_initial_condition([0.0] * mesh.num_nodes())
    solver.dirichlet(bt.Boundary.Left, 1.0).neumann(bt.Boundary.Right, 0.0)

    result = solver.solve_until(600.0, save_times=[60.0, 300.0])

    assert result.time == 600.0
    assert result.steps > 1000
    assert result.snapshots.times == (60.0, 300.0)
    assert abs(result.diagnostics.final_time - 600.0) <= result.steps * 4e-16 * 600.0

    # A solver whose clock is off by even one whole step is still rejected.
    class _OffByOneStep(bt.DiffusionSolver):
        def solve(self, dt: float, num_steps: int) -> None:  # type: ignore[override]
            super().solve(dt, num_steps + 1)

    broken = _OffByOneStep(mesh, 1.0e-9)
    broken.set_initial_condition([0.0] * mesh.num_nodes())
    with pytest.raises(RuntimeError, match="did not land"):
        broken.solve_until(1.0)


@pytest.mark.parametrize("cls", CLASSES, ids=IDS)
def test_solve_until_lands_exactly_and_matches_manual_stepping(cls) -> None:
    solver, explicit = _factory(cls)
    reference, _ = _factory(cls)
    end_time = 0.02
    kwargs = {} if explicit is None else {"time_step": explicit}

    result = solver.solve_until(end_time, **kwargs)

    assert isinstance(result, Result)
    assert result.time == end_time
    assert result.contract in {c.contract_id for c in [get_contract(cls.__name__)]}
    assert isinstance(result.diagnostics, StepDiagnostics)
    assert result.diagnostics.solver == cls.__name__
    assert result.steps == result.diagnostics.steps > 0
    assert result.diagnostics.segments == 1
    assert abs(solver.time() - end_time) <= 1e-13
    assert result.mesh is not None and result.mesh.num_nodes() == result.field.size
    assert result.native is solver
    if explicit is None:
        assert result.diagnostics.automatic_time_step
        assert result.diagnostics.stability_limit is not None
        assert (
            result.diagnostics.maximum_time_step
            <= 0.8 * result.diagnostics.stability_limit
        )
    else:
        assert not result.diagnostics.automatic_time_step
        assert result.diagnostics.maximum_time_step <= explicit * (1 + 1e-12)

    # Manual stepping with the same substep reproduces the field.
    dt = result.diagnostics.maximum_time_step
    steps = result.steps
    if hasattr(reference, "_native_solve_until"):
        reference._native_solve_until(end_time, dt)
    else:
        reference.solve(dt, steps)
    fields = bt.stepping._PROTOCOLS[cls].fields(reference)
    for name, values in result.fields.items():
        np.testing.assert_allclose(values, fields[name], rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("cls", CLASSES, ids=IDS)
def test_save_times_record_snapshots_at_requested_clocks(cls) -> None:
    solver, explicit = _factory(cls)
    kwargs = {} if explicit is None else {"time_step": explicit}
    result = solver.solve_until(0.03, save_times=[0.01, 0.02], **kwargs)

    assert result.snapshots.times == (0.01, 0.02)
    assert result.diagnostics.segments == 3
    assert result.snapshots[0.02].shape == result.field.shape
    assert not result.snapshots[0.01].flags.writeable
    # A second call continues from the current clock.
    later = solver.solve_until(0.04, **kwargs)
    assert later.diagnostics.start_time == pytest.approx(0.03)
    assert later.time == 0.04
    with pytest.raises(ValueError, match="must not precede"):
        solver.solve_until(0.01, **kwargs)


def test_classes_without_a_certificate_require_time_step() -> None:
    for cls in CLASSES:
        protocol = bt.stepping._PROTOCOLS[cls]
        solver, explicit = _factory(cls)
        if protocol.stability is None:
            assert explicit is not None
            with pytest.raises(TypeError, match="time_step"):
                solver.solve_until(0.01)
        else:
            solver.solve_until(0.001)


def test_boundary_verbs_forward_or_refuse() -> None:
    diffusion = bt.DiffusionSolver(MESH_1D, 0.01)
    assert diffusion.dirichlet(bt.Boundary.Left, 1.0) is diffusion
    assert diffusion.neumann(bt.Boundary.Right, 0.0) is diffusion
    assert diffusion.robin(bt.Boundary.Right, 1.0, 1.0, 0.0) is diffusion
    assert (
        diffusion.boundary(bt.Boundary.Left, bt.BoundaryCondition.dirichlet(2.0))
        is diffusion
    )
    with pytest.raises(TypeError, match="use neumann"):
        diffusion.outward_flux(bt.Boundary.Left, 1.0)
    with pytest.raises(TypeError, match="species"):
        diffusion.dirichlet(bt.Boundary.Left, 1.0, species=0)

    crank = bt.CrankNicolsonDiffusion(MESH_1D, 0.01)
    assert (
        crank.dirichlet(bt.Boundary.Left, 0.0).neumann(bt.Boundary.Right, 0.0) is crank
    )
    with pytest.raises(TypeError, match="Robin"):
        crank.robin(bt.Boundary.Left, 1.0, 1.0, 0.0)

    nernst = bt.NernstPlanckSolver(MESH_1D, bt.IonSpecies("Na+", 1, 1.33e-9))
    assert nernst.dirichlet(bt.Boundary.Left, 1.0) is nernst
    assert nernst.outward_flux(bt.Boundary.Right, 1.0e-9) is nernst
    with pytest.raises(TypeError, match="outward_flux"):
        nernst.neumann(bt.Boundary.Right, 0.0)

    multi = bt.MultiSpeciesSolver(MESH_1D, [0.01, 0.01])
    assert multi.dirichlet(bt.Boundary.Left, 1.0) is multi  # all species
    assert multi.neumann(bt.Boundary.Right, 0.0, species=1) is multi
    with pytest.raises(TypeError, match="Robin"):
        multi.robin(bt.Boundary.Left, 1.0, 1.0, 0.0)

    ions = bt.MultiIonSolver(MESH_1D, [bt.IonSpecies("Na+", 1, 1.33e-9)])
    assert ions.outward_flux(bt.Boundary.Right, 0.0, species=0) is ions
    with pytest.raises(TypeError, match="outward_flux"):
        ions.neumann(bt.Boundary.Right, 0.0)


def test_fluent_verbs_install_the_same_condition_as_the_native_setters() -> None:
    fluent = bt.DiffusionSolver(MESH_1D, 0.01)
    native = bt.DiffusionSolver(MESH_1D, 0.01)
    for solver in (fluent, native):
        solver.set_initial_condition(_bump_1d())
    fluent.dirichlet(bt.Boundary.Left, 0.25).neumann(bt.Boundary.Right, 0.5)
    native.set_dirichlet_boundary(bt.Boundary.Left, 0.25)
    native.set_neumann_boundary(bt.Boundary.Right, 0.5)

    result = fluent.solve_until(0.01)
    native.solve(result.diagnostics.maximum_time_step, result.steps)
    np.testing.assert_array_equal(fluent.solution(), native.solution())


def test_free_function_rejects_unregistered_objects_and_bad_inputs() -> None:
    with pytest.raises(TypeError, match="not a registered stepping solver"):
        bt.solve_until(object(), 1.0)
    solver, _ = _factory(bt.DiffusionSolver)
    with pytest.raises(ValueError, match="positive"):
        solver.solve_until(0.01, time_step=0.0)
    with pytest.raises(ValueError, match="strictly increasing"):
        solver.solve_until(0.03, save_times=[0.02, 0.01])
    with pytest.raises(ValueError, match="within"):
        solver.solve_until(0.03, save_times=[0.05])
    with pytest.raises(TypeError, match="unexpected keyword"):
        solver.solve_until(0.03, bogus=1)


def test_maximum_dt_spelling_is_deprecated() -> None:
    solver, _ = _factory(bt.MultiSpeciesSolver)
    with pytest.warns(BioTransportDeprecationWarning, match="time_step"):
        result = solver.solve_until(0.01, maximum_dt=0.001)
    assert result.time == 0.01
    with pytest.raises(TypeError, match="either time_step or maximum_dt"):
        solver.solve_until(0.02, time_step=0.001, maximum_dt=0.001)


def test_multi_field_results_name_their_fields() -> None:
    ions, _ = _factory(bt.MultiIonSolver)
    result = ions.solve_until(0.001)
    assert set(result.fields) == {"Na+", "Cl-", "potential"}
    assert result.primary == "potential"
    with pytest.raises(AttributeError, match="available fields"):
        result.concentration
    species, _ = _factory(bt.MultiSpeciesSolver)
    result = species.solve_until(0.001)
    assert set(result.fields) == {"species_0", "species_1"}
    assert result.field is result.fields["species_0"]
