"""
Tests for Newton-Raphson nonlinear solver.
"""

import numpy as np
import pytest

import biotransport as bt
from biotransport.newton_raphson import (
    NewtonRaphsonSolver,
    NonlinearDiffusionSolver,
    ConvergenceCriterion,
    michaelis_menten,
    hill_kinetics,
    bistable,
    exponential_decay,
)


class TestNewtonRaphsonSolver:
    """Tests for the general Newton-Raphson solver."""

    def test_simple_scalar_root(self):
        """Test finding root of x³ - x - 1 = 0."""

        def residual(u):
            return u**3 - u - 1

        def jacobian(u):
            return np.array([[3 * u[0] ** 2 - 1]])

        solver = NewtonRaphsonSolver(residual, jacobian, n=1)
        result = solver.solve(np.array([1.5]))

        assert result.converged
        # Real root is approximately 1.3247
        assert np.abs(result.solution[0] - 1.3247179572) < 1e-6
        assert result.iterations < 10

    def test_quadratic_root(self):
        """Test finding root of x² - 4 = 0 (x = ±2)."""

        def residual(u):
            return u**2 - 4

        solver = NewtonRaphsonSolver(residual, n=1)

        # Starting from positive, should find +2
        result = solver.solve(np.array([3.0]))
        assert result.converged
        assert np.abs(result.solution[0] - 2.0) < 1e-8

        # Starting from negative, should find -2
        result = solver.solve(np.array([-3.0]))
        assert result.converged
        assert np.abs(result.solution[0] + 2.0) < 1e-8

    def test_2d_system(self):
        """Test solving 2D nonlinear system."""

        # x² + y² = 1, x - y = 0  =>  x = y = ±1/√2
        def residual(u):
            x, y = u
            return np.array([x**2 + y**2 - 1, x - y])

        def jacobian(u):
            x, y = u
            return np.array([[2 * x, 2 * y], [1, -1]])

        solver = NewtonRaphsonSolver(residual, jacobian, n=2)
        result = solver.solve(np.array([0.5, 0.5]))

        assert result.converged
        expected = 1.0 / np.sqrt(2)
        assert np.abs(result.solution[0] - expected) < 1e-8
        assert np.abs(result.solution[1] - expected) < 1e-8

    def test_finite_difference_jacobian(self):
        """Test that FD Jacobian works when analytical not provided."""

        def residual(u):
            return u**2 - 4

        # No Jacobian provided - should use FD
        solver = NewtonRaphsonSolver(residual, n=1)
        result = solver.solve(np.array([3.0]))

        assert result.converged
        assert np.abs(result.solution[0] - 2.0) < 1e-6

    def test_max_iterations_failure(self):
        """Test that solver reports non-convergence when max_iter reached."""

        def residual(u):
            return u**2 - 4

        solver = NewtonRaphsonSolver(residual, n=1)
        solver.set_parameters(max_iterations=1, tol_residual=1e-20)
        result = solver.solve(np.array([100.0]))  # Far from root

        assert not result.converged
        assert result.iterations == 1

    def test_convergence_criterion_residual_only(self):
        """Test residual-only convergence criterion."""

        def residual(u):
            return u**2 - 4

        solver = NewtonRaphsonSolver(residual, n=1)
        solver.set_parameters(
            criterion=ConvergenceCriterion.RESIDUAL,
            tol_residual=1e-6,
            tol_update=1e-20,  # Very tight, shouldn't matter
        )
        result = solver.solve(np.array([3.0]))

        assert result.converged
        assert result.residual_norm < 1e-6

    def test_residual_history_tracked(self):
        """Test that residual history is recorded."""

        def residual(u):
            return u**2 - 4

        solver = NewtonRaphsonSolver(residual, n=1)
        result = solver.solve(np.array([10.0]))

        assert len(result.residual_history) > 1
        # Should decrease monotonically (mostly)
        assert result.residual_history[-1] < result.residual_history[0]

    def test_line_search_disabled(self):
        """Test solver works with line search disabled."""

        def residual(u):
            return u**2 - 4

        solver = NewtonRaphsonSolver(residual, n=1)
        solver.set_parameters(use_line_search=False, damping=1.0)
        result = solver.solve(np.array([3.0]))

        assert result.converged
        assert np.abs(result.solution[0] - 2.0) < 1e-8

    def test_damping_factor(self):
        """Test that damping factor is applied."""

        def residual(u):
            return u**2 - 4

        solver = NewtonRaphsonSolver(residual, n=1)
        solver.set_parameters(use_line_search=False, damping=0.5)
        result = solver.solve(np.array([3.0]))

        # Should still converge, just slower
        assert result.converged


class TestNonlinearDiffusionSolver:
    """Tests for the nonlinear reaction-diffusion solver."""

    def test_linear_source_recovers_analytical(self):
        """Test with linear source term (Poisson equation)."""
        # -D*u'' = f(x) with u(0)=0, u(1)=0
        # f(x) = sin(πx), exact: u = sin(πx) / (π²*D)
        mesh = bt.mesh_1d(50, 0, 1)
        x = bt.x_nodes(mesh)
        D = 1.0

        source = np.sin(np.pi * x)
        exact = np.sin(np.pi * x) / (np.pi**2 * D)

        solver = NonlinearDiffusionSolver(mesh, D=D)
        solver.set_source(source)
        solver.set_boundary(bt.Boundary.Left, 0.0)
        solver.set_boundary(bt.Boundary.Right, 0.0)

        result = solver.solve(np.zeros(len(x)))

        assert result.converged
        error = np.max(np.abs(result.solution[1:-1] - exact[1:-1]))
        assert error < 1e-3  # Second-order spatial error

    def test_exponential_decay_reaction(self):
        """Test with linear reaction (exponential decay)."""
        # -D*u'' + k*u = 0 with u(0)=1, u(L)=0
        # This is a Helmholtz equation
        mesh = bt.mesh_1d(100, 0, 1)
        x = bt.x_nodes(mesh)
        D = 1.0
        k = 10.0

        reaction, deriv = exponential_decay(k)

        solver = NonlinearDiffusionSolver(mesh, D=D)
        solver.set_reaction(reaction, deriv)
        solver.set_boundary(bt.Boundary.Left, 1.0)
        solver.set_boundary(bt.Boundary.Right, 0.0)

        initial = np.linspace(1, 0, len(x))
        result = solver.solve(initial)

        assert result.converged
        # Solution should be positive and decay
        assert np.all(result.solution >= -1e-10)
        assert result.solution[0] == pytest.approx(1.0, abs=1e-10)

    def test_michaelis_menten_convergence(self):
        """Test solver converges with Michaelis-Menten kinetics."""
        mesh = bt.mesh_1d(50, 0, 1)
        x = bt.x_nodes(mesh)

        vmax, km = 1.0, 0.5
        reaction, deriv = michaelis_menten(vmax, km)

        solver = NonlinearDiffusionSolver(mesh, D=0.1)
        solver.set_reaction(reaction, deriv)
        solver.set_boundary(bt.Boundary.Left, 1.0)  # Substrate at left
        solver.set_boundary(bt.Boundary.Right, 0.0)

        initial = np.linspace(1, 0, len(x))
        result = solver.solve(initial)

        assert result.converged
        assert result.residual_norm < 1e-8

    def test_hill_kinetics_convergence(self):
        """Test solver converges with Hill kinetics."""
        mesh = bt.mesh_1d(50, 0, 1)
        x = bt.x_nodes(mesh)

        vmax, km, n = 1.0, 0.5, 2
        reaction, deriv = hill_kinetics(vmax, km, n)

        solver = NonlinearDiffusionSolver(mesh, D=0.1)
        solver.set_reaction(reaction, deriv)
        solver.set_boundary(bt.Boundary.Left, 1.0)
        solver.set_boundary(bt.Boundary.Right, 0.1)

        initial = np.linspace(1, 0.1, len(x))
        result = solver.solve(initial)

        assert result.converged

    def test_neumann_boundary_condition(self):
        """Test Neumann (flux) boundary condition."""
        # -u'' = 0 with u'(0) = 1, u(1) = 0
        # Exact: u = 1 - x
        mesh = bt.mesh_1d(50, 0, 1)
        x = bt.x_nodes(mesh)

        solver = NonlinearDiffusionSolver(mesh, D=1.0)
        solver.set_boundary(bt.Boundary.Left, 1.0, bc_type="neumann")
        solver.set_boundary(bt.Boundary.Right, 0.0, bc_type="dirichlet")

        result = solver.solve(np.zeros(len(x)))

        assert result.converged
        # Check slope near left boundary
        slope = (result.solution[1] - result.solution[0]) / mesh.dx()
        assert slope == pytest.approx(1.0, rel=0.05)

    def test_2d_poisson(self):
        """Test 2D nonlinear diffusion (Poisson equation)."""
        mesh = bt.mesh_2d(20, 20, 0, 1, 0, 1)
        x = bt.x_nodes(mesh)
        y = bt.y_nodes(mesh)
        X, Y = np.meshgrid(x, y)

        # -∇²u = 2π²sin(πx)sin(πy) with u=0 on boundary
        # Exact: u = sin(πx)sin(πy)
        source = 2 * np.pi**2 * np.sin(np.pi * X) * np.sin(np.pi * Y)
        exact = np.sin(np.pi * X) * np.sin(np.pi * Y)

        solver = NonlinearDiffusionSolver(mesh, D=1.0)
        solver.set_source(source.flatten())
        solver.set_boundary(bt.Boundary.Left, 0.0)
        solver.set_boundary(bt.Boundary.Right, 0.0)
        solver.set_boundary(bt.Boundary.Bottom, 0.0)
        solver.set_boundary(bt.Boundary.Top, 0.0)

        result = solver.solve()

        assert result.converged
        # Check interior points
        error = np.max(np.abs(result.solution[1:-1, 1:-1] - exact[1:-1, 1:-1]))
        assert error < 0.1  # 2nd-order error on coarse mesh

    def test_verbose_output(self, capsys):
        """Test that verbose mode prints iteration info."""
        mesh = bt.mesh_1d(20, 0, 1)

        solver = NonlinearDiffusionSolver(mesh, D=1.0)
        solver.set_boundary(bt.Boundary.Left, 1.0)
        solver.set_boundary(bt.Boundary.Right, 0.0)
        solver.set_parameters(verbose=True)

        solver.solve()

        captured = capsys.readouterr()
        assert "Newton iteration" in captured.out

    def test_set_parameters_chaining(self):
        """Test that set_* methods return self for chaining."""
        mesh = bt.mesh_1d(20, 0, 1)

        solver = (
            NonlinearDiffusionSolver(mesh, D=1.0)
            .set_reaction(lambda u: u**2)
            .set_boundary(bt.Boundary.Left, 1.0)
            .set_boundary(bt.Boundary.Right, 0.0)
            .set_parameters(max_iterations=100, tol=1e-12)
        )

        assert solver.max_iterations == 100
        assert solver.tol == 1e-12


class TestReactionTerms:
    """Tests for built-in reaction term functions."""

    def test_michaelis_menten_values(self):
        """Test Michaelis-Menten reaction values."""
        vmax, km = 2.0, 0.5
        reaction, deriv = michaelis_menten(vmax, km)

        u = np.array([0.0, 0.5, 1.0, 10.0])
        R = reaction(u)

        # R(0) = 0
        assert R[0] == pytest.approx(0.0)
        # R(Km) = Vmax/2
        assert R[1] == pytest.approx(vmax / 2)
        # R(large) ≈ Vmax
        assert R[3] == pytest.approx(vmax, rel=0.1)

    def test_michaelis_menten_derivative(self):
        """Test Michaelis-Menten derivative values."""
        vmax, km = 2.0, 0.5
        reaction, deriv = michaelis_menten(vmax, km)

        u = np.array([0.0, 0.5, 1.0])
        dR = deriv(u)

        # dR/du at u=0: Vmax/Km
        assert dR[0] == pytest.approx(vmax / km)
        # Should be positive and decreasing
        assert all(dR > 0)
        assert dR[0] > dR[1] > dR[2]

    def test_hill_kinetics_values(self):
        """Test Hill kinetics behavior."""
        vmax, km, n = 1.0, 1.0, 2
        reaction, _ = hill_kinetics(vmax, km, n)

        u = np.array([0.0, 1.0, 10.0])
        R = reaction(u)

        # R(0) = 0
        assert R[0] == pytest.approx(0.0)
        # R(Km) = Vmax/2
        assert R[1] == pytest.approx(vmax / 2)
        # R(large) ≈ Vmax
        assert R[2] == pytest.approx(vmax, rel=0.01)

    def test_bistable_fixed_points(self):
        """Test bistable reaction fixed points."""
        a = 0.3
        reaction, deriv = bistable(a)

        # Fixed points at u=0, u=a, u=1
        u = np.array([0.0, a, 1.0])
        R = reaction(u)

        assert R[0] == pytest.approx(0.0, abs=1e-10)
        assert R[1] == pytest.approx(0.0, abs=1e-10)
        assert R[2] == pytest.approx(0.0, abs=1e-10)

    def test_exponential_decay_linear(self):
        """Test exponential decay is linear."""
        k = 0.5
        reaction, deriv = exponential_decay(k)

        u = np.array([0.0, 1.0, 2.0, 5.0])
        R = reaction(u)
        dR = deriv(u)

        np.testing.assert_allclose(R, k * u)
        np.testing.assert_allclose(dR, np.full_like(u, k))


class TestConvergenceAndAccuracy:
    """Tests for numerical convergence and accuracy."""

    def test_quadratic_convergence(self):
        """Test Newton's method achieves quadratic convergence rate."""

        def residual(u):
            return u**2 - 4

        def jacobian(u):
            return np.array([[2 * u[0]]])

        solver = NewtonRaphsonSolver(residual, jacobian, n=1)
        solver.set_parameters(use_line_search=False)
        result = solver.solve(np.array([3.0]))

        # Check that residual decreases quadratically
        # For quadratic convergence: e_{n+1} / e_n² ≈ constant
        errors = [abs(r) for r in result.residual_history]
        if len(errors) >= 3:
            ratios = []
            for i in range(1, len(errors) - 1):
                if (
                    errors[i] > 1e-14
                    and errors[i - 1] > 1e-14
                    and errors[i + 1] > 1e-14
                ):
                    # Quadratic: e_{n+1} ≈ C * e_n²
                    ratio = errors[i + 1] / (errors[i] ** 2)
                    ratios.append(ratio)
            # Ratios should be roughly constant (filter out zeros)
            ratios = [r for r in ratios if r > 1e-10]
            if len(ratios) >= 2:
                assert max(ratios) / min(ratios) < 100  # Within reasonable range

    def test_grid_convergence_poisson(self):
        """Test solution converges with grid refinement."""
        D = 1.0
        errors = []
        grids = [10, 20, 40]

        for n in grids:
            mesh = bt.mesh_1d(n, 0, 1)
            x = bt.x_nodes(mesh)

            source = np.sin(np.pi * x)
            exact = np.sin(np.pi * x) / (np.pi**2 * D)

            solver = NonlinearDiffusionSolver(mesh, D=D)
            solver.set_source(source)
            solver.set_boundary(bt.Boundary.Left, 0.0)
            solver.set_boundary(bt.Boundary.Right, 0.0)

            result = solver.solve(np.zeros(len(x)))
            error = np.max(np.abs(result.solution[1:-1] - exact[1:-1]))
            errors.append(error)

        # Should converge at O(h²)
        rates = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
        for rate in rates:
            assert rate > 3.5  # Close to 4 for 2nd order
