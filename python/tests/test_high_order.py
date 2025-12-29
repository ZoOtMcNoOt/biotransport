"""Tests for higher-order finite difference schemes."""

import numpy as np
import pytest

# Import from the module
from biotransport.high_order import (
    laplacian_2nd_order,
    laplacian_4th_order,
    laplacian_6th_order,
    gradient_4th_order,
    d2dx2,
    ddx,
    HighOrderDiffusionSolver,
    verify_order_of_accuracy,
)
import biotransport as bt


class TestLaplacianStencils:
    """Test finite difference stencil accuracy."""

    def test_laplacian_2nd_order_quadratic(self):
        """2nd-order Laplacian is exact for quadratic functions."""
        # u = x² has d²u/dx² = 2 (constant)
        n = 20
        x = np.linspace(0, 1, n + 1)
        dx = 1.0 / n
        u = x**2

        lap = laplacian_2nd_order(u, dx)

        # Should be exactly 2 in interior
        for i in range(1, n):
            assert lap[i] == pytest.approx(2.0, abs=1e-10)

    def test_laplacian_4th_order_quartic(self):
        """4th-order Laplacian is exact for up to 4th-degree polynomials."""
        # u = x⁴ has d²u/dx² = 12x²
        n = 20
        mesh = bt.mesh_1d(n, 0, 1)
        x = np.linspace(0, 1, n + 1)
        u = x**4

        lap = laplacian_4th_order(mesh, u)
        exact = 12 * x**2

        # Should be exact (within roundoff) for interior points with full stencil
        for i in range(2, n - 1):
            assert lap[i] == pytest.approx(exact[i], rel=1e-8)

    def test_laplacian_6th_order_6th_degree(self):
        """6th-order Laplacian is exact for up to 6th-degree polynomials."""
        # u = x⁶ has d²u/dx² = 30x⁴
        n = 30
        mesh = bt.mesh_1d(n, 0, 1)
        x = np.linspace(0, 1, n + 1)
        u = x**6

        lap = laplacian_6th_order(mesh, u)
        exact = 30 * x**4

        # Should be exact for deep interior points with full stencil
        for i in range(3, n - 2):
            assert lap[i] == pytest.approx(exact[i], rel=1e-6)

    def test_laplacian_4th_order_sine(self):
        """4th-order Laplacian on sine function converges with O(dx⁴)."""
        # u = sin(2πx) has d²u/dx² = -4π² sin(2πx)
        errors = []
        for n in [20, 40, 80]:
            mesh = bt.mesh_1d(n, 0, 1)
            x = np.linspace(0, 1, n + 1)
            u = np.sin(2 * np.pi * x)
            exact = -((2 * np.pi) ** 2) * np.sin(2 * np.pi * x)

            lap = laplacian_4th_order(mesh, u)

            # Compute error in interior (skip boundary cells)
            error = np.max(np.abs(lap[2:-2] - exact[2:-2]))
            errors.append(error)

        # Check 4th-order convergence: error ratio should be ~16 when dx halves
        ratio1 = errors[0] / errors[1]
        ratio2 = errors[1] / errors[2]

        assert ratio1 > 12, f"Expected ~16, got {ratio1:.1f}"
        assert ratio2 > 12, f"Expected ~16, got {ratio2:.1f}"

    def test_laplacian_2nd_order_sine_convergence(self):
        """2nd-order Laplacian converges with O(dx²)."""
        errors = []
        for n in [20, 40, 80]:
            x = np.linspace(0, 1, n + 1)
            dx = 1.0 / n
            u = np.sin(2 * np.pi * x)
            exact = -((2 * np.pi) ** 2) * np.sin(2 * np.pi * x)

            lap = laplacian_2nd_order(u, dx)
            error = np.max(np.abs(lap[1:-1] - exact[1:-1]))
            errors.append(error)

        # Check 2nd-order convergence: error ratio should be ~4 when dx halves
        ratio1 = errors[0] / errors[1]
        ratio2 = errors[1] / errors[2]

        assert 3 < ratio1 < 5, f"Expected ~4, got {ratio1:.1f}"
        assert 3 < ratio2 < 5, f"Expected ~4, got {ratio2:.1f}"


class TestGradientStencils:
    """Test gradient stencil accuracy."""

    def test_gradient_4th_order_cubic(self):
        """4th-order gradient is exact for cubic functions."""
        # u = x³ has du/dx = 3x²
        n = 20
        mesh = bt.mesh_1d(n, 0, 1)
        x = np.linspace(0, 1, n + 1)
        u = x**3

        grad = gradient_4th_order(mesh, u)
        exact = 3 * x**2

        # Should be exact for interior points
        for i in range(2, n - 1):
            assert grad[i] == pytest.approx(exact[i], rel=1e-8)

    def test_gradient_4th_order_sine_convergence(self):
        """4th-order gradient converges with O(dx⁴)."""
        errors = []
        for n in [20, 40, 80]:
            mesh = bt.mesh_1d(n, 0, 1)
            x = np.linspace(0, 1, n + 1)
            u = np.sin(2 * np.pi * x)
            exact = 2 * np.pi * np.cos(2 * np.pi * x)

            grad = gradient_4th_order(mesh, u)
            error = np.max(np.abs(grad[2:-2] - exact[2:-2]))
            errors.append(error)

        # Check 4th-order convergence
        ratio1 = errors[0] / errors[1]
        ratio2 = errors[1] / errors[2]

        assert ratio1 > 12, f"Expected ~16, got {ratio1:.1f}"
        assert ratio2 > 12, f"Expected ~16, got {ratio2:.1f}"


class TestConvenienceFunctions:
    """Test convenience wrapper functions."""

    def test_d2dx2_order_selection(self):
        """d2dx2 selects correct stencil based on order."""
        x = np.linspace(0, 1, 21)
        dx = 0.05
        u = x**2

        lap2 = d2dx2(u, dx, order=2)
        lap4 = d2dx2(u, dx, order=4)
        lap6 = d2dx2(u, dx, order=6)

        # All should give ~2 for quadratic
        assert lap2[5] == pytest.approx(2.0, abs=1e-8)
        assert lap4[5] == pytest.approx(2.0, abs=1e-8)
        assert lap6[5] == pytest.approx(2.0, abs=1e-8)

    def test_d2dx2_invalid_order(self):
        """d2dx2 raises error for invalid order."""
        u = np.zeros(10)
        with pytest.raises(ValueError, match="Unsupported order"):
            d2dx2(u, 0.1, order=3)

    def test_ddx_order_selection(self):
        """ddx selects correct stencil based on order."""
        x = np.linspace(0, 1, 21)
        dx = 0.05
        u = x**2

        grad2 = ddx(u, dx, order=2)
        grad4 = ddx(u, dx, order=4)

        # Both should give 2x for quadratic
        assert grad2[10] == pytest.approx(2 * x[10], rel=1e-6)
        assert grad4[10] == pytest.approx(2 * x[10], rel=1e-8)


class TestHighOrderDiffusionSolver:
    """Test the high-order diffusion solver."""

    def test_solver_initialization(self):
        """Solver initializes with valid parameters."""
        mesh = bt.mesh_1d(50, 0, 1)
        solver = HighOrderDiffusionSolver(mesh, D=0.01, order=4)

        assert solver.order == 4
        assert solver.D == 0.01
        assert solver.nx == 50

    def test_solver_invalid_order(self):
        """Solver rejects invalid order."""
        mesh = bt.mesh_1d(50, 0, 1)
        with pytest.raises(ValueError, match="Order must be"):
            HighOrderDiffusionSolver(mesh, D=0.01, order=3)

    def test_solver_invalid_diffusivity(self):
        """Solver rejects non-positive diffusivity."""
        mesh = bt.mesh_1d(50, 0, 1)
        with pytest.raises(ValueError, match="positive"):
            HighOrderDiffusionSolver(mesh, D=-0.01, order=4)

    def test_solver_stable_dt(self):
        """Solver computes reasonable stable dt."""
        mesh = bt.mesh_1d(50, 0, 1)
        solver = HighOrderDiffusionSolver(mesh, D=0.01, order=4)
        dt = solver.compute_stable_dt()

        # Should be positive and reasonable
        assert dt > 0
        assert dt < 1.0  # Should be much smaller than domain size / D

    def test_solver_1d_diffusion(self):
        """Solver produces reasonable diffusion solution."""
        mesh = bt.mesh_1d(50, 0, 1)
        solver = HighOrderDiffusionSolver(mesh, D=0.1, order=4)
        solver.set_boundary(bt.Boundary.Left, 1.0)
        solver.set_boundary(bt.Boundary.Right, 0.0)

        # Gaussian initial condition
        x = bt.x_nodes(mesh)
        initial = np.exp(-((x - 0.5) ** 2) / 0.01)

        result = solver.solve(initial, t_end=0.1)

        assert result.solution is not None
        assert len(result.solution) == 51
        assert result.time == pytest.approx(0.1, rel=1e-3)
        assert result.order == 4
        assert result.steps > 0

    def test_solver_boundary_conditions(self):
        """Solver applies boundary conditions correctly."""
        mesh = bt.mesh_1d(20, 0, 1)
        solver = HighOrderDiffusionSolver(mesh, D=0.1, order=4)
        solver.set_boundary(bt.Boundary.Left, 1.0)
        solver.set_boundary(bt.Boundary.Right, 2.0)

        initial = np.zeros(21)
        result = solver.solve(initial, t_end=0.01)

        assert result.solution[0] == pytest.approx(1.0)
        assert result.solution[-1] == pytest.approx(2.0)

    def test_solver_order_2_vs_4_accuracy(self):
        """4th-order spatial discretization is more accurate than 2nd-order."""
        # Test spatial accuracy by comparing Laplacian computation directly
        # rather than time-integrated solution (which involves time error too)
        mesh = bt.mesh_1d(40, 0, 1)
        x = bt.x_nodes(mesh)
        dx = mesh.dx()

        # Test function: sin(2πx) with known Laplacian = -4π²sin(2πx)
        u = np.sin(2 * np.pi * x)
        exact_laplacian = -((2 * np.pi) ** 2) * np.sin(2 * np.pi * x)

        # Compute with 2nd and 4th order
        lap2 = laplacian_2nd_order(u, dx)
        lap4 = laplacian_4th_order(mesh, u)

        # Compute errors in interior (skip boundary cells)
        error2 = np.max(np.abs(lap2[2:-2] - exact_laplacian[2:-2]))
        error4 = np.max(np.abs(lap4[2:-2] - exact_laplacian[2:-2]))

        # 4th-order should have smaller error (typically 10-100x smaller)
        assert (
            error4 < error2
        ), f"4th-order error {error4:.2e} >= 2nd-order error {error2:.2e}"
        # Expect significant improvement
        assert error4 < 0.1 * error2, "4th-order should be at least 10x more accurate"


class TestVerifyOrderOfAccuracy:
    """Test the order verification utility."""

    def test_verify_2nd_order(self):
        """Verification correctly identifies 2nd-order scheme."""

        def factory(n):
            dx = 1.0 / n
            return lambda u: laplacian_2nd_order(u, dx)

        def exact_solution(x):
            return np.sin(2 * np.pi * x)

        results = verify_order_of_accuracy(
            factory, exact_solution, grid_sizes=(20, 40, 80)
        )

        # Should see ~2nd order convergence
        assert len(results["observed_orders"]) >= 1
        # Allow some tolerance since we're computing numerical derivatives
        assert results["observed_orders"][-1] > 1.5

    def test_verify_4th_order(self):
        """Verification correctly identifies 4th-order scheme."""

        def factory(n):
            mesh = bt.mesh_1d(n, 0, 1)
            return lambda u: laplacian_4th_order(mesh, u)

        def exact_solution(x):
            return np.sin(2 * np.pi * x)

        results = verify_order_of_accuracy(
            factory, exact_solution, grid_sizes=(20, 40, 80)
        )

        # Should see ~4th order convergence
        assert len(results["observed_orders"]) >= 1
        assert results["observed_orders"][-1] > 3.0


class TestLaplacian2D:
    """Test 2D Laplacian stencils."""

    def test_laplacian_2nd_order_2d_quadratic(self):
        """2nd-order 2D Laplacian is exact for quadratic functions."""
        # u = x² + y² has ∇²u = 4
        nx, ny = 10, 10
        dx, dy = 0.1, 0.1
        x = np.linspace(0, 1, nx + 1)
        y = np.linspace(0, 1, ny + 1)
        X, Y = np.meshgrid(x, y)
        u = X**2 + Y**2

        lap = laplacian_2nd_order(u, dx, dy)

        # Should be 4 in interior
        assert lap[5, 5] == pytest.approx(4.0, abs=1e-8)

    def test_laplacian_4th_order_2d_quartic(self):
        """4th-order 2D Laplacian handles quartic functions."""
        # u = x⁴ + y⁴ has ∇²u = 12x² + 12y²
        nx, ny = 20, 20
        mesh = bt.mesh_2d(nx, ny, 0, 1, 0, 1)
        x = np.linspace(0, 1, nx + 1)
        y = np.linspace(0, 1, ny + 1)
        X, Y = np.meshgrid(x, y)
        u = X**4 + Y**4
        exact = 12 * X**2 + 12 * Y**2

        lap = laplacian_4th_order(mesh, u)
        # Reshape result to 2D for comparison
        lap = lap.reshape((ny + 1, nx + 1))

        # Check interior points with full stencil
        assert lap[10, 10] == pytest.approx(exact[10, 10], rel=1e-6)
