"""Tests for convergence module."""

import numpy as np
import pytest

from biotransport.convergence import (
    ConvergenceResult,
    GridConvergenceStudy,
    compute_order_of_accuracy,
    run_convergence_study,
    temporal_convergence_study,
)


class TestConvergenceResult:
    """Tests for ConvergenceResult dataclass."""

    def test_result_basic_fields(self):
        """Test ConvergenceResult stores basic fields."""
        result = ConvergenceResult(
            observed_order=2.1,
            theoretical_order=2.0,
            richardson_estimate=1.0,
            gci_fine=0.01,
            gci_coarse=0.02,
            asymptotic_ratio=1.0,
            mesh_sizes=np.array([0.1, 0.05, 0.025]),
        )

        assert result.observed_order == 2.1
        assert result.theoretical_order == 2.0
        assert result.richardson_estimate == 1.0
        assert result.gci_fine == 0.01
        assert result.gci_coarse == 0.02
        assert result.asymptotic_ratio == 1.0
        assert len(result.mesh_sizes) == 3

    def test_result_optional_fields(self):
        """Test ConvergenceResult optional fields default to None."""
        result = ConvergenceResult(
            observed_order=2.0,
            theoretical_order=2.0,
            richardson_estimate=1.0,
            gci_fine=0.01,
            gci_coarse=0.02,
            asymptotic_ratio=1.0,
            mesh_sizes=np.array([0.1, 0.05]),
        )

        assert result.errors is None
        assert result.solutions is None
        assert result.is_asymptotic is False

    def test_result_with_all_fields(self):
        """Test ConvergenceResult with all fields populated."""
        result = ConvergenceResult(
            observed_order=2.0,
            theoretical_order=2.0,
            richardson_estimate=1.0,
            gci_fine=0.01,
            gci_coarse=0.02,
            asymptotic_ratio=1.0,
            mesh_sizes=np.array([0.1, 0.05, 0.025]),
            errors=np.array([0.01, 0.0025, 0.000625]),
            solutions=np.array([1.01, 1.0025, 1.000625]),
            is_asymptotic=True,
        )

        assert len(result.errors) == 3
        assert len(result.solutions) == 3
        assert result.is_asymptotic is True


class TestGridConvergenceStudy:
    """Tests for GridConvergenceStudy class."""

    def test_init_default_values(self):
        """Test GridConvergenceStudy default initialization."""
        study = GridConvergenceStudy()

        assert study.theoretical_order == 2.0
        assert study.safety_factor == 1.25

    def test_init_custom_values(self):
        """Test GridConvergenceStudy custom initialization."""
        study = GridConvergenceStudy(theoretical_order=4.0, safety_factor=3.0)

        assert study.theoretical_order == 4.0
        assert study.safety_factor == 3.0

    def test_add_solution_basic(self):
        """Test adding solutions to study."""
        study = GridConvergenceStudy()
        study.add_solution(h=0.1, value=1.0)
        study.add_solution(h=0.05, value=1.1)
        study.add_solution(h=0.025, value=1.15)

        assert len(study._mesh_sizes) == 3
        assert len(study._values) == 3

    def test_add_solution_with_error(self):
        """Test adding solutions with optional error."""
        study = GridConvergenceStudy()
        study.add_solution(h=0.1, value=1.0, error=0.1)
        study.add_solution(h=0.05, value=1.1, error=0.05)

        assert len(study._errors) == 2
        assert study._errors[0] == 0.1

    def test_add_solution_chaining(self):
        """Test method chaining for add_solution."""
        study = GridConvergenceStudy()
        result = (
            study.add_solution(h=0.1, value=1.0)
            .add_solution(h=0.05, value=1.1)
            .add_solution(h=0.025, value=1.15)
        )

        assert result is study
        assert len(study._mesh_sizes) == 3

    def test_clear(self):
        """Test clearing stored solutions."""
        study = GridConvergenceStudy()
        study.add_solution(h=0.1, value=1.0, error=0.1)
        study.add_solution(h=0.05, value=1.1, error=0.05)
        study.clear()

        assert len(study._mesh_sizes) == 0
        assert len(study._values) == 0
        assert len(study._errors) == 0

    def test_clear_chaining(self):
        """Test method chaining for clear."""
        study = GridConvergenceStudy()
        study.add_solution(h=0.1, value=1.0)
        result = study.clear()

        assert result is study

    def test_analyze_requires_3_solutions(self):
        """Test analyze raises with fewer than 3 solutions."""
        study = GridConvergenceStudy()
        study.add_solution(h=0.1, value=1.0)
        study.add_solution(h=0.05, value=1.1)

        with pytest.raises(ValueError, match="Need at least 3 grid levels"):
            study.analyze()

    def test_analyze_second_order(self):
        """Test analyze with second-order convergent data."""
        # Simulate second-order convergence: f(h) = f_exact + C*h^2
        f_exact = 1.0
        C = 1.0
        h_values = [0.1, 0.05, 0.025]
        f_values = [f_exact + C * h**2 for h in h_values]

        study = GridConvergenceStudy(theoretical_order=2.0)
        for h, f in zip(h_values, f_values):
            study.add_solution(h=h, value=f)

        result = study.analyze()

        # Should detect order ~2
        assert result.observed_order == pytest.approx(2.0, rel=0.1)
        # Richardson estimate should be close to exact
        assert result.richardson_estimate == pytest.approx(f_exact, rel=0.01)

    def test_analyze_first_order(self):
        """Test analyze with first-order convergent data."""
        # Simulate first-order convergence: f(h) = f_exact + C*h
        f_exact = 2.0
        C = 0.5
        h_values = [0.2, 0.1, 0.05]
        f_values = [f_exact + C * h for h in h_values]

        study = GridConvergenceStudy(theoretical_order=1.0)
        for h, f in zip(h_values, f_values):
            study.add_solution(h=h, value=f)

        result = study.analyze()

        assert result.observed_order == pytest.approx(1.0, rel=0.1)
        assert result.richardson_estimate == pytest.approx(f_exact, rel=0.01)

    def test_analyze_fourth_order(self):
        """Test analyze with fourth-order convergent data."""
        # Simulate fourth-order: f(h) = f_exact + C*h^4
        f_exact = 3.14159
        C = 10.0
        h_values = [0.2, 0.1, 0.05]
        f_values = [f_exact + C * h**4 for h in h_values]

        study = GridConvergenceStudy(theoretical_order=4.0)
        for h, f in zip(h_values, f_values):
            study.add_solution(h=h, value=f)

        result = study.analyze()

        assert result.observed_order == pytest.approx(4.0, rel=0.1)

    def test_analyze_gci_computed(self):
        """Test that GCI values are computed."""
        f_exact = 1.0
        h_values = [0.1, 0.05, 0.025]
        f_values = [f_exact + h**2 for h in h_values]

        study = GridConvergenceStudy()
        for h, f in zip(h_values, f_values):
            study.add_solution(h=h, value=f)

        result = study.analyze()

        assert result.gci_fine >= 0
        assert result.gci_coarse >= 0
        # Fine grid GCI should be smaller
        assert result.gci_fine < result.gci_coarse

    def test_analyze_asymptotic_ratio(self):
        """Test asymptotic ratio detection."""
        f_exact = 1.0
        h_values = [0.1, 0.05, 0.025]
        f_values = [f_exact + h**2 for h in h_values]

        study = GridConvergenceStudy()
        for h, f in zip(h_values, f_values):
            study.add_solution(h=h, value=f)

        result = study.analyze()

        # For well-behaved convergence, ratio should be ~1
        assert result.asymptotic_ratio == pytest.approx(1.0, rel=0.1)
        assert result.is_asymptotic  # Use truthy check instead of `is True`

    def test_analyze_sorts_by_mesh_size(self):
        """Test that analyze handles unsorted input."""
        f_exact = 1.0
        study = GridConvergenceStudy()
        # Add in random order
        study.add_solution(h=0.05, value=f_exact + 0.05**2)
        study.add_solution(h=0.1, value=f_exact + 0.1**2)
        study.add_solution(h=0.025, value=f_exact + 0.025**2)

        result = study.analyze()

        # Should still work
        assert result.observed_order == pytest.approx(2.0, rel=0.1)
        # mesh_sizes should be sorted coarse to fine
        assert result.mesh_sizes[0] > result.mesh_sizes[-1]

    def test_analyze_with_errors(self):
        """Test analyze includes errors when provided."""
        study = GridConvergenceStudy()
        study.add_solution(h=0.1, value=1.01, error=0.01)
        study.add_solution(h=0.05, value=1.0025, error=0.0025)
        study.add_solution(h=0.025, value=1.000625, error=0.000625)

        result = study.analyze()

        assert result.errors is not None
        assert len(result.errors) == 3


class TestComputeOrderOfAccuracy:
    """Tests for compute_order_of_accuracy function."""

    def test_second_order(self):
        """Test order computation for second-order data."""
        h = np.array([0.1, 0.05, 0.025, 0.0125])
        errors = 0.1 * h**2  # Second-order errors

        order, C, r_squared = compute_order_of_accuracy(h, errors)

        assert order == pytest.approx(2.0, rel=0.01)
        assert C == pytest.approx(0.1, rel=0.01)
        assert r_squared > 0.99  # Perfect fit

    def test_first_order(self):
        """Test order computation for first-order data."""
        h = np.array([0.2, 0.1, 0.05, 0.025])
        errors = 0.5 * h**1  # First-order errors

        order, C, r_squared = compute_order_of_accuracy(h, errors)

        assert order == pytest.approx(1.0, rel=0.01)
        assert r_squared > 0.99

    def test_fourth_order(self):
        """Test order computation for fourth-order data."""
        h = np.array([0.2, 0.1, 0.05, 0.025])
        errors = 2.0 * h**4  # Fourth-order errors

        order, C, r_squared = compute_order_of_accuracy(h, errors)

        assert order == pytest.approx(4.0, rel=0.01)

    def test_noisy_data(self):
        """Test order computation with noisy data."""
        np.random.seed(42)
        h = np.array([0.2, 0.1, 0.05, 0.025])
        # Add some noise to second-order data
        errors = 0.1 * h**2 * (1 + 0.1 * np.random.randn(4))

        order, C, r_squared = compute_order_of_accuracy(h, errors)

        # Should still be approximately 2
        assert order == pytest.approx(2.0, rel=0.2)
        # R-squared should be lower due to noise
        assert r_squared > 0.9


class TestRunConvergenceStudy:
    """Tests for run_convergence_study convenience function."""

    def test_basic_study(self):
        """Test basic convergence study run."""

        # Simple solver: f(N) = 1 + 1/N^2, error = 1/N^2
        def solve(n):
            error = 1.0 / n**2
            value = 1.0 + error
            return value, error

        result = run_convergence_study(
            solve,
            n_values=[10, 20, 40],
            theoretical_order=2.0,
            verbose=False,
        )

        assert isinstance(result, ConvergenceResult)
        assert result.observed_order == pytest.approx(2.0, rel=0.1)

    def test_study_with_more_grids(self):
        """Test convergence study with more grid levels."""

        def solve(n):
            error = 1.0 / n**2
            return 1.0 + error, error

        result = run_convergence_study(
            solve,
            n_values=[8, 16, 32, 64, 128],
            theoretical_order=2.0,
            verbose=False,
        )

        assert len(result.mesh_sizes) >= 3
        assert result.observed_order == pytest.approx(2.0, rel=0.1)

    def test_study_first_order(self):
        """Test convergence study for first-order method."""

        def solve(n):
            error = 1.0 / n
            return 2.0 + error, error

        result = run_convergence_study(
            solve,
            n_values=[10, 20, 40],
            theoretical_order=1.0,
            verbose=False,
        )

        assert result.observed_order == pytest.approx(1.0, rel=0.1)

    def test_study_without_error(self):
        """Test study when solver returns only value."""

        def solve(n):
            return 1.0 + 1.0 / n**2  # Just value, no error tuple

        result = run_convergence_study(
            solve,
            n_values=[10, 20, 40],
            theoretical_order=2.0,
            verbose=False,
        )

        assert result is not None
        assert result.errors is None


class TestTemporalConvergenceStudy:
    """Tests for temporal_convergence_study function."""

    def test_basic_temporal_study(self):
        """Test basic temporal convergence study."""

        # First-order method: error ~ dt
        def solve(dt):
            error = dt
            value = 1.0 + error
            return value, error

        result = temporal_convergence_study(
            solve,
            dt_values=[0.1, 0.05, 0.025],
            theoretical_order=1.0,
            verbose=False,
        )

        assert isinstance(result, ConvergenceResult)
        assert result.observed_order == pytest.approx(1.0, rel=0.1)

    def test_second_order_temporal(self):
        """Test temporal study for second-order method (e.g., Crank-Nicolson)."""

        def solve(dt):
            error = dt**2
            return 2.0 + error, error

        result = temporal_convergence_study(
            solve,
            dt_values=[0.1, 0.05, 0.025],
            theoretical_order=2.0,
            verbose=False,
        )

        assert result.observed_order == pytest.approx(2.0, rel=0.1)

    def test_temporal_without_error(self):
        """Test temporal study when solver returns only value."""

        def solve(dt):
            return 1.0 + dt  # Just value

        result = temporal_convergence_study(
            solve,
            dt_values=[0.1, 0.05, 0.025],
            theoretical_order=1.0,
            verbose=False,
        )

        assert result is not None


class TestConvergenceIntegration:
    """Integration tests for convergence study workflows."""

    def test_full_workflow(self):
        """Test complete convergence study workflow."""
        # Simulate heat equation solution convergence
        # Exact: u(x) = sin(pi*x), error ~ h^2 for central differences

        def solve(n):
            h = 1.0 / n
            # Approximate L2 error for 2nd-order method
            error = 0.1 * h**2
            # Solution value at midpoint
            value = 1.0 - error
            return value, error

        result = run_convergence_study(
            solve,
            n_values=[16, 32, 64],
            theoretical_order=2.0,
            verbose=False,
        )

        # Verify order
        assert abs(result.observed_order - result.theoretical_order) < 0.3

        # GCI provides uncertainty estimate
        assert result.gci_fine > 0
        assert result.gci_fine < 0.1  # Should be small for fine grid

    def test_richardson_extrapolation_accuracy(self):
        """Test Richardson extrapolation gives accurate estimate."""
        exact = np.pi  # True value

        def solve(n):
            h = 1.0 / n
            value = exact + 2.0 * h**2  # O(h^2) error
            error = abs(value - exact)
            return value, error

        result = run_convergence_study(
            solve,
            n_values=[10, 20, 40],
            theoretical_order=2.0,
            verbose=False,
        )

        # Richardson estimate should be close to exact
        assert result.richardson_estimate == pytest.approx(exact, rel=0.01)

    def test_detects_wrong_order(self):
        """Test that wrong theoretical order is detected."""

        def solve(n):
            h = 1.0 / n
            error = h  # First-order error
            return 1.0 + error, error

        result = run_convergence_study(
            solve,
            n_values=[10, 20, 40],
            theoretical_order=2.0,  # Expecting 2nd order
            verbose=False,
        )

        # Should detect actual order is ~1, not 2
        assert result.observed_order == pytest.approx(1.0, rel=0.2)
        assert abs(result.observed_order - result.theoretical_order) > 0.5
