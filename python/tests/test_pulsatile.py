"""Tests for pulsatile boundary conditions."""

import numpy as np
import pytest

import biotransport as bt


class TestBasicWaveforms:
    """Test basic waveform BC types."""

    def test_constant_bc(self):
        """ConstantBC returns same value for all times."""
        bc = bt.reference.ConstantBC(value=42.0)
        assert bc(0.0) == 42.0
        assert bc(1.0) == 42.0
        assert bc(100.0) == 42.0
        assert bc.period() == 0.0  # Non-periodic

    def test_sinusoidal_bc_at_key_points(self):
        """SinusoidalBC produces correct values at key phase points."""
        bc = bt.reference.SinusoidalBC(
            mean=10.0, amplitude=5.0, frequency=1.0, phase=0.0
        )

        # At t=0, sin(0) = 0, so value = mean
        assert bc(0.0) == pytest.approx(10.0, abs=1e-10)

        # At t=0.25, sin(pi/2) = 1, so value = mean + amplitude
        assert bc(0.25) == pytest.approx(15.0, abs=1e-10)

        # At t=0.5, sin(pi) = 0, so value = mean
        assert bc(0.5) == pytest.approx(10.0, abs=1e-10)

        # At t=0.75, sin(3pi/2) = -1, so value = mean - amplitude
        assert bc(0.75) == pytest.approx(5.0, abs=1e-10)

        # Period = 1/frequency
        assert bc.period() == pytest.approx(1.0)

    def test_sinusoidal_bc_with_phase(self):
        """SinusoidalBC with phase offset."""
        # Phase of pi/2 shifts sine to cosine
        bc = bt.reference.SinusoidalBC(
            mean=0.0, amplitude=1.0, frequency=1.0, phase=np.pi / 2
        )
        # At t=0, value = sin(pi/2) = 1
        assert bc(0.0) == pytest.approx(1.0, abs=1e-10)

    def test_ramp_bc(self):
        """RampBC produces linear interpolation."""
        bc = bt.reference.RampBC(
            start_value=0.0, end_value=100.0, t_start=1.0, duration=2.0
        )

        # Before ramp
        assert bc(0.5) == 0.0

        # Start of ramp
        assert bc(1.0) == 0.0

        # Middle of ramp
        assert bc(2.0) == pytest.approx(50.0)

        # End of ramp
        assert bc(3.0) == 100.0

        # After ramp
        assert bc(10.0) == 100.0

    def test_step_bc(self):
        """StepBC transitions at step time."""
        bc = bt.reference.StepBC(value_before=0.0, value_after=1.0, t_step=5.0)

        assert bc(4.9) == 0.0
        assert bc(5.0) == 1.0
        assert bc(5.1) == 1.0

    def test_square_wave_bc(self):
        """SquareWaveBC alternates between high and low."""
        bc = bt.reference.SquareWaveBC(
            high_value=10.0, low_value=0.0, frequency=2.0, duty_cycle=0.5
        )

        # Period = 0.5s, duty_cycle = 0.5, so high for 0.25s then low
        assert bc(0.0) == 10.0  # Start of high phase
        assert bc(0.1) == 10.0  # Still in high phase
        assert bc(0.3) == 0.0  # In low phase
        assert bc(0.5) == 10.0  # Next cycle

        assert bc.period() == pytest.approx(0.5)

    @pytest.mark.parametrize(
        "value",
        [
            True,
            1.0 + 2.0j,
            np.complex128(3.0 + 4.0j),
            "1.25",
            b"1.25",
            np.str_("1.25"),
            np.bytes_("1.25"),
        ],
    )
    @pytest.mark.parametrize(
        "factory",
        [
            lambda value: bt.reference.ConstantBC(value=value),
            lambda value: bt.reference.SinusoidalBC(frequency=value),
            lambda value: bt.reference.CustomBC(func=lambda _t: 0.0, T=value),
        ],
    )
    def test_waveform_constructors_reject_nonreal_scalars(self, factory, value):
        with pytest.raises(TypeError, match="real scalar"):
            factory(value)


class TestCardiacWaveforms:
    """Test physiological cardiac waveforms."""

    def test_arterial_pressure_range(self):
        """ArterialPressureBC stays within physiological range."""
        bc = bt.reference.ArterialPressureBC(systolic=120, diastolic=80, heart_rate=72)

        # Sample over one cardiac cycle
        period = bc.period()
        times = np.linspace(0, period, 100)
        values = [bc(t) for t in times]

        # Should be bounded near systolic/diastolic (with some tolerance for harmonics)
        assert min(values) > 65  # Close to diastolic (Fourier approx can undershoot)
        assert max(values) < 135  # Close to systolic

        # Period should be 60/72 = 0.833s
        assert bc.period() == pytest.approx(60.0 / 72, rel=1e-6)

    def test_arterial_pressure_periodicity(self):
        """ArterialPressureBC is periodic."""
        bc = bt.reference.ArterialPressureBC(heart_rate=60)  # 1 Hz for easy testing
        period = bc.period()

        # Values at t and t + period should be the same
        for t in [0.0, 0.2, 0.5, 0.8]:
            assert bc(t) == pytest.approx(bc(t + period), rel=1e-10)
            assert bc(t) == pytest.approx(bc(t + 2 * period), rel=1e-10)

    def test_venous_pressure_range(self):
        """VenousPressureBC stays in venous range."""
        bc = bt.reference.VenousPressureBC(
            mean_pressure=8.0, amplitude=4.0, heart_rate=72
        )

        times = np.linspace(0, bc.period(), 100)
        values = [bc(t) for t in times]

        # Venous pressure is much lower than arterial
        assert min(values) >= 0
        assert max(values) < 20

    def test_cardiac_output_shape(self):
        """CardiacOutputBC has high systolic peak and low diastolic."""
        bc = bt.reference.CardiacOutputBC(mean_flow=5.0, peak_flow=25.0, heart_rate=72)

        period = bc.period()
        times = np.linspace(0, period, 100)
        values = [bc(t) for t in times]

        # Should have a clear peak during ejection
        assert max(values) > 20
        # Diastolic flow should be low
        assert min(values) >= 0

    def test_respiratory_bc(self):
        """RespiratoryBC has correct respiratory rate."""
        bc = bt.reference.RespiratoryBC(mean=0.0, amplitude=1.0, respiratory_rate=12)

        # Period = 60/12 = 5 seconds
        assert bc.period() == pytest.approx(5.0)

        # Should oscillate between mean-amplitude and mean+amplitude
        times = np.linspace(0, 5, 100)
        values = [bc(t) for t in times]
        assert min(values) >= -0.1  # Near mean (0)
        assert max(values) <= 1.1  # Near mean + amplitude

    def test_drug_infusion_phases(self):
        """DrugInfusionBC has bolus and maintenance phases."""
        bc = bt.reference.DrugInfusionBC(
            bolus_concentration=1.0,
            maintenance_concentration=0.1,
            bolus_duration=60.0,
            infusion_start=0.0,
        )

        # Before infusion
        bc_pre = bt.reference.DrugInfusionBC(
            bolus_concentration=1.0,
            maintenance_concentration=0.1,
            bolus_duration=60.0,
            infusion_start=10.0,
        )
        assert bc_pre(5.0) == 0.0

        # Bolus phase (high concentration)
        assert bc(0.0) == pytest.approx(1.0)
        assert bc(30.0) > 0.5  # Still in bolus

        # Maintenance phase (low concentration)
        assert bc(120.0) == pytest.approx(0.1)


class TestCompositeWaveforms:
    """Test composite boundary conditions."""

    def test_composite_add(self):
        """CompositeBC adds waveforms correctly."""
        bc1 = bt.reference.ConstantBC(value=10.0)
        bc2 = bt.reference.ConstantBC(value=5.0)
        composite = bt.reference.CompositeBC(components=[bc1, bc2], operation="add")

        assert composite(0.0) == 15.0
        assert composite(1.0) == 15.0

    def test_composite_multiply(self):
        """CompositeBC multiplies waveforms correctly."""
        bc1 = bt.reference.ConstantBC(value=10.0)
        bc2 = bt.reference.ConstantBC(value=0.5)
        composite = bt.reference.CompositeBC(
            components=[bc1, bc2], operation="multiply"
        )

        assert composite(0.0) == 5.0

    def test_composite_with_sinusoidal(self):
        """CompositeBC can modulate sinusoidal with respiratory."""
        # Mean arterial pressure with respiratory modulation
        base = bt.reference.SinusoidalBC(
            mean=100, amplitude=20, frequency=1.2
        )  # ~72 bpm
        resp = bt.reference.SinusoidalBC(
            mean=1.0, amplitude=0.05, frequency=0.2
        )  # ~12 breaths/min

        composite = bt.reference.CompositeBC(
            components=[base, resp], operation="multiply"
        )

        # Should modulate between ~95 and ~105 times the base
        val = composite(0.0)
        assert 80 < val < 120


class TestCustomBC:
    """Test custom user-defined BCs."""

    def test_custom_lambda(self):
        """CustomBC accepts lambda functions."""
        bc = bt.reference.CustomBC(func=lambda t: t**2, T=0.0)
        assert bc(2.0) == 4.0
        assert bc(3.0) == 9.0

    def test_custom_periodic(self):
        """CustomBC can specify period."""
        bc = bt.reference.CustomBC(func=lambda t: np.sin(2 * np.pi * t), T=1.0)
        assert bc.period() == 1.0

    @pytest.mark.parametrize(
        "value",
        [
            True,
            1.0 + 2.0j,
            np.complex128(3.0 + 4.0j),
            "1.25",
            b"1.25",
            np.str_("1.25"),
            np.bytes_("1.25"),
        ],
    )
    def test_custom_callback_rejects_nonreal_scalars(self, value):
        bc = bt.reference.CustomBC(func=lambda _t: value)

        with pytest.raises(TypeError, match="must return a real scalar"):
            bc(0.0)


class TestUtilityFunctions:
    """Test utility functions."""

    def test_heart_rate_to_period(self):
        """heart_rate_to_period converts correctly."""
        assert bt.reference.heart_rate_to_period(60) == pytest.approx(1.0)
        assert bt.reference.heart_rate_to_period(72) == pytest.approx(60 / 72)
        assert bt.reference.heart_rate_to_period(120) == pytest.approx(0.5)

    def test_period_to_heart_rate(self):
        """period_to_heart_rate converts correctly."""
        assert bt.reference.period_to_heart_rate(1.0) == pytest.approx(60)
        assert bt.reference.period_to_heart_rate(0.5) == pytest.approx(120)

    def test_sample_waveform(self):
        """sample_waveform produces correct arrays."""
        bc = bt.reference.SinusoidalBC(mean=0, amplitude=1, frequency=1)
        times, values = bt.reference.sample_waveform(
            bc, t_start=0, t_end=1, num_points=5
        )

        assert len(times) == 5
        assert len(values) == 5
        assert times[0] == 0.0
        assert times[-1] == 1.0


class TestSolvePulsatile:
    """Test the solve_pulsatile solver."""

    def test_solve_pulsatile_basic(self):
        """solve_pulsatile runs without error."""
        mesh = bt.mesh_1d(50, 0, 1)
        problem = (
            bt.Problem(mesh).diffusivity(0.01).initial_condition(bt.uniform(mesh, 0.5))
        )

        bc_left = bt.reference.SinusoidalBC(mean=1.0, amplitude=0.5, frequency=1.0)

        result = bt.reference.solve_pulsatile(
            problem, t_end=0.5, pulsatile_bcs={bt.Boundary.Left: bc_left}, dt=0.0001
        )

        assert result.solution is not None
        assert len(result.solution) == 51  # 50 cells = 51 nodes
        assert result.time == pytest.approx(0.5, rel=1e-3)

    def test_solve_pulsatile_bc_applied(self):
        """solve_pulsatile correctly applies time-varying BC."""
        mesh = bt.mesh_1d(20, 0, 1)
        initial = np.zeros(21)
        problem = (
            bt.Problem(mesh)
            .diffusivity(0.1)
            .initial_condition(initial)
            .dirichlet(bt.Boundary.Right, 0.0)
        )

        # Step BC that turns on at t=0
        bc_left = bt.reference.ConstantBC(value=1.0)

        result = bt.reference.solve_pulsatile(
            problem,
            t_end=0.1,
            pulsatile_bcs={bt.Boundary.Left: bc_left},
            dt=0.0001,
        )

        # Left boundary should be 1.0
        assert result.solution[0] == pytest.approx(1.0)
        # Explicit fixed-zero right boundary should remain 0.
        assert result.solution[-1] == pytest.approx(0.0)
        # Interior should have some diffused concentration
        assert result.solution[5] > 0

    def test_solve_pulsatile_with_history(self):
        """solve_pulsatile saves history when requested."""
        mesh = bt.mesh_1d(20, 0, 1)
        problem = (
            bt.Problem(mesh).diffusivity(0.01).initial_condition(bt.uniform(mesh, 0.0))
        )

        bc = bt.reference.SinusoidalBC(mean=1.0, amplitude=0.5, frequency=2.0)

        result = bt.reference.solve_pulsatile(
            problem,
            t_end=0.1,
            pulsatile_bcs={bt.Boundary.Left: bc},
            dt=0.0001,
            save_every=100,
        )

        # Should have multiple snapshots
        assert len(result.time_history) > 1
        assert len(result.solution_history) > 1
        assert bt.Boundary.Left in result.bc_history

    def test_solve_pulsatile_stats(self):
        """solve_pulsatile returns statistics."""
        mesh = bt.mesh_1d(20, 0, 1)
        problem = (
            bt.Problem(mesh).diffusivity(0.1).initial_condition(bt.uniform(mesh, 0.0))
        )

        result = bt.reference.solve_pulsatile(
            problem,
            t_end=0.01,
            pulsatile_bcs={bt.Boundary.Left: bt.reference.ConstantBC(1.0)},
        )

        assert "steps" in result.stats
        assert "dt" in result.stats
        assert result.stats["steps"] > 0


class TestCardiacCycleIntegration:
    """Integration tests for cardiac cycle simulations."""

    def test_arterial_pressure_simulation(self):
        """Simulate pressure-driven transport with arterial waveform."""
        mesh = bt.mesh_1d(50, 0, 0.1)  # 10cm vessel
        initial = np.ones(51) * 80  # Start at diastolic

        problem = bt.Problem(mesh).diffusivity(1e-5).initial_condition(initial)

        # Arterial pressure at inlet
        arterial = bt.reference.ArterialPressureBC(
            systolic=120, diastolic=80, heart_rate=72
        )

        # Run for one cardiac cycle
        period = arterial.period()
        result = bt.reference.solve_pulsatile(
            problem,
            t_end=period,
            pulsatile_bcs={bt.Boundary.Left: arterial},
            save_every=50,
        )

        # Inlet should track arterial pressure
        # Since diffusion is slow, mostly the inlet changes
        assert result.solution[0] > 70
        assert result.solution[0] < 130

    def test_two_sided_pulsatile(self):
        """Pulsatile BCs on both sides."""
        mesh = bt.mesh_1d(30, 0, 1)
        problem = (
            bt.Problem(mesh).diffusivity(0.1).initial_condition(bt.uniform(mesh, 0.5))
        )

        bc_left = bt.reference.SinusoidalBC(mean=1.0, amplitude=0.2, frequency=1.0)
        bc_right = bt.reference.SinusoidalBC(
            mean=0.0, amplitude=0.2, frequency=1.0, phase=np.pi
        )  # Out of phase

        result = bt.reference.solve_pulsatile(
            problem,
            t_end=0.5,
            pulsatile_bcs={bt.Boundary.Left: bc_left, bt.Boundary.Right: bc_right},
            dt=0.0001,
        )

        # Both boundaries should reflect their BCs at final time
        assert result.solution[0] == pytest.approx(bc_left(0.5), rel=0.01)
        assert result.solution[-1] == pytest.approx(bc_right(0.5), rel=0.01)
