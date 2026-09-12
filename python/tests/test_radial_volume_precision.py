"""Compare thin-shell measures with independently evaluated exact integrals."""

from decimal import Decimal, localcontext

import numpy as np
import pytest

import biotransport as bt


@pytest.mark.parametrize("geometry,power", [("cylindrical", 2), ("spherical", 3)])
@pytest.mark.parametrize("inner", [1e-80, 1.0, 1e60])
def test_thin_shell_volume_retains_float64_precision(geometry, power, inner):
    mesh = bt.mesh_1d(100, inner, inner * (1 + 1e-6), geometry)
    for index in (0, 1, 50, 99, 100):
        lower = mesh.x(0) if index == 0 else mesh.x(0) + (index - 0.5) * mesh.dx()
        upper = mesh.x(100) if index == 100 else mesh.x(0) + (index + 0.5) * mesh.dx()
        with localcontext() as context:
            context.prec = 90
            a, b = Decimal.from_float(lower), Decimal.from_float(upper)
            exact = float((b**power - a**power) / power)
        assert mesh.control_volume(index) == pytest.approx(exact, rel=8*np.finfo(float).eps, abs=0)
