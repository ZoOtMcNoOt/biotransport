"""Typical parameter ranges for biotransport simulations.

This module provides reference values and typical ranges for common
physical parameters used in mass transport and heat transfer simulations.
"""

from typing import Any, Dict


def get_parameter_ranges() -> Dict[str, Dict[str, Any]]:
    """Return typical parameter ranges for reference.

    Returns
    -------
    dict
        Nested dictionary with parameter names as keys and
        dicts containing 'min', 'max', 'typical', 'unit', 'description'.

    Examples
    --------
    >>> ranges = get_parameter_ranges()
    >>> print(ranges['D_drug']['typical'])
    1e-11
    """
    return {
        "D_drug": {
            "min": 1e-12,
            "max": 1e-9,
            "typical": 1e-11,
            "unit": "m²/s",
            "description": "Drug diffusion coefficient in tissue",
        },
        "D_oxygen": {
            "min": 1e-10,
            "max": 5e-9,
            "typical": 2e-9,
            "unit": "m²/s",
            "description": "Oxygen diffusion coefficient in tissue",
        },
        "D_glucose": {
            "min": 1e-10,
            "max": 1e-9,
            "typical": 6e-10,
            "unit": "m²/s",
            "description": "Glucose diffusion coefficient in tissue",
        },
        "k_tissue": {
            "min": 0.2,
            "max": 0.8,
            "typical": 0.5,
            "unit": "W/(m·K)",
            "description": "Thermal conductivity of soft tissue",
        },
        "c_tissue": {
            "min": 3000,
            "max": 4000,
            "typical": 3600,
            "unit": "J/(kg·K)",
            "description": "Specific heat of soft tissue",
        },
        "w_blood": {
            "min": 0.0001,
            "max": 0.01,
            "typical": 0.0005,
            "unit": "1/s",
            "description": "Blood perfusion rate",
        },
        "IFP_tumor": {
            "min": 5,
            "max": 60,
            "typical": 20,
            "unit": "mmHg",
            "description": "Interstitial fluid pressure in tumor",
        },
        "MVD": {
            "min": 10,
            "max": 400,
            "typical": 100,
            "unit": "vessels/mm²",
            "description": "Microvascular density",
        },
    }
