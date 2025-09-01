import numpy as np
# import jax.numpy as jnp
from ..jax_import import jnp
import pickle

from .jax_bicubic_HEOS_interpolation_1 import compute_bicubic_coefficients_of_ij, bicubic_interpolant


def bicubic_interpolant_property(h, P, table):
    """
    Interpolates all properties at a given enthalpy (h) and pressure (P) using bicubic interpolation.

    Args:
        h (float): Enthalpy in J/kg
        P (float): Pressure in Pa
        table_path (str): Path to saved property table (dict-based .pkl)

    Returns:
        dict: Dictionary of interpolated property values at (h, P)
    """
    h_vals = jnp.array(table['h'])
    P_vals = jnp.array(table['P'])

    Nh, Np = len(h_vals), len(P_vals)
    hmin, hmax = float(h_vals[0]), float(h_vals[-1])
    Lmin, Lmax = float(jnp.log(P_vals[0])), float(jnp.log(P_vals[-1]))

    logP = jnp.log(P)

    # Identify cell (i, j)
    i = int((h - hmin) / (hmax - hmin) * (Nh - 1))
    j = int((logP - Lmin) / (Lmax - Lmin) * (Np - 1))

    i = np.clip(i, 0, Nh - 2)
    j = np.clip(j, 0, Np - 2)

    deltah = float(h_vals[1] - h_vals[0])
    deltaL = float(jnp.log(P_vals[1]) - jnp.log(P_vals[0]))

    interpolated_props = {}

    for prop, prop_data in table.items():
        if prop in ['h', 'P']:
            continue  # skip grid axes

        f_grid = prop_data['value']
        fx_grid = prop_data['d_dh']
        fy_grid = prop_data['d_dP']
        fxy_grid = prop_data['d2_dhdP']
        
        coeffs_local = compute_bicubic_coefficients_of_ij(
            i, j,
            f_grid,
            fx_grid * deltah,
            fy_grid * deltaL,
            fxy_grid * deltah * deltaL
        )

        coeffs = jnp.zeros((Nh, Np, 16), dtype=jnp.float64)
        coeffs = coeffs.at[i, j, :].set(coeffs_local)

        val = bicubic_interpolant(
            h, P, h_vals, jnp.log(P_vals), coeffs,
            Nh, Np, hmin, hmax, Lmin, Lmax
        )

        interpolated_props[prop] = float(val)

    return interpolated_props
