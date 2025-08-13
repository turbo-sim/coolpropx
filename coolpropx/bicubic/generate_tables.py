import numpy as np
import jax.numpy as jnp
import CoolProp.CoolProp as cp
# import turboflow as tf
import os
import pickle
from coolpropx.fluid_properties import Fluid

def generate_property_table(hmin, hmax, Pmin, Pmax, fluid_name, Nh, Np, outdir='fluid_tables'):
    fluid = Fluid(fluid_name)
    # fluid = tf.Fluid(fluid_name)
    h_vals = jnp.linspace(hmin, hmax, Nh)
    Lmin=jnp.log(Pmin) # Log of P
    Lmax=jnp.log(Pmax) # Log of P
    P_vals = jnp.linspace(Lmin, Lmax, Np)

    deltah = float(h_vals[1] - h_vals[0])
    deltaL = P_vals[1]-P_vals[0]
    eps_h = 0.001 * deltah
    eps_P = 1e-6 * float(Pmin)

    properties = {
        'T': 'T',       # Temperature [K]
        'd': 'D',       # Density [kg/m³]
        's': 'S',       # Entropy [J/kg/K]
        'mu': 'V',      # Viscosity [Pa·s]
        'k': 'L',       # Thermal conductivity [W/m/K]
    }

    # Initialize property grids
    table = {
        'h': np.array(h_vals),
        'P': np.array(jnp.exp(P_vals))
    }

    for key in properties:
        table[key] = {
            'value': np.zeros((Nh, Np), dtype=np.float64),
            'd_dh':  np.zeros((Nh, Np), dtype=np.float64),
            'd_dP':  np.zeros((Nh, Np), dtype=np.float64),
            'd2_dhdP': np.zeros((Nh, Np), dtype=np.float64),
        }

    # Loop over grid and populate values
    for i, h in enumerate(h_vals):
        for j, P in enumerate(P_vals):
            hf = float(h)
            Pf = jnp.exp(P)

            try:        
                f_0 = fluid.get_state(cp.HmassP_INPUTS, float(hf), float(Pf))
                f_h = fluid.get_state(cp.HmassP_INPUTS, float(hf + eps_h), float(Pf))
                f_p = fluid.get_state(cp.HmassP_INPUTS, float(hf), float(Pf + eps_P))
                f_hp = fluid.get_state(cp.HmassP_INPUTS, float(hf + eps_h), float(Pf + eps_P))
            # Uncomment below if using turboflow
                # f_0 = tf.get_props_custom_jvp(fluid, cp.HmassP_INPUTS, hf, Pf)
                # f_h = tf.get_props_custom_jvp(fluid, cp.HmassP_INPUTS, hf + eps_h, Pf)
                # f_p = tf.get_props_custom_jvp(fluid, cp.HmassP_INPUTS, hf, Pf + eps_P)
                # f_hp = tf.get_props_custom_jvp(fluid, cp.HmassP_INPUTS, hf + eps_h, Pf + eps_P)
            except Exception:
                continue  # Skip invalid points

            for key in properties:
                val = f_0[key]
                dval_dh = (f_h[key] - f_0[key]) / eps_h
                dval_dP = (f_p[key] - f_0[key]) / eps_P
                d2val_dhdP = (f_hp[key] - f_h[key] - f_p[key] + f_0[key]) / (eps_h * eps_P)

                table[key]['value'][i, j] = val
                table[key]['d_dh'][i, j] = dval_dh
                table[key]['d_dP'][i, j] = dval_dP
                table[key]['d2_dhdP'][i, j] = d2val_dhdP

    # Save as pickle only (most useful for JAX processing)
    os.makedirs(outdir, exist_ok=True)
    pkl_path = os.path.join(outdir, f'{fluid_name}_{Nh}_x_{Np}.pkl')

    with open(pkl_path, 'wb') as f:
        pickle.dump(table, f)

    print(f" Saved the table to:\n -> Pickle: {pkl_path}")

    return table
