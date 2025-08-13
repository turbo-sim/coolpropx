import os
import pickle
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
import CoolProp.CoolProp as cp
from coolpropx.fluid_properties import Fluid

import coolpropx as cpx


# ---------------------------
# Configuration
# ---------------------------
fluid_name = "CO2"
hmin = 200e3     # J/kg
hmax = 600e3     # J/kg
Pmin = 2e6       # Pa
Pmax = 20e6      # Pa
Nh = 60           # Grid size
Np = 40
save_prefix = "test_table"
SAVE_FIGURES = False  # Set to False to just display plots

fluid = Fluid(fluid_name)

# ---------------------------
# Generate Table
# ---------------------------
table = cpx.bicubic.generate_property_table(hmin, hmax, Pmin, Pmax, fluid_name, Nh, Np)


# ---------------------------
# Construct midpoints grid
# ---------------------------
h_nodes = jnp.linspace(hmin, hmax, Nh)
logP_nodes = jnp.linspace(jnp.log(Pmin), jnp.log(Pmax), Np)
P_nodes = jnp.exp(logP_nodes)

# Compute midpoints
h_vals = 0.5 * (h_nodes[:-1] + h_nodes[1:])
P_vals = 0.5 * (P_nodes[:-1] + P_nodes[1:])

# ---------------------------
# Properties to test
# ---------------------------
cp_keys = {'T': 'T', 'd': 'D', 's': 'S', 'mu': 'V', 'k': 'L'}
properties = list(cp_keys.keys())

# ---------------------------
# Evaluate and plot errors
# ---------------------------



for prop in properties:
    interp_grid = np.zeros((len(h_vals), len(P_vals)))
    true_grid = np.zeros_like(interp_grid)
    rel_error = np.zeros_like(interp_grid)

    print(f"\nTesting: {prop}")

    for i, h in enumerate(h_vals):
        for j, P in enumerate(P_vals):
            try:
                interp_val = cpx.bicubic.bicubic_interpolant_property(float(h), float(P), table)[prop]
                # cp_val = cp.PropsSI(cp_keys[prop], 'H', float(h), 'P', float(P), fluid)
                cp_val = fluid.get_state(cp.HmassP_INPUTS, float(h), float(P))[prop]
                # cp_val = tf.get_props_custom_jvp(fluid, cp.HmassP_INPUTS, float(h), float(P))[prop]
                interp_grid[i, j] = interp_val
                true_grid[i, j] = cp_val
                rel_error[i, j] = np.abs((interp_val - cp_val) / cp_val)
                # print(interp_val, cp_val, rel_error[i, j])
            except Exception:
                interp_grid[i, j] = np.nan
                true_grid[i, j] = np.nan
                rel_error[i, j] = np.nan

    percent_error = rel_error * 100

    print(f"   -> Abs error: min={np.nanmin(np.abs(interp_grid - true_grid)):.3e}, "
          f"max={np.nanmax(np.abs(interp_grid - true_grid)):.3e}")
    print(f"   -> % error: min={np.nanmin(percent_error):.3e}%, "
          f"max={np.nanmax(percent_error):.3e}%, mean={np.nanmean(percent_error):.3e}%")

    # ---------------------------
    # Plot error contour
    # ---------------------------
    H_mesh, P_mesh = np.meshgrid(h_vals, P_vals / 1e5, indexing='ij')  # P in bar

    plt.figure(figsize=(8, 5))
    levels = np.logspace(-12, 2, 15)
    masked_error = np.clip(percent_error, levels[0], levels[-1])

    contour = plt.contourf(
        H_mesh, P_mesh, masked_error,
        levels=levels,
        norm=LogNorm(vmin=levels[0], vmax=levels[-1]),
        cmap='viridis', extend='both'
    )

    cbar = plt.colorbar(contour)
    cbar.set_ticks(levels)
    cbar.set_ticklabels([f"$10^{{{int(np.log10(l))}}}$" for l in levels])
    cbar.set_label("Percentage error [%] (log scale)")

    plt.xlabel("Enthalpy [J/kg]")
    plt.ylabel("Pressure [bar]")
    plt.title(f"Percentage Error (Midpoints): {prop}")

    # ---------------------------
    # Plot saturation lines
    # ---------------------------
    P_sats = np.linspace(Pmin, Pmax, 200)
    h_l, h_v, P_sats_bar = [], [], []

    for P_sat in P_sats:
        try:
            hl = cp.PropsSI('H', 'P', P_sat, 'Q', 0, fluid_name)
            hv = cp.PropsSI('H', 'P', P_sat, 'Q', 1, fluid_name)
            h_l.append(hl)
            h_v.append(hv)
            P_sats_bar.append(P_sat / 1e5)
        except:
            continue

    plt.plot(h_l, P_sats_bar, 'w--', lw=1.5, label='Sat. Liquid')
    plt.plot(h_v, P_sats_bar, 'w--', lw=1.5, label='Sat. Vapor')
    plt.legend()

    plt.tight_layout()

    # ---------------------------
    # Save or show
    # ---------------------------
    if SAVE_FIGURES:
        fig_dir = os.path.join(script_dir, "verification_figures")
        os.makedirs(fig_dir, exist_ok=True)
        fig_path = os.path.join(fig_dir, f"{prop}_midpoint_interp_error_contour.png")
        plt.savefig(fig_path, dpi=300)
        print(f"Saved figure: {fig_path}")
        plt.close()
    else:
        plt.show()
