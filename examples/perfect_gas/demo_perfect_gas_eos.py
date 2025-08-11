"""
Demo: perfect gas API usage with air

What this shows:
- how to obtain constants (precomputed vs computed)
- how to evaluate states from different input pairs
- how to evaluate transport properties as vectorized functions of temperature
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import coolpropx as cpx
import coolpropx.perfect_gas as pg

cpx.set_plot_options()


# --------------------------- get constants -------------------------- #

constants = pg.get_perfect_gas_constants("air", 298.15, 101_325.0, display=False)
print(f"\nPerfect-gas constants:")
cpx.print_dict(constants)

# -------------------------- basic evaluations ----------------------- #

# Property calculation using (p, T)
P0 = 101_325.0  # Pa
T0 = 300.0      # K
state_PT = pg.get_props(cpx.PT_INPUTS, P0, T0, constants)
print(f"\nState from PT (p = {P0:.0f} Pa, T = {T0:.2f} K)")
cpx.print_dict(state_PT)

# Property calculation using (h, s)
H = state_PT["h"]
S = state_PT["s"]
state_hs = pg.get_props(cpx.HmassSmass_INPUTS, H, S, constants)
print("\nState from HmassSmass (h, s)")
cpx.print_dict(state_hs)

# Property calculation using (h, p)
state_hP = pg.get_props(cpx.HmassP_INPUTS, H, P0, constants)
print("\nState from HmassP (h, p)")
cpx.print_dict(state_hP)

# Property calculation using (p, s)
state_Ps = pg.get_props(cpx.PSmass_INPUTS, P0, S, constants)
print("\nState from PSmass (p, s)")
cpx.print_dict(state_Ps)

# Property calculation using (rho, h)
rho0 = state_PT["d"]
state_rhoh = pg.get_props(cpx.DmassHmass_INPUTS, rho0, H, constants)
print("\nState from DmassHmass (rho, h)")
cpx.print_dict(state_rhoh)

# ------------------- vectorized calculations and plotting ---------------- #

# Compute viscosity over the temperature range
T_range = np.linspace(300.0, 1000.0, 200)
mu_vals = pg.viscosity_from_T(T_range, constants)

# Create figure
fig, ax = plt.subplots(figsize=(6, 5))
ax.grid(False)
ax.set_xlabel("Temperature [K]")
ax.set_ylabel("Dynamic viscosity [Pa·s]")
ax.set_xlim([T_range.min(), T_range.max()])

# Plot viscosity
ax.plot(T_range, mu_vals)
plt.tight_layout(pad=1)

# Show plot if not disabled
if not os.environ.get("DISABLE_PLOTS"):
    plt.show()
