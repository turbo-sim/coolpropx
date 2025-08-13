import os
import numpy as np

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
SAVE_FIGURES = True  # Set to False to just display plots

fluid = cpx.Fluid(fluid_name)

# ---------------------------
# Generate Table
# ---------------------------
table = cpx.bicubic.generate_property_table(hmin, hmax, Pmin, Pmax, fluid_name, Nh, Np)


# ---------------------------
# Step 2: Interpolate at (h, P)
# ---------------------------

test_h = 300e3   # Test enthalpy [J/kg]
test_P = 8e6     # Test pressure [Pa]
print(f"\n Interpolating at h = {test_h:.0f} J/kg, P = {test_P/1e5:.2f} bar")

interp_props = cpx.bicubic.bicubic_interpolant_property(test_h, test_P, table)

# ---------------------------
# Step 3: Compare with CoolProp
# ---------------------------
print("\n Comparison with CoolProp values:")
cp_keys = {'T': 'T', 'd': 'D', 's': 'S', 'mu': 'V', 'k': 'L'}

for prop, interp_val in interp_props.items():
    cp_key = cp_keys[prop]
    try:
        cp_val = fluid.get_state(cpx.HmassP_INPUTS, test_h, test_P)[prop]
        rel_error = abs(interp_val - cp_val) / cp_val
        print(f" - {prop}: Interpolated = {interp_val:.4e}, CoolProp = {cp_val:.4e}, Rel. Error = {rel_error:.2%}")
    except Exception as e:
        print(f" - {prop}: Interpolated = {interp_val:.4e}, CoolProp = N/A (Error: {e})")
