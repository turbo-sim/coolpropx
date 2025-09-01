import numpy as np
import coolpropx as cpx
import coolpropx.perfect_gas as pg

# config
input_pair = cpx.HmassP_INPUTS
h = 3.00e5    # J/kg
p = 101325.0  # Pa

# constants for air
try:
    constants = pg.GAS_CONSTANTS_AIR
except AttributeError:
    constants = pg.get_constants("air", 298.15, 101325.0, display=False)

# base state
base = pg.get_props(input_pair, h, p, constants)
print("base state:")
print(f"  T   : {float(base['T']):+0.6f} K")
print(f"  p   : {float(base['p']):+0.6f} Pa")
print(f"  rho : {float(base['d']):+0.6f} kg/m^3")
print(f"  h   : {float(base['h']):+0.6f} J/kg")
print(f"  s   : {float(base['s']):+0.6f} J/(kg K)")

# finite differences jacobian
dfdx_fd, dfdp_fd = pg.get_props_gradient(input_pair, constants, h, p, method="fd")

# jax jacobian (will crash if jax is not installed)
dfdx_jax, dfdp_jax = pg.get_props_gradient(input_pair, constants, h, p, method="jax")

print("\npartials (finite differences):")
print(f"  dT/dh|p   : {float(dfdx_fd['T']):+0.10e} K*kg/J")
print(f"  dT/dp|h   : {float(dfdp_fd['T']):+0.10e} K/Pa")
print(f"  drho/dh|p : {float(dfdx_fd['d']):+0.10e} kg*m^-3*kg/J")
print(f"  drho/dp|h : {float(dfdp_fd['d']):+0.10e} kg*m^-3/Pa")

print("\npartials (jax):")
print(f"  dT/dh|p   : {float(dfdx_jax['T']):+0.10e} K*kg/J")
print(f"  dT/dp|h   : {float(dfdp_jax['T']):+0.10e} K/Pa")
print(f"  drho/dh|p : {float(dfdx_jax['d']):+0.10e} kg*m^-3*kg/J")
print(f"  drho/dp|h : {float(dfdp_jax['d']):+0.10e} kg*m^-3/Pa")

# relative errors
def rel(a, b):
    a = float(a); b = float(b)
    return abs(a - b) / max(1.0, abs(b))

print("\nrelative errors (fd vs jax):")
print(f"  dT/dh|p   : {rel(dfdx_fd['T'], dfdx_jax['T']):+0.10e}")
print(f"  dT/dp|h   : {rel(dfdp_fd['T'], dfdp_jax['T']):+0.10e}")
print(f"  drho/dh|p : {rel(dfdx_fd['d'], dfdx_jax['d']):+0.10e}")
print(f"  drho/dp|h : {rel(dfdp_fd['d'], dfdp_jax['d']):+0.10e}")

# identity checks using FD results
cp, _ = pg.specific_heat(constants)
R = constants["R"]
T = float(base["T"])

print("\nidentity checks (value, expected, rel. error):")
val_dT_dh = float(dfdx_fd["T"])
exp_dT_dh = 1.0 / float(cp)
err_dT_dh = abs(val_dT_dh - exp_dT_dh) / abs(exp_dT_dh)
print(f"  dT/dh|p   : {val_dT_dh:+0.6f}   {exp_dT_dh:+0.6f}   {err_dT_dh:+0.6f}")

val_drho_dp = float(dfdp_fd["d"])
exp_drho_dp = 1.0 / (float(R) * T)
err_drho_dp = abs(val_drho_dp - exp_drho_dp) / abs(exp_drho_dp)
print(f"  drho/dp|h : {val_drho_dp:+0.6f}   {exp_drho_dp:+0.6f}   {err_drho_dp:+0.6f}")
