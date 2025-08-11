# import os
# os.environ["JAX_DISABLE_JIT"] = "1"

import jax
import jax.numpy as jnp
import diffrax as dfx
import equinox as eqx
import optimistix as optx
import coolpropx as cpx
import matplotlib.pyplot as plt

from time import perf_counter
from matplotlib import gridspec

cpx.set_plot_options(grid=False)


# -----------------------------------------------------------------------------
# Main API to the converging-diverging nozzle model
# -----------------------------------------------------------------------------
@eqx.filter_jit
def nozzle_single_phase(
    params,
    fluid,
    wall_friction: bool = False,
    heat_transfer: bool = False,
    solver_name: str = "Dopri5",
    adjoint_name: str = "DirectAdjoint",
    number_of_points: int | None = None,
    rtol: float = 1e-6,
    atol: float = 1e-6,
):
    """
    1D variable-area nozzle with friction and optional heat transfer (Reynolds analogy).
    State vector: y = [v, rho, p].
    """
    # Rename parameters
    p0_in = params["p0_in"]
    T0_in = params["T0_in"]
    Ma_in = params["Ma_in"]
    length = params["length"]
    eps_wall = params["roughness"]
    T_wall = params["T_wall"]

    # Compute inlet conditions iteratively
    p_in, s0 = compute_inlet_static_state(p0_in, T0_in, Ma_in, fluid)
    state_in = get_props(cpx.PSmass_INPUTS, p_in, s0, fluid)
    rho_in, a_in = state_in["rho"], state_in["a"]
    v_in = Ma_in * a_in
    y0 = jnp.array([v_in, rho_in, p_in])

    # Group the ODE system constant parameters
    args = (length, eps_wall, T_wall, p_in, p0_in, wall_friction, heat_transfer, fluid)

    # Create and configure the solver
    solver = cpx.jax_import.make_diffrax_solver(solver_name)
    adjoint = cpx.jax_import.make_diffrax_adjoint(adjoint_name)
    term = dfx.ODETerm(_nozzle_odefun)
    ctrl = dfx.PIDController(rtol=rtol, atol=atol)
    if number_of_points is not None:
        ts = jnp.linspace(0.0, length, number_of_points)
        saveat = dfx.SaveAt(ts=ts, t1=True, dense=False, fn=_postprocess_ode)
    else:
        saveat = dfx.SaveAt(t1=True, fn=_postprocess_ode)

    # Define event for the singular point
    root_finder = optx.Bisection(rtol=1e-10, atol=1e-10)
    event = dfx.Event(
        cond_fn=_sonic_event_cond,
        root_finder=root_finder,
    )

    # Solve the ODE system
    sol = dfx.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=length,
        dt0=None,
        y0=y0,
        saveat=saveat,
        stepsize_controller=ctrl,
        args=args,
        adjoint=adjoint,
        event=event,
        max_steps=20_000,
    )

    return sol


# -----------------------------------------------------------------------------
# inlet static state from stagnation and Mach
# -----------------------------------------------------------------------------
def compute_inlet_static_state(p0, T0, Ma, fluid):
    """solve h0 - h(p,s0) - 0.5 a(p,s0)^2 Ma^2 = 0 for p"""
    st0 = get_props(cpx.PT_INPUTS, p0, T0, fluid)
    s0, h0 = st0["s"], st0["h"]

    def residual(p, _):
        st = get_props(cpx.PSmass_INPUTS, p, s0, fluid)
        a, h = st["a"], st["h"]
        v = a * Ma
        return h0 - h - 0.5 * v * v

    p_init = 0.9 * p0
    solver = optx.Newton(rtol=1e-10, atol=1e-10)
    sol = optx.root_find(residual, solver, y0=p_init, args=None)
    return sol.value, s0


# -----------------------------------------------------------------------------
# Right hand side of the ODE system
# -----------------------------------------------------------------------------

def _nozzle_odefun(t, y, args):
    x = t
    v, rho, p = y
    (L, eps_wall, T_ext, _, _, wall_friction, heat_transfer, fluid) = args

    # Compute thermodynamic state
    state = get_props(cpx.DmassP_INPUTS, rho, p, fluid)
    T = state["T"]
    h = state["h"]
    s = state["s"]
    a = state["a"]
    cp = state["cp"]
    mu = state["mu"]
    G = state["gruneisen"]

    # Compute local nozzle geometry
    A, dAdx, perimeter, diameter = get_nozzle_geometry(x, L)

    # Compute wall friction
    if wall_friction:
        Re = v * rho * diameter / jnp.maximum(mu, 1e-12)
        f_D = get_friction_factor_haaland(Re, eps_wall, diameter)
        tau_w = get_wall_viscous_stress(f_D, rho, v)
    else:
        tau_w = 0.0

    # Compute heat transfer
    if heat_transfer:
        Re = v * rho * diameter / jnp.maximum(mu, 1e-12)
        f_D = get_friction_factor_haaland(Re, eps_wall, diameter)
        htc = get_heat_transfer_coefficient(f_D, v, rho, cp)
        q_w = htc * (T_ext - T)
    else:
        q_w = 0.0

    # Evaluate ODE system
    M = jnp.asarray([[rho, v, 0.0],[rho*v,0.0,1.0],[0.0,-(a**2),1.0]])
    b = jnp.asarray([
        -rho * v / A * dAdx,
        -(perimeter / A) * tau_w,
        (perimeter / A) * (G / v) * (tau_w * v + q_w),
    ])
    return jnp.linalg.solve(M, b)


# Event: stop when Ma^2 - 1 < tol
def _sonic_event_cond(t, y, args, **kwargs):
    v, rho, p = y
    (L, eps, T_ext, p_in, p0_in, _, _, fluid) = args
    a = get_props(cpx.DmassP_INPUTS, rho, p, fluid)["a"]
    Ma_sqr = (v / a) ** 2
    tolerance = 1e-5
    return Ma_sqr - (1.0 - tolerance)


def _postprocess_ode(t, y, args):
    x = t
    v, rho, p = y
    (length, eps, T_ext, _, _, _, _, fluid) = args

    # Compute geometry
    A, dAdx, perimeter, diameter = get_nozzle_geometry(x, length)

    # Compute thermodynamic state
    state = get_props(cpx.DmassP_INPUTS, rho, p, fluid)

    # Compute stagnation state
    h0 = state["h"] + 0.5 * v**2
    state0 = get_props(cpx.HmassSmass_INPUTS, h0, state["s"], fluid)
    p0 = state0["p"]
    T0 = state0["T"]

    # Compute Raynolds number
    Re  = v * rho * diameter / state["mu"]
    f_D = get_friction_factor_haaland(Re, eps, diameter)

    base = {
        "x": x, "v": v, "rho": rho, "p": p,
        "A": A, "dAdx": dAdx, "diameter": diameter, "perimeter": perimeter,
        "h0": h0, "p0": p0, "T0": T0,
        "Ma": v / state["a"], "Re": Re, "f_D": f_D, 
        "m_dot": rho * v * A,

    }

    return {**base, **state}


# ------------------------------------------------------------------
# Describe the geometry of the converging diverging nozzle
# ------------------------------------------------------------------
def get_nozzle_area(x, L):
    """Area A(x) using physical x (m) and length L (m)."""
    # Special case of symmetric parabolic nozzle
    A_THROAT = 0.15  # m^2
    A_INLET  = 0.30  # m^2
    xi = jnp.clip(x / L, 0.0, 1.0)
    return A_INLET - 4.0 * (A_INLET - A_THROAT) * xi * (1.0 - xi)

# Take the gradient with JAX
get_nozzle_area_gradient = jax.grad(get_nozzle_area, argnums=0)

def get_nozzle_geometry(x, L):
    """Nozzle geometric parameters as a function of physical x (m) and total length L (m)."""
    A = get_nozzle_area(x, L)              # m^2
    dAdx = get_nozzle_area_gradient(x, L)  # m
    radius = jnp.sqrt(A / jnp.pi)          # m
    diameter = 2.0 * radius                # m
    perimeter = jnp.pi * diameter          # m
    return A, dAdx, perimeter, diameter


# -----------------------------------------------------------------------------
# Functions to calculate heat transfer and friction
# -----------------------------------------------------------------------------

def get_friction_factor_haaland(Reynolds, roughness, diameter):
    """
    Computes the Darcy-Weisbach friction factor using the Haaland equation.

    The Haaland equation provides an explicit formulation for the friction factor
    that is simpler to use than the Colebrook equation, with an acceptable level
    of accuracy for most engineering applications.
    This function implements the Haaland equation as it is presented in many fluid
    mechanics textbooks, such as "Fluid Mechanics Fundamentals and Applications"
    by Cengel and Cimbala (equation 12-93).

    Parameters
    ----------
    reynolds : float
        The Reynolds number, dimensionless, indicating the flow regime.
    roughness : float
        The absolute roughness of the pipe's internal surface (m).
    diameter : float
        The inner diameter of the pipe (m).

    Returns
    -------
    f : float
        The computed friction factor, dimensionless.
    """
    Re_safe = jnp.maximum(Reynolds, 1.0)
    term = 6.9 / Re_safe + (roughness / diameter / 3.7) ** 1.11
    f = (-1.8 * jnp.log10(term)) ** -2
    return f

def get_wall_viscous_stress(darcy_friction_factor, density, velocity):
    """Wall shear stress from Darcy-Weisbach friction factor.

    Parameters
    ----------
    darcy_friction_factor : float
        Darcy-Weisbach friction factor (dimensionless).
    density : float
        Fluid density (kg/m^3).
    velocity : float
        Fluid velocity (m/s).

    Returns
    -------
    float
        Wall shear stress (Pa).
    """
    return 0.125 * darcy_friction_factor * density * velocity**2


def get_heat_transfer_coefficient(
    velocity, density, heat_capacity, darcy_friction_factor
):
    """
    Estimates the heat transfer using the Reynolds analogy.

    This function is an adaptation of the Reynolds analogy which relates the heat transfer
    coefficient to the product of the Fanning friction factor, velocity, density, and heat
    capacity of the fluid. The Fanning friction factor one fourth of the Darcy friction factor.

    Parameters
    ----------
    velocity : float
        Velocity of the fluid (m/s).
    density : float
        Density of the fluid (kg/m^3).
    heat_capacity : float
        Specific heat capacity of the fluid at constant pressure (J/kg·K).
    darcy_friction_factor : float
        Darcy friction factor, dimensionless.

    Returns
    -------
    float
        Estimated heat transfer coefficient (W/m^2·K).

    """
    fanning_friction_factor = darcy_friction_factor / 4
    return 0.5 * fanning_friction_factor * velocity * density * heat_capacity


# -----------------------------------------------------------------------------
# Converging-diverging nozzle example
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    
    # Define model parameters
    backend = "perfect_gas"
    # backend = "jaxprop"
    fluid_name = "air"
    params = {
        "p0_in": 1.0e5,        # Pa
        "T0_in": 300.0,        # K
        "Ma_in": 0.20,         # -
        "D_in": 0.050,         # m
        "length": 5.00,        # m
        "roughness": 10e-6,    # m
        "T_wall": 300.0,       # K
    }
    
    # Convert to JAX array types
    params = {k: jnp.asarray(v) for k, v in params.items()}

    # Define working fluid depending on backend selected
    if backend == "perfect_gas":
        from coolpropx.perfect_gas import get_props, get_perfect_gas_constants
        fluid = get_perfect_gas_constants(fluid_name, params["T0_in"], params["p0_in"])

    elif backend == "jaxprop":
        from coolpropx.jaxprop import get_props
        fluid = cpx.Fluid(name=fluid_name, backend="HEOS")

    else:
        raise ValueError("Invalid fluid backend seclection")
    
    # Inlet Mach number sensitivity analysis
    print("\n" + "-" * 60)
    print("Running inlet Mach number sensitivity analysis")
    print("-" * 60)
    Ma_array = jnp.asarray([0.05, 0.10, 0.15, 0.20, 0.25, 0.35])
    colors = plt.cm.magma(jnp.linspace(0.2, 0.8, len(Ma_array)))  # Generate colors
    solution_list = []
    for i, Ma in enumerate(Ma_array):
        t0 = perf_counter()
        params["Ma_in"] = Ma
        sol = nozzle_single_phase(params, fluid, number_of_points=100)#, solver_name="Kvaerno5")
        print(f"Ma_in = {Ma:0.2f} | Solultion time: {(perf_counter() - t0) * 1e3:7.3f} ms")
        solution_list.append(sol)


    # Create the figure
    fig = plt.figure(figsize=(5, 7))
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 3, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    # geometry from any solution (shared nozzle)
    xg = solution_list[0].ys["x"]
    rg = solution_list[0].ys["diameter"] / 2.0

    # --- row 1: pressure (bar): solid p0, dashed p ---
    ax1.set_ylabel("Pressure (bar)")
    for color, Ma, sol in zip(colors, Ma_array, solution_list):
        x = sol.ys["x"]
        ax1.plot(x, sol.ys["p0"] * 1e-5, linestyle="-",  color=color, label=f"Ma={Ma:.2f} p0")
        ax1.plot(x, sol.ys["p"]  * 1e-5, linestyle="--", color=color, label=f"Ma={Ma:.2f} p")

    # legend: keep only p0 entries
    handles, labels = ax1.get_legend_handles_labels()
    keep = [(" p0" in lab) for lab in labels]
    ax1.legend([h for h, k in zip(handles, keep) if k],
            [lab.replace(" p0", "") for lab, k in zip(labels, keep) if k],
            title="Inlet Mach", fontsize=9, title_fontsize=9, loc="lower right")

    # --- row 2: Mach number ---
    ax2.set_ylabel("Mach number (-)")
    for color, Ma, sol in zip(colors, Ma_array, solution_list):
        ax2.plot(sol.ys["x"], sol.ys["Ma"], color=color, label=f"Ma={Ma:.2f}")

    # --- row 3: nozzle geometry ---
    ax3.fill_between(xg, -rg, +rg, color="lightgrey")  # shaded nozzle
    x_closed = jnp.concatenate([xg, xg[::-1], xg[:1]])
    y_closed = jnp.concatenate([rg, -rg[::-1], rg[:1]])
    ax3.plot(x_closed, y_closed, "k", linewidth=1.2)
    r_abs_max = float(jnp.max(jnp.abs(rg)))
    ax3.set_ylim(-1.5 * r_abs_max, 1.5 * r_abs_max)
    ax3.set_aspect("equal", adjustable="box")
    ax3.set_xlabel("Axial coordinate x (m)")
    fig.tight_layout(pad=1)

    # Show figures
    plt.show()
