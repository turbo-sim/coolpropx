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
    p_in = params["p_in"]
    p0_in = params["p0_in"]
    T0_in = params["T0_in"]
    length = params["length"]
    eps_wall = params["roughness"]
    T_wall = params["T_wall"]

    # Compute inlet conditions iteratively
    state0_in = get_props(cpx.PT_INPUTS, p0_in, T0_in, fluid)
    h0_in, s0_in = state0_in["h"], state0_in["s"]
    state_in = get_props(cpx.PSmass_INPUTS, p_in, s0_in, fluid)
    rho_in, a_in, h_in = state_in["rho"], state_in["a"], state_in["h"]
    v_in = jnp.sqrt(2 * (h0_in - h_in))
    x_start = 1e-9  # Start slightly after the nozzle inlet
    y0 = jnp.array([x_start, v_in, rho_in, p_in])

    # Group the ODE system constant parameters
    args = (length, eps_wall, T_wall, p_in, p0_in, wall_friction, heat_transfer, fluid)

    # Create and configure the solver
    t_start = 0.0   # Start at tau=0 (arbitrary)
    t_final = 1e+9  # Large value that will not be reached
    solver = cpx.jax_import.make_diffrax_solver(solver_name)
    adjoint = cpx.jax_import.make_diffrax_adjoint(adjoint_name)
    term = dfx.ODETerm(_nozzle_odefun_autonomous)
    ctrl = dfx.PIDController(rtol=rtol, atol=atol)

    # Define event for the singular point
    root_finder = optx.Bisection(rtol=1e-10, atol=1e-10)
    event = dfx.Event(
        cond_fn=_event_nozzle_bounds,
        root_finder=root_finder,
    )

    # Solve the ODE system without saving solution
    sol = dfx.diffeqsolve(
        term,
        solver,
        t0=t_start,
        t1=t_final,
        dt0=None,
        y0=y0,
        stepsize_controller=ctrl,
        args=args,
        adjoint=adjoint,
        event=event,
        max_steps=20_000,
    )   

    # Solve the ODE system again saving the solution
    ts = jnp.linspace(1e-9, sol.ts[-1], number_of_points)
    saveat = dfx.SaveAt(ts=ts, t1=True, fn=_postprocess_ode_autonomous)
    sol_dense = dfx.diffeqsolve(
        term,
        solver,
        t0=1e-9,
        t1=sol.ts[-1],
        dt0=None,
        y0=y0,
        saveat=saveat,
        stepsize_controller=ctrl,
        args=args,
        adjoint=adjoint,
        # event=event,  # No need to check event this time
        max_steps=20_000,
    )

    return sol_dense


# -----------------------------------------------------------------------------
# Right hand side of the ODE system
# -----------------------------------------------------------------------------

def _nozzle_odefun_autonomous(tau, Y, args):
    """
    Autonomous formulation of the nozzle equations:
        dx/dt   = det(A)
        dy_i/dt = det(A with column i replaced by b)

    State vector: Y = [x, v, rho, p]
    """
    x, v, rho, p = Y
    (L, eps_wall, T_ext, _, _, wall_friction, heat_transfer, fluid) = args

    # --- Thermodynamic state ---
    state = get_props(cpx.DmassP_INPUTS, rho, p, fluid)
    T  = state["T"]
    h  = state["h"]
    s  = state["s"]
    a  = state["a"]
    cp = state["cp"]
    mu = state["mu"]
    G  = state["gruneisen"]

    # --- Geometry ---
    A, dAdx, perimeter, diameter = get_nozzle_geometry(x, L)

    # --- Wall friction ---
    if wall_friction:
        Re = v * rho * diameter / jnp.maximum(mu, 1e-12)
        f_D = get_friction_factor_haaland(Re, eps_wall, diameter)
        tau_w = get_wall_viscous_stress(f_D, rho, v)
    else:
        tau_w = 0.0

    # --- Heat transfer ---
    if heat_transfer:
        Re = v * rho * diameter / jnp.maximum(mu, 1e-12)
        f_D = get_friction_factor_haaland(Re, eps_wall, diameter)
        htc = get_heat_transfer_coefficient(v, rho, cp, f_D)
        q_w = htc * (T_ext - T)
    else:
        q_w = 0.0

    # --- Build A matrix and b vector ---
    A_mat = jnp.array([
        [rho,     v,         0.0],
        [rho*v,   0.0,       1.0],
        [0.0,     -(a**2),   1.0]
    ])

    b_vec = jnp.array([
        -rho * v / A * dAdx,
        -(perimeter / A) * tau_w,
        (perimeter / A) * (G / v) * (tau_w * v + q_w),
    ])

    # --- Determinants ---
    D = jnp.linalg.det(A_mat)

    # Replace columns one by one to compute N_i
    N = []
    for i in range(3):
        A_mod = A_mat.at[:, i].set(b_vec)
        N.append(jnp.linalg.det(A_mod))
    N = jnp.array(N)

    # --- Autonomous system: dx/dτ = D, dy/dτ = N_i ---
    dx_dtau   = D
    dv_dtau   = N[0]
    drho_dtau = N[1]
    dp_dtau   = N[2]

    return jnp.array([dx_dtau, dv_dtau, drho_dtau, dp_dtau])



def _event_nozzle_bounds(t, y, args, **kwargs):
    x, v, rho, p = y
    (L, eps, T_ext, p_in, p0_in, _, _, fluid) = args
    # Positive if inside domain, negative outside
    return jnp.minimum(x, L - x)



def _postprocess_ode_autonomous(tau, Y, args):
    x, v, rho, p = Y
    (length, eps, T_ext, _, _, _, _, fluid) = args

    # Geometry
    A, dAdx, perimeter, diameter = get_nozzle_geometry(x, length)

    # Thermodynamic state
    state = get_props(cpx.DmassP_INPUTS, rho, p, fluid)

    # Stagnation state
    h0 = state["h"] + 0.5 * v**2
    state0 = get_props(cpx.HmassSmass_INPUTS, h0, state["s"], fluid)
    p0 = state0["p"]
    T0 = state0["T"]

    # Reynolds number
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
    xi = x/L
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
        "p_in": 0.9e5,         # Pa
        "p0_in": 1.0e5,        # Pa
        "T0_in": 300.0,        # K
        "D_in": 0.050,         # m
        "length": 5.00,        # m
        "roughness": 10e-6,    # m
        "T_wall": 300.0,       # K
    }
    
    # Convert to JAX array types
    params = {k: jnp.asarray(v) for k, v in params.items()}

    # Define working fluid depending on backend selected
    if backend == "perfect_gas":
        from coolprop.perfect_gas import get_props, get_perfect_gas_constants
        fluid = get_perfect_gas_constants(fluid_name, params["T0_in"], params["p0_in"])

    elif backend == "jaxprop":
        from coolprop.jaxprop import get_props
        fluid = cpx.Fluid(name=fluid_name, backend="HEOS")

    else:
        raise ValueError("Invalid fluid backend seclection")
    
    # Inlet Mach number sensitivity analysis
    print("\n" + "-" * 60)
    print("Running inlet Mach number sensitivity analysis")
    print("-" * 60)
    PR_array = jnp.asarray([0.95, 0.94, 0.939, 0.938, 0.9375, 0.9373, 0.937, 0.935, 0.932, 0.93, 0.92, 0.91])
    colors = plt.cm.magma(jnp.linspace(0.2, 0.8, len(PR_array)))  # Generate colors
    solution_list = []
    for i, PR in enumerate(PR_array):
        t0 = perf_counter()
        params["p_in"] = PR*params["p0_in"]
        sol = nozzle_single_phase(params, fluid, number_of_points=100)#, solver_name="Kvaerno5")
        print(f"p_in/p0 = {PR:0.2f} | Solultion time: {(perf_counter() - t0) * 1e3:7.3f} ms")
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
    for color, PR, sol in zip(colors, PR_array, solution_list):
        x = sol.ys["x"]
        ax1.plot(x, sol.ys["p0"] * 1e-5, linestyle="-",  color=color, label=f"p/p0={PR:.5f}")
        ax1.plot(x, sol.ys["p"]  * 1e-5, linestyle="--", color=color)
    ax1.legend(loc="lower right", fontsize=7)

    # --- row 2: Mach number ---
    ax2.set_ylabel("Mach number (-)")
    for color, PR, sol in zip(colors, PR_array, solution_list):
        ax2.plot(sol.ys["x"], sol.ys["Ma"], color=color, label=f"p/p0={PR:.5f} p")

    # --- row 3: nozzle geometry ---
    ax3.fill_between(xg, -rg, +rg, color="lightgrey")  # shaded nozzle
    x_closed = jnp.concatenate([xg, xg[::-1], xg[:1]])
    y_closed = jnp.concatenate([rg, -rg[::-1], rg[:1]])
    ax3.plot(x_closed, y_closed, "k", linewidth=1.2)
    # r_abs_max = float(jnp.max(jnp.abs(rg)))
    # ax3.set_ylim(-1.5 * r_abs_max, 1.5 * r_abs_max)
    ax3.set_aspect("equal", adjustable="box")
    ax3.set_xlabel("Axial coordinate x (m)")
    fig.tight_layout(pad=1)

    # Show figures
    plt.show()
