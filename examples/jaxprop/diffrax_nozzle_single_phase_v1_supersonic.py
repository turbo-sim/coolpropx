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

from examples.jaxprop.nozzle_model_core import nozzle_single_phase_core

# v1 solves the ode system using the space marching in non-autonomous form


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
    number_of_points: int = 50,
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
    args = (length, eps_wall, T_wall, wall_friction, heat_transfer, fluid)

    # Define functions to evaluate the ODE
    eval_ode_full = nozzle_single_phase_core
    eval_ode_rhs = lambda t, y, args: eval_ode_full(t, y, args)["rhs"]

    # Create and configure the solver
    solver = cpx.jax_import.make_diffrax_solver(solver_name)
    adjoint = cpx.jax_import.make_diffrax_adjoint(adjoint_name)
    term = dfx.ODETerm(eval_ode_rhs)
    ctrl = dfx.PIDController(rtol=rtol, atol=atol)
    ts = jnp.linspace(length/2, length, number_of_points)
    saveat = dfx.SaveAt(ts=ts, t1=True, dense=False, fn=eval_ode_full)

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
        t0=length/2,
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

# Event: stop when Ma^2 - 1 < tol
def _sonic_event_cond(t, y, args, **kwargs):
    v, rho, p = y
    fluid = args[-1]
    a = get_props(cpx.DmassP_INPUTS, rho, p, fluid)["a"]
    Ma_sqr = (v / a) ** 2
    tolerance = 1e-5
    return Ma_sqr - (1.0 - tolerance)


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
        "Ma_in": 1.01,         # -
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
    input_array = jnp.asarray([1.01])
    colors = plt.cm.magma(jnp.linspace(0.2, 0.8, len(input_array)))  # Generate colors
    solution_list = []
    for i, Ma in enumerate(input_array):
        t0 = perf_counter()
        params["Ma_in"] = Ma
        sol = nozzle_single_phase(params, fluid, number_of_points=100)#, solver_name="Kvaerno5")
        print(f"Ma_in = {Ma:0.2f} | Solution time: {(perf_counter() - t0) * 1e3:7.3f} ms")
        solution_list.append(sol)

    # Create the figure
    fig = plt.figure(figsize=(5, 7))
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 3, 1])
    xg = solution_list[0].ys["x"]
    rg = solution_list[0].ys["diameter"] / 2.0
    axs = [
        fig.add_subplot(gs[0]),
        fig.add_subplot(gs[1], sharex=None),  # temporarily no sharex
        fig.add_subplot(gs[2], sharex=None)
    ]

    # --- row 1: pressure (bar): solid p0, dashed p ---
    axs[0].set_ylabel("Pressure (bar)")
    for color, val, sol in zip(colors, input_array, solution_list):
        x = sol.ys["x"]
        axs[0].plot(x, sol.ys["p0"] * 1e-5, linestyle="--",  color=color)
        axs[0].plot(x, sol.ys["p"]  * 1e-5, linestyle="-", color=color, label=rf"$\text{{Ma}}_\mathrm{{in}} = {val:0.3f}$")
    axs[0].legend(loc="lower right", fontsize=7)

    # --- row 2: Mach number ---
    axs[1].set_ylabel("Mach number (-)")
    for color, val, sol in zip(colors, input_array, solution_list):
        axs[1].plot(sol.ys["x"], sol.ys["Ma"], color=color)

    # --- row 3: nozzle geometry ---
    axs[2].fill_between(xg, -rg, +rg, color="lightgrey")  # shaded nozzle
    x_closed = jnp.concatenate([xg, xg[::-1], xg[:1]])
    y_closed = jnp.concatenate([rg, -rg[::-1], rg[:1]])
    axs[2].plot(x_closed, y_closed, "k", linewidth=1.2)
    # r_abs_max = float(jnp.max(jnp.abs(rg)))
    # ax3.set_ylim(-1.5 * r_abs_max, 1.5 * r_abs_max)
    axs[2].set_aspect("equal", adjustable="box")
    axs[2].set_xlabel("Axial coordinate x (m)")
    fig.tight_layout(pad=1)

    # Show figures
    plt.show()
