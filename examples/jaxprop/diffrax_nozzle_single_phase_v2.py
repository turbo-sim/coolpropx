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

from examples.jaxprop.nozzle_model_core import nozzle_single_phase_autonomous

# v1 solves the ode system using the space marching in non-autonomous form
# v2 solves the autonomous system with events for the bounds of domain ends

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
    p_in = params["p_in"]
    p0_in = params["p0_in"]
    T0_in = params["T0_in"]
    length = params["length"]
    eps_wall = params["roughness"]
    T_wall = params["T_wall"]

    # Compute inlet conditions
    state0_in = get_props(cpx.PT_INPUTS, p0_in, T0_in, fluid)
    h0_in, s0_in = state0_in["h"], state0_in["s"]
    state_in = get_props(cpx.PSmass_INPUTS, p_in, s0_in, fluid)
    rho_in, h_in = state_in["rho"], state_in["h"]
    v_in = jnp.sqrt(2 * (h0_in - h_in))
    x_start = 1e-9  # Start slightly after the nozzle inlet
    y0 = jnp.array([x_start, v_in, rho_in, p_in])

    # Group the ODE system constant parameters
    args = (length, eps_wall, T_wall, wall_friction, heat_transfer, fluid)

    # Define functions to evaluate the ODE
    eval_ode_full = nozzle_single_phase_autonomous
    eval_ode_rhs = lambda t, y, args: eval_ode_full(t, y, args)["rhs_autonomous"]

    # Create and configure the solver
    t_start = 0.0   # Start at tau=0 (arbitrary)
    t_final = 1e+9  # Large value that will not be reached
    solver = cpx.jax_import.make_diffrax_solver(solver_name)
    adjoint = cpx.jax_import.make_diffrax_adjoint(adjoint_name)
    term = dfx.ODETerm(eval_ode_rhs)
    ctrl = dfx.PIDController(rtol=rtol, atol=atol)

    # Define event for reaching the end of the domain [0, L]
    event = dfx.Event(
        cond_fn=lambda t, y, args, **kwargs: jnp.minimum(y[0], args[0] - y[0]),
        root_finder=optx.Bisection(rtol=1e-10, atol=1e-10),
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
    saveat = dfx.SaveAt(ts=ts, t1=True, fn=eval_ode_full)
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
        from coolpropx.perfect_gas import get_props, get_perfect_gas_constants
        fluid = get_perfect_gas_constants(fluid_name, params["T0_in"], params["p0_in"])

    elif backend == "jaxprop":
        from coolpropx.jaxprop import get_props
        fluid = cpx.Fluid(name=fluid_name, backend="HEOS")

    else:
        raise ValueError("Invalid fluid backend seclection")
    
    # Inlet Mach number sensitivity analysis
    print("\n" + "-" * 60)
    print("Running inlet pressure sensitivity analysis")
    print("-" * 60)
    input_array = jnp.linspace(0.99, 0.90, 10)
    # PR_array = jnp.asarray([0.99, 0.95, 0.94, 0.939, 0.938, 0.9375, 0.9373, 0.937, 0.935, 0.932, 0.93, 0.92, 0.91])
    # PR_array = jnp.asarray([0.9385, 0.938,  0.9375, 0.937])
    colors = plt.cm.magma(jnp.linspace(0.2, 0.8, len(input_array)))  # Generate colors
    solution_list = []
    for i, val in enumerate(input_array):
        t0 = perf_counter()
        params["p_in"] = val*params["p0_in"]
        sol = nozzle_single_phase(params, fluid, number_of_points=100, solver_name="Dopri5")#, solver_name="Kvaerno5")
        print(f"p_in/p0 = {val:0.5f} | Solution time: {(perf_counter() - t0) * 1e3:7.3f} ms")
        solution_list.append(sol)

    # Create the figure
    fig = plt.figure(figsize=(5, 7))
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 3, 1])
    xg = solution_list[0].ys["x"]
    rg = solution_list[0].ys["diameter"] / 2.0
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    axs = [ax0, ax1, ax2]
    
    # --- row 1: pressure (bar): solid p0, dashed p ---
    axs[0].set_ylabel("Pressure (bar)")
    for color, val, sol in zip(colors, input_array, solution_list):
        x = sol.ys["x"]
        axs[0].plot(x, sol.ys["p0"] * 1e-5, linestyle="--",  color=color)
        axs[0].plot(x, sol.ys["p"]  * 1e-5, linestyle="-", color=color, label=rf"$p_\mathrm{{in}}/p_0 = {val:0.3f}$")
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
