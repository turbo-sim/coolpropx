import time
import jax
import jax.numpy as jnp
import diffrax as dfx
import equinox as eqx
import optimistix as optx
import coolpropx as cpx
import matplotlib.pyplot as plt

from matplotlib import gridspec

cpx.set_plot_options(grid=False)

from coolpropx.perfect_gas import get_props
from examples.jaxprop.nozzle_model_core import (
    nozzle_single_phase_core,
    symmetric_nozzle_geometry,
)

from examples.jaxprop.nozzle_model_solver import (
    NozzleParams,
    IVPSettings,
    replace_param,
    compute_static_state,
)

# v1 solves the ode system using the space marching in non-autonomous form


# -----------------------------------------------------------------------------
# Main API to the converging-diverging nozzle model
# -----------------------------------------------------------------------------
@eqx.filter_jit
def nozzle_single_phase(
    params_model,
    params_solver,
):
    """
    1D variable-area nozzle with friction and optional heat transfer (Reynolds analogy).
    State vector: y = [v, rho, p].
    """

    # Compute inlet conditions iteratively
    state_in = compute_static_state(
        params_model.p0_in,
        params_model.d0_in,
        params_model.Ma_in,
        params_model.fluid,
    )
    p_in, rho_in, a_in = state_in["p"], state_in["rho"], state_in["a"]
    v_in = params_model.Ma_in * a_in
    y0 = jnp.array([v_in, rho_in, p_in])

    # Create and configure the solver
    solver = cpx.jax_import.make_diffrax_solver(params_solver.solver_name)
    adjoint = cpx.jax_import.make_diffrax_adjoint(params_solver.adjoint_name)
    term = dfx.ODETerm(eval_ode_rhs)
    ctrl = dfx.PIDController(rtol=params_solver.rtol, atol=params_solver.atol)
    ts = jnp.linspace(0.0, params_model.length, params_solver.number_of_points)
    saveat = dfx.SaveAt(ts=ts, t1=True, dense=False, fn=eval_ode_full)

    event = dfx.Event(
        cond_fn=_sonic_event_cond,
        root_finder=optx.Bisection(rtol=1e-10, atol=1e-10),
    )

    # Solve the ODE system
    sol = dfx.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=params_model.length,
        dt0=None,
        y0=y0,
        args=params_model,
        saveat=saveat,
        stepsize_controller=ctrl,
        adjoint=adjoint,
        event=event,
        max_steps=20_000,
    )

    return sol


# Define ODE RHS functions
def eval_ode_full(t, y, _):
    return nozzle_single_phase_core(t, y, args)


def eval_ode_rhs(t, y, _):
    return nozzle_single_phase_core(t, y, args)["rhs"]


# Event: stop when Ma^2 - 1 < tol
def _sonic_event_cond(t, y, args, **kwargs):
    v, rho, p = y
    a = get_props(cpx.DmassP_INPUTS, rho, p, args.fluid)["a"]
    Ma_sqr = (v / a) ** 2
    margin = 1e-5
    return Ma_sqr - (1.0 - margin)


# -----------------------------------------------------------------------------
# Converging-diverging nozzle example
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    # Define model parameters
    fluid_name = "air"
    fluid = cpx.perfect_gas.get_constants(fluid_name, T_ref=300, P_ref=101325)

    args = NozzleParams(
        Ma_in=0.25,
        p0_in=1.0e5,  # Pa
        d0_in=1.20,  # kg/m³
        D_in=0.050,  # m
        length=5.00,  # m
        roughness=1e-6,  # m
        T_wall=300.0,  # K
        heat_transfer=0.0,
        wall_friction=0.0,
        fluid=fluid,
        geometry=symmetric_nozzle_geometry,
    )

    params_solver = IVPSettings(solver_name="Dopri5", rtol=1e-8, atol=1e-8)

    # Inlet Mach number sensitivity analysis
    print("\n" + "-" * 60)
    print("Running inlet Mach number sensitivity analysis")
    print("-" * 60)
    input_array = jnp.asarray([0.05, 0.10, 0.15, 0.20, 0.25, 0.35])
    colors = plt.cm.magma(jnp.linspace(0.2, 0.8, len(input_array)))  # Generate colors
    solution_list = []
    for i, Ma in enumerate(input_array):
        t0 = time.perf_counter()
        args = replace_param(args, "Ma_in", Ma)
        sol = nozzle_single_phase(args, params_solver)
        print(
            f"Ma_in = {Ma:0.2f} | Solution time: {(time.perf_counter() - t0) * 1e3:7.3f} ms"
        )
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
        axs[0].plot(x, sol.ys["p0"] * 1e-5, linestyle="--", color=color)
        axs[0].plot(
            x,
            sol.ys["p"] * 1e-5,
            linestyle="-",
            color=color,
            label=rf"$\text{{Ma}}_\mathrm{{in}} = {val:0.3f}$",
        )
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
