# import os
# os.environ["JAX_DISABLE_JIT"] = "1"

import jax
import jax.tree_util
import jax.numpy as jnp
import diffrax as dfx
import equinox as eqx
import optimistix as optx
import coolpropx as cpx
import matplotlib.pyplot as plt

from time import perf_counter
from matplotlib import gridspec

cpx.set_plot_options(grid=False)

# Import specific functions
from coolpropx.perfect_gas import get_props
from examples.jaxprop.nozzle_model_core import nozzle_single_phase_core
from jax_euler_equations_collocation_v3 import (
    solve_nozzle_model_collocation,
    initialize_flowfield,
)

# from coolpropx.jaxprop import get_props


# v1 solves the ode system using the space marching in non-autonomous form
# v2 solves the autonomous system with events for the bounds of domain ends
# v3 uses the non-autonomous system and determines the inlet critical condition with collocation method


# -----------------------------------------------------------------------------
# Main API to the converging-diverging nozzle model
# -----------------------------------------------------------------------------
@eqx.filter_jit
def transonic_nozzle_single_phase(
    params,
    fluid,
    wall_friction: bool = False,
    heat_transfer: bool = False,
    solver_name: str = "Dopri5",
    adjoint_name: str = "DirectAdjoint",
    number_of_points: int = 100,
    rtol: float = 1e-6,
    atol: float = 1e-6,
):
    """
    1D variable-area nozzle with friction and optional heat transfer (Reynolds analogy).
    State vector: y = [v, d, p].
    """
    # Rename parameters
    length = params["length"]
    eps_wall = params["roughness"]
    T_wall = params["T_wall"]
    Ma_low = params["Ma_low"]
    Ma_high = params["Ma_high"]

    # Numerical settings
    num_points = 50
    tolerance = 1e-8
    max_steps = 500
    jac_mode = "bwd"
    verbose = False

    # Determine critical condition with contuation method
    Ma_array = jnp.asarray([0.5, 0.9, 0.99999])
    z0 = initialize_flowfield(num_points, params, fluid)
    for Ma in Ma_array:
        params = jax.tree_util.tree_map(jnp.asarray, {**params, "Ma_target": Ma})
        out, sol = solve_nozzle_model_collocation(
            z0,
            params,
            fluid,
            wall_friction=wall_friction,
            heat_transfer=heat_transfer,
            num_points=num_points,
            max_steps=max_steps,
            jac_mode=jac_mode,
            verbose=verbose,
            rtol=tolerance,
            atol=tolerance,
        )

        # Substitute values, keep same array
        z0 = z0.at[:].set(sol.value)

    # solvers and controls
    solver = cpx.jax_import.make_diffrax_solver(solver_name)
    adjoint = cpx.jax_import.make_diffrax_adjoint(adjoint_name)
    ctrl = dfx.PIDController(rtol=rtol, atol=atol)

    # ---------- first pass: inlet → x_crit (base RHS) ----------

    # pick critical location and inlet state for ODE
    n_half = out["Ma"].shape[0] // 2
    idx_crit = jnp.argmin(jnp.abs(out["Ma"][:n_half] - Ma_low))
    x_crit = out["x"][idx_crit]
    y_inlet = jnp.array([out["v"][0], out["d"][0], out["p"][0]])

    # Define the integration interval
    t0 = 1e-9
    t1 = x_crit
    ts = jnp.linspace(t0, t1, number_of_points)

    # Define ode parameters
    args_base = (length, eps_wall, T_wall, wall_friction, heat_transfer, fluid)

    # Solve the subsonic trajectory
    term1 = dfx.ODETerm(_ode_rhs_evaluation_base)
    save1 = dfx.SaveAt(ts=ts, t1=True, fn=_ode_full_evaluation_base)
    sol1 = dfx.diffeqsolve(
        term1,
        solver,
        t0=t0,
        t1=t1,
        dt0=None,
        y0=y_inlet,
        saveat=save1,
        stepsize_controller=ctrl,
        args=args_base,
        adjoint=adjoint,
        max_steps=20_000,
    )

    # state at x_crit (last entry of first pass)
    x_crit = sol1.t1
    y_crit = jnp.array(
        [
            sol1.ys["v"][-1],
            sol1.ys["d"][-1],
            sol1.ys["p"][-1],
        ]
    )

    # evaluate rhs_crit once at (x_crit, y_crit)
    rhs_crit = nozzle_single_phase_core(x_crit, y_crit, args_base)["rhs"]

    # ---------- second pass: 0 → L (bypass/blend RHS) ----------
    t0 = 1e-9
    t1 = length
    ts = jnp.linspace(t0, t1, number_of_points)

    args_bypass = (
        length,
        eps_wall,
        T_wall,
        wall_friction,
        heat_transfer,
        fluid,
        rhs_crit,
        Ma_low,
        Ma_high,
    )
    term2 = dfx.ODETerm(_ode_rhs_evaluation_blend)
    save2 = dfx.SaveAt(ts=ts, t1=True, fn=_ode_full_evaluation_blend)
    sol2 = dfx.diffeqsolve(
        term2,
        solver,
        t0=t0,
        t1=t1,
        dt0=None,
        y0=y_inlet,
        saveat=save2,
        stepsize_controller=ctrl,
        args=args_bypass,
        adjoint=adjoint,
        max_steps=5_000,
    )

    return sol2


# -----------------------------------------------------------------------------
# Right hand side of the ODE system
# -----------------------------------------------------------------------------


def _ode_full_evaluation_base(t, y, args):
    return nozzle_single_phase_core(t, y, args)


def _ode_rhs_evaluation_base(t, y, args):
    return _ode_full_evaluation_base(t, y, args)["rhs"]



def _ode_full_evaluation_blend(t, y, args):
    """
    piecewise rhs:
      - outside [Ma_low, Ma_high] -> rhs_true
      - inside  [Ma_low, Ma_high] -> constant rhs_crit
    args = (L, eps_wall, T_wall, wall_friction, heat_transfer, fluid,
            rhs_crit, Ma_low, Ma_high)
    """
    (L, eps_wall, T_wall, wall_friction, heat_transfer, fluid,
     rhs_crit, Ma_low, Ma_high) = args

    base = nozzle_single_phase_core(t, y, (L, eps_wall, T_wall,
                                           wall_friction, heat_transfer, fluid))
    M = base["Ma"]
    rhs_true = base["rhs"]
    rhs_inside = rhs_crit

    # # Second option
    # w = smooth_cos_ramp(M, Ma_low, Ma_high)  # 0 → 1 across band
    # w = _smoothstep(M, Ma_low, Ma_high)
    # rhs_inside = (1.0 - w) * rhs_crit + w * rhs_true

    # Impose only inside the Ma limits
    inside = (M >= Ma_low) & (M <= Ma_high)
    rhs_blend = jnp.where(inside, rhs_inside, rhs_true)

    return {**base, "rhs_blend": rhs_blend}

def _ode_rhs_evaluation_blend(t, y, args):
    return _ode_full_evaluation_blend(t, y, args)["rhs_blend"]


def smooth_band_cos(M, M_lo, M_hi):
    """C11 'raised-cosine' band. 0 at M_lo/M_hi with zero slope; ~1 at M=mid."""
    mid = 0.5 * (M_lo + M_hi)
    r   = 0.5 * (M_hi - M_lo) + 1e-12
    u   = jnp.clip((M - mid) / r, -1.0, 1.0)     # outside band -> ±1 (flat)
    return 0.5 * (1.0 + jnp.cos(jnp.pi * u))     # 1 at mid, 0 at edges

def smooth_cos_ramp(M, M_lo, M_hi):
    """0 at M_lo, 1 at M_hi, smooth cosine ramp"""
    u = jnp.clip((M - M_lo) / (M_hi - M_lo + 1e-12), 0.0, 1.0)
    return 0.5 * (1.0 - jnp.cos(jnp.pi * u))


def _smoothstep(M, M_lo, M_hi):
    # 0 at M_lo, 1 at M_hi, C1-continuous, monotone
    u = jnp.clip((M - M_lo) / (M_hi - M_lo + 1e-12), 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


# -----------------------------------------------------------------------------
# Converging-diverging nozzle example
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    # Define model parameters
    backend = "perfect_gas"
    fluid_name = "air"
    params = jax.tree_util.tree_map(
        jnp.asarray,
        {
            "p0_in": 1.0e5,  # Pa
            "d0_in": 1.20,  # kg/m3
            "D_in": 0.050,  # m
            "length": 5.00,  # m
            "roughness": 1e-6,  # m
            "T_wall": 300.0,  # K
            "Ma_low": 0.99,
            "Ma_high": 1.01,
        },
    )

    # Define working fluid depending on backend selected
    if backend == "perfect_gas":
        from coolpropx.perfect_gas import get_props, get_perfect_gas_constants

        fluid = get_perfect_gas_constants(fluid_name, T_ref=300, P_ref=101325)

    elif backend == "jaxprop":
        from coolpropx.jaxprop import get_props

        fluid = cpx.Fluid(name=fluid_name, backend="HEOS")

    else:
        raise ValueError("Invalid fluid backend seclection")

    # Solve the problem
    print("\n" + "-" * 60)
    print("Evaluating transonic solution")
    print("-" * 60)
    input_array = jnp.asarray([0, 0, 0])
    colors = plt.cm.magma(jnp.linspace(0.2, 0.8, len(input_array)))  # Generate colors
    solution_list = []
    for i, Ma in enumerate(input_array):
        t0 = perf_counter()
        sol = transonic_nozzle_single_phase(
            params,
            fluid,
            wall_friction=False,
            heat_transfer=False,
            number_of_points=100,
            solver_name="Dopri5",
        )
        print(f"Solutuon {i} | Solution time: {(perf_counter() - t0) * 1e3:7.3f} ms")
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
            marker="o",
            markersize="3",
            label=rf"$p_\mathrm{{in}}/p_0 = {val:0.3f}$",
        )
    axs[0].legend(loc="upper right", fontsize=7)

    # --- row 2: Mach number ---
    axs[1].set_ylabel("Mach number (-)")
    for color, val, sol in zip(colors, input_array, solution_list):
        axs[1].plot(sol.ys["x"], sol.ys["Ma"], color=color,
            marker="o",
            markersize="3",)

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
