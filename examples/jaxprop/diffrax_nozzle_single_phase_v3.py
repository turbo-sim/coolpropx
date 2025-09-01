import time
import jax
import jax.numpy as jnp
import diffrax as dfx
import equinox as eqx
import optimistix as optx
import coolpropx as cpx
import matplotlib.pyplot as plt

from matplotlib import gridspec
from coolpropx.perfect_gas import get_props
from examples.jaxprop.nozzle_model_core import (
    nozzle_single_phase_core,
    symmetric_nozzle_geometry,
)
from examples.jaxprop.nozzle_model_solver import (
    NozzleParams,
    BVPSettings,
    IVPSettings,
    replace_param,
    solve_nozzle_model_collocation,
    initialize_flowfield,
)

cpx.set_plot_options(grid=False)


# v1 solves the ode system using the space marching in non-autonomous form
# v2 solves the autonomous system with events for the bounds of domain ends
# v3 uses the non-autonomous system and determines the inlet critical condition with collocation method


# -----------------------------------------------------------------------------
# Main API to the converging-diverging nozzle model
# -----------------------------------------------------------------------------
@eqx.filter_jit
def transonic_nozzle_single_phase(
    params_model: NozzleParams,
    params_bvp: BVPSettings,
    params_ivp: IVPSettings,
):
    """
    Transonic converging-diverging nozzle.
    First uses a collocation method to find the critical (sonic) state,
    then integrates the ODE system with a blended RHS near the critical location.
    """

    # --- 1. Find critical state with continuation --
    Ma_target = jnp.asarray([0.999])
    z0 = initialize_flowfield(params_bvp.num_points, params_model)
    for Ma in Ma_target:
        params_model = replace_param(params_model, "Ma_target", Ma)
        out, sol = solve_nozzle_model_collocation(
            initial_guess=z0,
            params_model=params_model,
            params_solver=params_bvp,
        )

        # Substitute values, keep same array
        z0 = z0.at[:].set(sol.value)

    # --- Define helpers with closure over args_base and fluid ---
    def ode_full_subsonic(t, y, args):
        return nozzle_single_phase_core(t, y, args)

    def ode_rhs_subsonic(t, y, args):
        return ode_full_subsonic(t, y, args)["rhs"]

    # --- 2. First pass: inlet → Ma_low ---
    solver = cpx.jax_import.make_diffrax_solver(params_ivp.solver_name)
    adjoint = cpx.jax_import.make_diffrax_adjoint(params_ivp.adjoint_name)
    ctrl = dfx.PIDController(rtol=params_ivp.rtol, atol=params_ivp.atol)
    y_inlet = jnp.array([out["v"][0], out["d"][0], out["p"][0]])
    event1 = dfx.Event(
        cond_fn=_mach_event_cond,
        root_finder=optx.Bisection(rtol=1e-10, atol=1e-10),
    )
    term1 = dfx.ODETerm(ode_rhs_subsonic)
    save1 = dfx.SaveAt(t1=True, fn=ode_full_subsonic)
    sol1 = dfx.diffeqsolve(
        term1,
        solver,
        t0=1e-9,
        t1=1000,
        dt0=None,
        y0=y_inlet,
        args=params_model,
        saveat=save1,
        stepsize_controller=ctrl,
        adjoint=adjoint,
        event=event1,
        max_steps=20_000,
    )

    # --- 3. Linearization close to critical point ---
    x_crit = sol1.ys["x"][-1]
    y_crit = jnp.array([sol1.ys["v"][-1], sol1.ys["d"][-1], sol1.ys["p"][-1]])
    f_star, Jy, Jt = _linearize_rhs_at(x_crit, y_crit, params_model)

    def ode_full_transonic(t, y, args):
        base = nozzle_single_phase_core(t, y, args)
        M = base["Ma"]
        rhs_true = base["rhs"]
        rhs_lin = f_star + Jy @ (y - y_crit) + Jt * (t - x_crit)

        # Start blending slightly after sonic conditions, finish at Ma_high
        Ma_start = 1.001
        Ma_end = args.Ma_high
        s = (M - Ma_start) / (Ma_end - Ma_start)
        w = _smoothstep(jnp.clip(s, 0.0, 1.0))  # smoothly 0→1 in that narrow range
        rhs_blend = (1.0 - w) * rhs_lin + w * rhs_true
        in_window = (M >= args.Ma_low) & (M <= args.Ma_high)
        rhs_blend = jnp.where(in_window, rhs_blend, rhs_true)

        return {**base, "rhs": rhs_true, "rhs_blend": rhs_blend}

    def ode_rhs_transonic(t, y, args):
        return ode_full_transonic(t, y, args)["rhs_blend"]

    # --- 4. Second pass ---
    # TODO, perhaps it is more robust to do 1 pass for each segment instead of attempting a single pass?
    ts2 = jnp.linspace(1e-9, params_model.length, params_ivp.number_of_points)
    term2 = dfx.ODETerm(ode_rhs_transonic)
    save2 = dfx.SaveAt(ts=ts2, t1=True, fn=ode_full_transonic)
    sol2 = dfx.diffeqsolve(
        term2,
        solver,
        t0=1e-9,
        t1=params_model.length,
        dt0=None,
        y0=y_inlet,
        args=params_model,
        saveat=save2,
        stepsize_controller=ctrl,
        adjoint=adjoint,
        max_steps=5_000,
    )

    return sol2


def _mach_event_cond(t, y, args, **kwargs):
    """Event function: zero when M^2 - Ma_low^2 = 0."""
    v, rho, p = y
    a = get_props(cpx.DmassP_INPUTS, rho, p, args.fluid)["a"]
    Ma_sqr = (v / a) ** 2
    return Ma_sqr - args.Ma_low**2


def _linearize_rhs_at(x_star, y_star, model):
    """Return Taylor expansion coefficients of RHS around (x_star, y_star)."""
    rhs_fn = lambda xx, yy: nozzle_single_phase_core(xx, yy, model)["rhs"]
    f_star = rhs_fn(x_star, y_star)  # RHS at expansion point
    Jy = jax.jacrev(lambda yy: rhs_fn(x_star, yy))(y_star)  # ∂f/∂y
    Jt = jax.jacrev(lambda xx: rhs_fn(xx, y_star))(x_star)  # ∂f/∂t
    return f_star, Jy, Jt


def _smoothstep(x):
    x = jnp.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)

    # return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


# -----------------------------------------------------------------------------
# Converging-diverging nozzle example
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    # Define model parameters
    fluid_name = "air"
    fluid = cpx.perfect_gas.get_constants(fluid_name, T_ref=300, P_ref=101325)

    params_model = NozzleParams(
        p0_in=1.0e5,  # Pa
        d0_in=1.20,  # kg/m³
        D_in=0.050,  # m
        length=5.00,  # m
        roughness=10e-6,  # m
        T_wall=300.0,  # K
        Ma_low=0.975,
        Ma_high=1.025,
        heat_transfer=0.0,
        wall_friction=0.0,
        fluid=fluid,
        geometry=symmetric_nozzle_geometry,
    )

    params_bvp = BVPSettings(
        solve_mode="mach_crit",
        num_points=60,
        rtol=1e-8,
        atol=1e-8,
        max_steps=500,
        jac_mode="bwd",
        verbose=False,
        method="GaussNewton",
        warmup_method="Dogleg",
        warmup_steps=10,
    )

    params_ivp = IVPSettings(
        solver_name="Dopri5",
        adjoint_name="DirectAdjoint",
        number_of_points=50,
        rtol=1e-8,
        atol=1e-8,
    )

    # Solve the problem
    print("\n" + "-" * 60)
    print("Evaluating transonic solution")
    print("-" * 60)
    input_array = jnp.asarray([2]) * 1e5
    colors = plt.cm.magma(jnp.linspace(0.2, 0.8, len(input_array)))  # Generate colors
    solution_list = []
    for i, p0_in in enumerate(input_array):
        t0 = time.perf_counter()
        params_model = replace_param(params_model, "p0_in", p0_in)
        sol = transonic_nozzle_single_phase(params_model, params_bvp, params_ivp)

        # Relative error diagnostics
        dt_ms = (time.perf_counter() - t0) * 1e3
        out = sol.ys
        mdot_error = (out["m_dot"].max() - out["m_dot"].min()) / out["m_dot"][0]
        h0_error = (out["h0"].max() - out["h0"].min()) / out["h0"][0]
        s_error = (out["s"].max() - out["s"].min()) / out["s"][0]

        print(
            f"Solution {i} | Solver status {sol.result._value:2d} | "
            f"steps {int(sol.stats['num_steps']):3d} | "
            f"mdot error {mdot_error:0.2e} | h0 error {h0_error:0.2e} | "
            f"s_error {s_error:0.2e} | time {dt_ms:7.2f} ms"
        )

        solution_list.append(sol)

    # Create the figure
    fig = plt.figure(figsize=(5, 10))
    gs = gridspec.GridSpec(5, 1, height_ratios=[3, 3, 3, 3, 1])
    xg = solution_list[0].ys["x"]
    rg = solution_list[0].ys["diameter"] / 2.0
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    ax3 = fig.add_subplot(gs[3], sharex=ax0)
    ax4 = fig.add_subplot(gs[4], sharex=ax0)
    axs = [ax0, ax1, ax2, ax3, ax4]

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
        axs[1].plot(
            sol.ys["x"],
            sol.ys["Ma"],
            color=color,
            marker="o",
            markersize="3",
        )

    # Static and stagnation enthalpy
    axs[2].set_ylabel("Enthalpy (J/kg)")
    for color, val, sol in zip(colors, input_array, solution_list):
        # axs[2].plot(sol.ys["x"], sol.ys["h"], color=r["color"], markersize=3, marker"o")
        axs[2].plot(sol.ys["x"], sol.ys["h0"], color=color, markersize=3, marker="o")

    # Entropy
    axs[3].set_ylabel("Entropy (J/kg/K)")
    for color, val, sol in zip(colors, input_array, solution_list):
        axs[3].plot(sol.ys["x"], sol.ys["s"], color=color, markersize=3, marker="o")

    # --- row 3: nozzle geometry ---
    axs[4].fill_between(xg, -rg, +rg, color="lightgrey")  # shaded nozzle
    x_closed = jnp.concatenate([xg, xg[::-1], xg[:1]])
    y_closed = jnp.concatenate([rg, -rg[::-1], rg[:1]])
    axs[4].plot(x_closed, y_closed, "k", linewidth=1.2)
    # r_abs_max = float(jnp.max(jnp.abs(rg)))
    # ax3.set_ylim(-1.5 * r_abs_max, 1.5 * r_abs_max)
    axs[4].set_aspect("equal", adjustable="box")
    axs[4].set_xlabel("Axial coordinate x (m)")
    fig.tight_layout(pad=1)

    # Show figures
    plt.show()

    # # --- Differentiability check: mdot vs. p0_in ---
    # def mdot_vs_p0(p0_in):
    #     num_points = 50
    #     local_params = replace_param(params_model, "p0_in", p0_in)
    #     sol = transonic_nozzle_single_phase(local_params, params_bvp, params_ivp, fluid)
    #     return sol.ys["m_dot"][0]

    # # Base point
    # p0_in = 101325.0

    # # JAX derivative
    # mdot_val = mdot_vs_p0(p0_in)
    # mdot_grad = jax.grad(mdot_vs_p0)(p0_in)

    # # Finite difference derivative
    # h = 1.0  # small perturbation in Pa
    # fd_grad = (mdot_vs_p0(p0_in + h) - mdot_vs_p0(p0_in - h)) / (2 * h)

    # # Print results
    # print()
    # print(f" mdot(p0_in={p0_in:.3f}) = {mdot_val:.6e}")
    # print(f" JAX   d(mdot)/d(p0_in) = {mdot_grad:.6e}")
    # print(f" FD    d(mdot)/d(p0_in) = {fd_grad:.6e}")
    # print(f" Relative diff = {abs((mdot_grad - fd_grad) / fd_grad):.3e}")
