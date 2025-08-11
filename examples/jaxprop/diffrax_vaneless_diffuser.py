import os
import difflib
import jax
import jax.numpy as jnp
import diffrax as dfx
import equinox as eqx
import optimistix as optx
import coolpropx as cpx
import matplotlib.pyplot as plt

from time import perf_counter

cpx.set_plot_options(grid=False)


# -----------------------------------------------------------------------------
# Main API to the vaneless diffuser model
# -----------------------------------------------------------------------------
@eqx.filter_jit
def vaneless_diffuser(
    params,
    fluid,
    solver_name: str = "Dopri5",
    adjoint_name: str = "DirectAdjoint",
    number_of_points: int | None = None,
    rtol: float = 1e-6,
    atol: float = 1e-6,
):
    """Evaluate one-dimensional flow in an annular vaneless diffuser"""

    # Rename parameters
    p0_in = params["p0_in"]
    T0_in = params["T0_in"]
    Ma_in = params["Ma_in"]
    alpha_in = params["alpha_in"]
    Cf = params["Cf"]
    q_w = params["q_w"]
    r_in = params["r_in"]
    r_out = params["r_out"]
    b_in = params["b_in"]
    phi = params["phi"]
    div = params["div"]
    L = r_out - r_in

    # Compute initial conditions for ODE system
    p_in, s_in = compute_inlet_static_state(p0_in, T0_in, Ma_in, fluid)
    state = get_props(cpx.PSmass_INPUTS, p_in, s_in, fluid)
    d_in = state["rho"]
    a_in = state["a"]
    v_in = Ma_in * a_in
    v_m_in = v_in * jnp.cos(alpha_in)
    v_t_in = v_in * jnp.sin(alpha_in)
    y0 = jnp.array([v_m_in, v_t_in, d_in, p_in, 0.0, 0.0])

    # Group the ODE system constant parameters
    args = (Cf, q_w, r_in, b_in, phi, div, p0_in, p_in, fluid)

    # Create and configure the solver
    solver = cpx.jax_import.make_diffrax_solver(solver_name)
    adjoint = cpx.jax_import.make_diffrax_adjoint(adjoint_name)
    term = dfx.ODETerm(diffuser_odefun)
    ctrl = dfx.PIDController(rtol=rtol, atol=atol)
    if number_of_points is not None:
        ts = jnp.linspace(0.0, L, number_of_points)
        saveat = dfx.SaveAt(ts=ts, dense=False, fn=postprocess_ode)
    else:
        saveat = dfx.SaveAt(t1=True, fn=postprocess_ode)

    # Solve the ODE system
    sol = dfx.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=L,
        dt0=None,
        y0=y0,
        saveat=saveat,
        stepsize_controller=ctrl,
        args=args,
        max_steps=10_000,
        adjoint=adjoint,
    )

    return sol


# -----------------------------------------------------------------------------
# Right hand side of the diffuser ODE system
# -----------------------------------------------------------------------------


def diffuser_odefun(t, y, args):
    # Rename from ODE terminology to physical variables
    length = t
    v_m, v_t, d, p, s_gen, theta = y
    Cf, q_w, r_in, b_in, phi, div, p0_in, p_in, fluid = args

    # Calculate velocity
    v = jnp.sqrt(v_t**2 + v_m**2)
    alpha = jnp.arctan2(v_t, v_m)

    # Calculate local geometry
    r = radius_fun(r_in, phi, length)
    b = width_fun(b_in, div, length)
    dAdr = area_grad(length, b_in, div, r_in, phi)

    # Calculate thermodynamic state
    state = get_props(cpx.DmassP_INPUTS, d, p, fluid)
    a = state["a"]
    h = state["h"]
    s = state["s"]
    T = state["T"]
    G = state["gruneisen"]
    h0 = h + 0.5 * v**2

    # Stress at the wall
    tau_w = Cf * d * v**2 / 2

    # Compute coefficient matrix
    M = jnp.asarray(
        [
            [d, 0.0, v_m, 0.0],
            [d * v_m, 0.0, 0.0, 1.0],
            [0.0, d * v_m, 0.0, 0.0],
            [0.0, 0.0, -a**2, d],
        ]
    )

    # Compute source term
    S = jnp.asarray(
        [
            -d * v_m / (b * r) * dAdr,
            d * v_t**2 / r * jnp.sin(phi) - 2 * tau_w / b * jnp.cos(alpha),
            -d * v_t * v_m / r * jnp.sin(phi) - 2 * tau_w / b * jnp.sin(alpha),
            (2 / b) * (G / v_m) * (tau_w * v + q_w),
        ]
    )

    # Compute solution
    core = jnp.linalg.solve(M, S)  # dy = [dv_m, dv_t, d, p]
    sigma = 2.0 / b * (tau_w * v)
    s_gen_dot = sigma / (d * v_m) / T  # Entropy generation
    theta_dot = (v_t / v_m) / r  # Streamline wrapping angle (only for phi=pi/2)

    return jnp.concatenate([core, jnp.array([s_gen_dot, theta_dot])])


def postprocess_ode(t, y, args):
    """compute derived outputs at save times"""
    Cf, q_w, r_in, b_in, phi, div, p0_in, p_in, fluid = args
    v_m, v_t, d, p, s_gen, theta = y
    v = jnp.sqrt(v_t**2 + v_m**2)
    state = get_props(cpx.DmassP_INPUTS, d, p, fluid)
    a = state["a"]
    r = radius_fun(r_in, phi, t)
    b = width_fun(b_in, div, t)
    out = {
        "v_t": v_t,
        "v_m": v_m,
        "v": v,
        "Ma": v / a,
        "Ma_m": v_m / a,
        "Ma_t": v_t / a,
        "alpha": jnp.arctan2(v_t, v_m),
        "d": d,
        "p": p,
        "s": state["s"],
        "s_gen": s_gen,
        "h": state["h"],
        "h0": state["h"] + 0.5 * v**2,
        "theta": theta,
        "r": r,
        "b": b,
        "m": t,
        "radius_ratio": r / r_in,
        "area_ratio": (b * r) / (b_in * r_in),
        "Cp": (p - p_in) / (p0_in - p_in),
    }
    return out


def radius_fun(r_in, phi, m):
    """Calculate the radius from the meridional coordinate"""
    return r_in + jnp.sin(phi) * m


def width_fun(b_in, div, m):
    """Calculate the channel width from the meridional coordinate"""
    return b_in + 2 * jnp.tan(div) * m


def area_fun(m, b_in, div, r_in, phi):
    return width_fun(b_in, div, m) * radius_fun(r_in, phi, m) # Does not include the 2*pi


area_grad = jax.grad(area_fun, argnums=0)


# -----------------------------------------------------------------------------
# Compute the inlet state with a differentiable optimistix root finder
# -----------------------------------------------------------------------------


def compute_inlet_static_state(p0, T0, Ma, fluid):
    """
    Calculate the static pressure from stagnation conditions and Mach number using
    a pure-JAX Newton solver from Optimistix.

    Parameters
    ----------
    p0 : float
        Stagnation pressure (Pa).
    T0 : float
        Stagnation temperature (K).
    Ma : float
        Mach number.
    fluid : dict
        CoolProp-compatible fluid constants (already queried, JAX array-friendly).

    Returns
    -------
    p_static : float
        Static pressure (Pa).
    s0 : float
        Stagnation entropy (J/kg/K).
    """

    # Compute stagnation state properties
    state0 = get_props(cpx.PT_INPUTS, p0, T0, fluid)
    s0, h0 = state0["s"], state0["h"]

    # Residual function for root find
    def residual(p, _):
        # f(p) = h0 - h(p,s0) - 0.5 a(p,s0)^2 Ma^2
        state = get_props(cpx.PSmass_INPUTS, p, s0, fluid)
        a, h = state["a"], state["h"]
        v = a * Ma
        return h0 - h - 0.5 * v * v

    # Solve the equation
    p_init = 0.9 * p0
    solver = optx.Newton(rtol=1e-10, atol=1e-10)
    sol = optx.root_find(
        residual,
        solver,
        y0=p_init,
        args=None,
    )
    return sol.value, s0


# -----------------------------------------------------------------------------
# example usage / plotting
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    # Define model parameters
    fluid_name = "air"
    params = {
        "p0_in": 101325.0,
        "T0_in": 273.15 + 20.0,
        "Ma_in": 0.75,
        "alpha_in": 65 * jnp.pi / 180,
        "Cf": 0.0,
        "q_w": 0.0,
        "r_in": 1.0,
        "r_out": 3.0,
        "b_in": 0.25,
        "phi": 90 * jnp.pi / 180,  # pi/2 for radial channel
        "div": 0 * jnp.pi / 180,  # 0 for constant width channel
    }

    # Convert to JAX types
    params = {k: jnp.asarray(v) for k, v in params.items()}

    # Initialize fluid depending on backend
    backend = "perfect_gas"
    # backend = "jaxprop"
    x_prop, y_prop = "s", "T"
    if backend == "perfect_gas":
        from coolpropx.perfect_gas import get_props, get_perfect_gas_constants
        fluid = get_perfect_gas_constants(fluid_name, params["T0_in"], params["p0_in"])

    elif backend == "jaxprop":
        from coolpropx.jaxprop import get_props
        fluid = cpx.Fluid(name=fluid_name, backend="HEOS")

    else:
        raise ValueError("Invalid fluid backend seclection")
    

    # Plot the pressure recovery coefficient distribution
    fig_1, ax_1 = plt.subplots(figsize=(6, 5))
    ax_1.grid(True)
    ax_1.set_xlabel("Radius ratio")
    ax_1.set_ylabel("Pressure recovery coefficient\n")

    # Plot the Mach number distribution
    fig_2, ax_2 = plt.subplots()
    ax_2.grid(True)
    ax_2.set_xlabel("Radius ratio")
    ax_2.set_ylabel("Mach number\n")

    # Plot streamlines
    number_of_streamlines = 5
    fig_3, ax_3 = plt.subplots()
    ax_3.set_aspect("equal", adjustable="box")
    ax_3.grid(False)
    ax_3.set_xlabel("x coordinate")
    ax_3.set_ylabel("y coordinate\n")
    ax_3.set_title("Diffuser streamlines\n")
    ax_3.axis(1.1 * params["r_out"] * jnp.array([-1, 1, -1, 1]))
    theta = jnp.linspace(0, 2 * jnp.pi, 100)
    x_in = params["r_in"] * jnp.cos(theta)
    y_in = params["r_in"] * jnp.sin(theta)
    x_out = params["r_out"] * jnp.cos(theta)
    y_out = params["r_out"] * jnp.sin(theta)
    ax_3.plot(x_in, y_in, "k", label=None)  # HandleVisibility='off'
    ax_3.plot(x_out, y_out, "k", label=None)  # HandleVisibility='off'
    theta = jnp.linspace(0, 2 * jnp.pi, number_of_streamlines + 1)

    # Compute diffuser performance for different friction factors
    Cf_array = jnp.asarray([0.0, 0.01, 0.02, 0.03])
    colors = plt.cm.magma(jnp.linspace(0.2, 0.8, len(Cf_array)))  # Generate colors
    print()
    print("-" * 42)
    print("Friction factor sensitivity analysis")
    print("-" * 42)
    for i, Cf in enumerate(Cf_array):
        params["Cf"] = Cf
        t0 = perf_counter()
        out = vaneless_diffuser(params, fluid, number_of_points=100)
        print(f"Call {i+1}: Model evaluation time: {(perf_counter()-t0)*1e3:.4f} ms")

        # Plot the pressure recovery coefficient distribution
        ax_1.plot(
            out.ys["radius_ratio"],
            out.ys["Cp"],
            label=f"$C_f = {Cf:0.3f}$",
            color=colors[i],
        )
        ax_1.legend(loc="lower right")

        # Plot the Mach number distribution
        ax_2.plot(
            out.ys["radius_ratio"],
            out.ys["Ma"],
            label=f"$C_f = {Cf:0.3f}$",
            color=colors[i],
        )
        ax_2.legend(loc="upper right")

        # Plot streamlines
        for j in range(len(theta)):
            x = out.ys["r"] * jnp.cos(out.ys["theta"] + theta[j])
            y = out.ys["r"] * jnp.sin(out.ys["theta"] + theta[j])
            if j == 0:
                ax_3.plot(x, y, label=f"$C_f = {Cf:0.3f}$", color=colors[i])
            else:
                ax_3.plot(x, y, color=colors[i])

    # Adjust pad
    for fig in [fig_1, fig_2, fig_3]:
        fig.tight_layout(pad=1)

    # --------------------------------------------------------------------- #
    # ----------  Compute the gradients of an objective function ---------- #
    # --------------------------------------------------------------------- #
    def objective(params, fluid):
        sol = vaneless_diffuser(
            params,
            fluid,
            number_of_points=None,
            solver_name="Dopri5",
            adjoint_name="DirectAdjoint",
        )
        return jnp.squeeze(sol.ys["Cp"])  # scalar


    # JIT the scalar oand gradients
    objective_jit = eqx.filter_jit(objective)
    jac_fwd = eqx.filter_jit(jax.jacfwd(objective))
    jac_rev = eqx.filter_jit(jax.jacrev(objective))
    hess_fn = eqx.filter_jit(jax.jacfwd(jax.jacrev(objective)))

    # Print the results
    print()
    print("-" * 42)
    print("Function timings (Cp only):")
    print("-" * 42)
    for i in range(5):
        t0 = perf_counter()
        v = objective_jit(params, fluid)
        print(f"Call {i+1}: {(perf_counter()-t0)*1e3:.4f} ms")

    print()
    print("-" * 42)
    print("Gradient timings (forward-mode, Cp only):")
    print("-" * 42)
    for i in range(5):
        t0 = perf_counter()
        g_ad = jac_fwd(params, fluid)
        print(f"Call {i+1}: {(perf_counter()-t0)*1e3:.4f} ms")

    print()
    print("-" * 42)
    print("Gradient timings (reverse-mode, Cp only):")
    print("-" * 42)
    for i in range(5):
        t0 = perf_counter()
        g_ad = jac_rev(params, fluid)
        print(f"Call {i+1}: {(perf_counter()-t0)*1e3:.4f} ms")

    print()
    print("-" * 42)
    print("Gradient of Cp_out w.r.t. parameters")
    print("-" * 42)
    rel_eps=1e-6
    for k in g_ad.keys():
        base = params[k]
        eps = rel_eps * (jnp.abs(base) + 1.0)
        p_plus  = dict(params); p_plus[k]  = base + eps
        p_minus = dict(params); p_minus[k] = base - eps
        f_plus  = objective(p_plus,  fluid)
        f_minus = objective(p_minus, fluid)
        g_fd = (f_plus - f_minus) / (2.0 * eps)
        err_abs = jnp.abs(g_ad[k]-g_fd)
        err_rel = err_abs/(g_ad[k]+1e-16)
        print(f" {k:<10}  AD: {g_ad[k]: .6e}   FD: {g_fd: .6e}   abs.err: {err_abs: .3e}   rel.err: {err_rel: .3e}")


    if backend == "perfect_gas":

        print("\n------------------------------------------")
        print("Hessian timings (forward-over-reverse):")
        print("------------------------------------------")
        for i in range(5):
            t0 = perf_counter()
            H = hess_fn(params, fluid)
            print(f"Call {i+1}: {(perf_counter()-t0)*1e3:.4f} ms")


        def print_hessian(H):
            names = list(H.keys())
            # header
            header = " " * 11 + "".join(f"{n:>12s}" for n in names)
            print(header)
            # rows
            for r in names:
                row = "".join(f"{H[r][c]:+12.3e}" for c in names)
                print(f"{r:>10s} {row}")

        # example
        print("\nHessian matrix:")
        print_hessian(H)


    # Show plots
    plt.show()
