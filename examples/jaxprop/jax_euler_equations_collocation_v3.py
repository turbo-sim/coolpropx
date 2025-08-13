from __future__ import annotations
import jax
import jax.tree_util
import jax.numpy as jnp
import optimistix as opx
import matplotlib.pyplot as plt
import equinox as eqx
import coolpropx as cpx

from time import perf_counter

jax.config.update("jax_enable_x64", True)

cpx.set_plot_options()

# Import specific functions
from coolpropx.perfect_gas import get_props
from examples.jaxprop.nozzle_model_core import nozzle_single_phase_core


# This code works like a charm.
# It uses the collocation method calling the nozzle_single_phase right hand side
# The problem is solved in [u, ln(p), ln(d)] to improve stability
# The solution is, of course, a bit sensitive to the initial guess provided
# The boundary conditions given to the model are [p0_in, d0_in, Ma_in]
# The are converted into the corresponding values of [u, ln(p), ln(d)] at the beginning of the calculations in order to impose the latter in the BC residuals

# v3 This function calculates the maximum with a Newton root finder initialized with softwmax
# It converges like a charm for decent number of nordes
# Using continuation, it can converge to extremely tight Mach numbers with just a couple of steps


# ---------- Main function call ----------
# @eqx.filter_jit
def solve_nozzle_model_collocation(
    z0,
    params,
    fluid,
    wall_friction=False,
    heat_transfer=False,
    num_points=50,
    rtol=1e-8,
    atol=1e-8,
    max_steps=200,
    jac_mode="bwd",
    verbose=False,
):

    # Group up the fixed parameters
    length = params["length"]
    eps_wall = params["roughness"]
    T_wall = params["T_wall"]
    ode_args = (length, eps_wall, T_wall, wall_friction, heat_transfer, fluid)

    # Compute the inlet boundary condition iteratively
    boundary_conditions = (params["p0_in"], params["d0_in"], params["Ma_target"])

    # Compute the Chebyshev basis” (only once per call)
    x, D = chebyshev_lobatto_basis(num_points, 0.0, params["length"])

    # Build the residual vector function
    residual_fn = build_residual_vector(D, x)

    # Configure the solver and solve the problem
    vars = {"step", "loss", "accepted", "step_size"}
    vset = frozenset(vars) if verbose else frozenset()
    solver = opx.GaussNewton(rtol=rtol, atol=atol, verbose=vset)
    # solver = opx.Dogleg(rtol=rtol, atol=atol, verbose=vset)
    # solver = opx.LevenbergMarquardt(rtol=rtol, atol=atol, verbose=vset)
    solution = opx.least_squares(
        residual_fn,
        solver,
        z0,
        args=(ode_args, boundary_conditions),
        options={"jac": jac_mode},  # "bwd" or "fwd"
        max_steps=max_steps,
    )

    # Evaluate the flowfield at the converged solution
    out_data = evaluate_ode_rhs(x, solution.value, ode_args)

    return out_data, solution


# ---------- Create function handle for the residual vector ----------
def build_residual_vector(Dx, x):

    # We solve for ln(d) and ln(p) instead of d and p directly.
    #   - This enforces strict positivity of density and pressure without explicit constraints.
    #   - The PDE in ln(d) or ln(p) form contains a 1/d or 1/p factor in the derivative term:
    #       d/dx[ln(d)] = (1/d) * d(d)/dx
    #       d/dx[ln(p)] = (1/p) * d(p)/dx
    #     so when forming the residuals, the nonlinear terms N_all[:,1] and N_all[:,2] must be divided
    #     by the current d and p values, respectively.
    #   - This scaling also normalizes the residual magnitude for variables with very different units
    #     and prevents density/pressure from dominating the Jacobian purely due to scale.

    def residual(z, parameters):
        # Unpack parameters
        args, boundary_conditions = parameters
        p0_in, d0_in, Ma_target = boundary_conditions

        # Unpack solution vector
        u, ln_d, ln_p = split_z(z, x.shape[0])
        d = jnp.exp(ln_d)
        p = jnp.exp(ln_p)

        # Compute right hand side of the ODE
        out = evaluate_ode_rhs(x, z, args)
        N_all = out["N"]
        D_tau = out["D"]

        # Evaluate residuals at collocation points
        R_u = (Dx @ u) - N_all[:, 0] / D_tau
        R_d = (Dx @ ln_d) - N_all[:, 1] / D_tau / d
        R_p = (Dx @ ln_p) - N_all[:, 2] / D_tau / p

        # Evaluate residual at the boundary conditions
        x_star, Ma_max = find_maximum_mach(x, out["Ma"])
        R_u = R_u.at[0].set(Ma_target - Ma_max)
        R_d = R_d.at[0].set(jnp.log(d0_in / out["d0"][0]))
        R_p = R_p.at[0].set(jnp.log(p0_in / out["p0"][0]))

        return jnp.concatenate([R_u, R_d, R_p])

    return residual


def find_maximum_mach(x_nodes, y_nodes, newton_steps=50, rtol=1e-10, atol=1e-10):
    """
    Locate the single interior maximum of the Chebyshev-Lobatto interpolant.

    Uses a smooth soft-argmax to pick an initial guess (avoids non-diff argmax),
    then runs a fixed number of Newton iterations on p'(x) = 0.

    Parameters
    ----------
    x_nodes : (N+1,) array
        Chebyshev-Lobatto nodes in the domain.
    y_nodes : (N+1,) array
        Function values at the nodes (e.g., Mach number).
    newton_steps : int, optional
        Maximum Newton iterations. Default 50.
    rtol, atol : float, optional
        Relative and absolute tolerances for Newton.

    Returns
    -------
    x_star : float
        Location of the maximum in [x1, x2].
    p_star : float
        Value of the interpolant at x_star.
    """
    x1, x2 = x_nodes[0], x_nodes[-1]

    # Smooth initial guess: soft-argmax over node values
    alpha = 50.0  # higher → sharper, closer to discrete argmax
    y_shift = y_nodes - jnp.max(y_nodes)
    w = jax.nn.softmax(alpha * y_shift)
    x0 = jnp.sum(w * x_nodes)

    # Residual for p'(x) = 0
    def resid(x, _):
        _, dp = chebyshev_lobatto_interpolate_and_derivative(x_nodes, y_nodes, x)
        return dp

    # Newton solve (ignore success flag, fixed iteration count)
    solver = opx.Newton(rtol=rtol, atol=atol)
    sol = opx.root_find(resid, solver, y0=x0, args=None, max_steps=newton_steps)
    x_star = sol.value

    # Clip to domain and guard against NaN fallback
    x_star = jnp.clip(jnp.nan_to_num(x_star, nan=x0), x1, x2)

    # Value at maximum
    p_star, _ = chebyshev_lobatto_interpolate_and_derivative(x_nodes, y_nodes, x_star)
    return x_star, p_star



# ---------- helpers: pack/unpack and per-node wrapper ----------
def split_z(z, num_points):
    u = z[0:num_points]
    ln_d = z[num_points : 2 * num_points]
    ln_p = z[2 * num_points : 3 * num_points]
    return u, ln_d, ln_p


def evaluate_ode_rhs(x, z, args):
    """Vectorized full-model eval at all nodes from z=[u, ln(rho), ln(p)]."""
    u, ln_d, ln_p = split_z(z, x.shape[0])

    def per_node(ui, ln_di, ln_pi, xi):
        di = jnp.exp(ln_di)
        pi = jnp.exp(ln_pi)
        yi = jnp.array([ui, di, pi])
        return nozzle_single_phase_core(xi, yi, args)

    return jax.vmap(per_node, in_axes=(0, 0, 0, 0))(u, ln_d, ln_p, x)


# ---------- Generate flow field for initial guess ----------
def initialize_flowfield(num_points, params, fluid, Ma_min=0.1, Ma_max=0.2):
    """
    Generate an initial guess for the flowfield using a concave Mach number profile.

    The Mach profile is defined as a symmetric parabola with its maximum (Ma_max)
    at the domain midpoint and its minimum (Ma_min) at both inlet and outlet.
    The corresponding velocity, density, and pressure fields are computed
    from the specified inlet stagnation state.

    Parameters
    ----------
    num_points : int
        Number of interior collocation points. The Chebyshev-Lobatto grid will
        contain num_points + 1 points in total.
    params : dict
        Must contain:
            p0_in : float
                Inlet stagnation pressure.
            d0_in : float
                Inlet stagnation density.
    fluid : object
        Fluid property accessor compatible with get_props().
    Ma_min : float, optional
        Minimum Mach number at the inlet and outlet. Default is 0.1.
    Ma_max : float, optional
        Maximum Mach number at the domain midpoint. Default is 0.5.

    Returns
    -------
    z0 : ndarray, shape (3*(num_points+1),)
        Initial guess vector at collocation points, concatenated as:
        [velocity, ln_density, ln_pressure].
    """
    # Inlet stagnation state
    state0_in = get_props(cpx.DmassP_INPUTS, params["p0_in"], params["d0_in"], fluid)
    a_in = state0_in["a"]  # use inlet speed of sound for initial guess everywhere
    h_in = state0_in["h"]
    s_in = state0_in["s"]

    # Create coordinate array from 0 to 1 (Chebyshev–Lobatto nodes not needed for init)
    x_uniform = jnp.linspace(0.0, 1.0, num_points + 1)

    # Parabolic Mach profile: peak at x=0.5, symmetric, concave
    # Parabola passing through (0, M_min), (0.5, M_max), (1, M_min)
    mach_profile = Ma_min + (Ma_max - Ma_min) * (1.0 - 4.0 * (x_uniform - 0.5) ** 2)

    # Velocity from Mach (constant a_in for initial guess)
    flowfield_v = mach_profile * a_in

    # Static density/pressure from h0 = h + v^2/2, s = s_in
    h_static = h_in - 0.5 * flowfield_v**2
    state_static = get_props(cpx.HmassSmass_INPUTS, h_static, s_in, fluid)
    d_static = jnp.maximum(state_static["rho"], 1e-12)
    p_static = jnp.maximum(state_static["p"], 1e-12)

    # Log variables
    flowfield_ln_d = jnp.log(d_static)
    flowfield_ln_p = jnp.log(p_static)

    # Concatenate into initial guess vector
    return jnp.concatenate([flowfield_v, flowfield_ln_d, flowfield_ln_p])


# ---------- Define Chebyshev-Lobatto nodes and differentiation matrix ----------
def chebyshev_lobatto_basis(N: int, x1: float, x2: float):
    """
    Return:
      x : (N+1,) physical nodes in [x1, x2]
      D : (N+1, N+1) differentiation s.t. (u_x)(x_i) ≈ sum_j D[i,j] * u(x_j)

    Built from Trefethen's formula. D acts on nodal values to produce nodal derivatives.
    """

    # Standard Trefethen ordering
    k = jnp.arange(N + 1)
    x_hat = jnp.cos(jnp.pi * k / N)
    x = 0.5 * (x_hat + 1.0) * (x2 - x1) + x1

    c = jnp.where((k == 0) | (k == N), 2.0, 1.0) * ((-1.0) ** k)
    X = jnp.tile(x_hat, (N + 1, 1))
    dX = X - X.T + jnp.eye(N + 1)
    C = jnp.outer(c, 1.0 / c)
    D_hat = C / dX
    D_hat = D_hat - jnp.diag(jnp.sum(D_hat, axis=1))

    # Scale derivative from [-1,1] to [0,L]
    D = -(2.0 / (x2 - x1)) * D_hat

    # Reorder so x[0] = 0, x[-1] = L
    idx = jnp.arange(N + 1)[::-1]
    x = x[idx]
    D = D[idx][:, idx]

    return x, D


def chebyshev_lobatto_interpolate(x_nodes, y_nodes, x_eval):
    """
    Evaluate the Chebyshev-Lobatto barycentric interpolant at one or more points.

    This function is a thin wrapper around `chebyshev_lobatto_interpolate_and_derivative`
    that discards the derivative and returns only the interpolated value.

    Parameters
    ----------
    x_nodes : array_like, shape (N+1,)
        The Chebyshev-Lobatto nodes in the physical domain [x_min, x_max].
    y_nodes : array_like, shape (N+1,)
        Function values at the Chebyshev-Lobatto nodes.
    x_eval : float or array_like
        Point(s) in the domain where the interpolant should be evaluated.

    Returns
    -------
    p : float or ndarray
        Interpolated value(s) p(x_eval).
    """
    p, _ = chebyshev_lobatto_interpolate_and_derivative(x_nodes, y_nodes, x_eval)
    return p


def chebyshev_lobatto_interpolate_and_derivative(x_nodes, y_nodes, x_eval):
    """
    Evaluate the Chebyshev-Lobatto barycentric interpolant and its derivative.

    Parameters
    ----------
    x_nodes : array_like, shape (N+1,)
        The Chebyshev-Lobatto nodes in the physical domain [x_min, x_max].
    y_nodes : array_like, shape (N+1,)
        Function values at the Chebyshev-Lobatto nodes.
    x_eval : float or array_like
        Point(s) in the domain where the interpolant and its derivative
        should be evaluated.

    Returns
    -------
    p : float or ndarray
        Interpolated value(s) p(x_eval).
    dp : float or ndarray
        First derivative p'(x_eval) with respect to x.

    Notes
    -----
    - Uses the barycentric interpolation formula, which is numerically stable
      even for high-degree polynomials.
    - Correctly handles the case where x_eval coincides with one of the nodes,
      returning the exact nodal value and the exact derivative at that node.
    - Works for scalar or vector x_eval.
    """
    n = x_nodes.size - 1
    k = jnp.arange(n + 1)
    w = jnp.where((k == 0) | (k == n), 0.5, 1.0) * ((-1.0) ** k)

    def _scalar_interp_and_deriv(x):
        diff = x - x_nodes
        is_node = diff == 0.0

        def at_node():
            # Build terms excluding idx manually
            idx = jnp.argmax(is_node).astype(int)
            p = y_nodes[idx]

            # diff and weights excluding idx
            diff_ex = x_nodes[idx] - x_nodes
            ydiff_ex = y_nodes[idx] - y_nodes

            # Set excluded self-term to 0 safely
            diff_ex = diff_ex.at[idx].set(1.0)  # avoid division by 0
            ydiff_ex = ydiff_ex.at[idx].set(0.0)
            w_ex = w.at[idx].set(0.0)

            dp = jnp.sum((w_ex / w[idx]) * (ydiff_ex) / (diff_ex))
            return p, dp

        def generic():
            r = w / diff
            S = jnp.sum(r)
            N = jnp.sum(r * y_nodes)
            p = N / S
            rp = -w / (diff * diff)
            S1 = jnp.sum(rp)
            N1 = jnp.sum(rp * y_nodes)
            dp = (N1 - p * S1) / S
            return p, dp

        return jax.lax.cond(jnp.any(is_node), at_node, generic)

    if jnp.ndim(x_eval) == 0:
        return _scalar_interp_and_deriv(x_eval)
    else:
        return jax.vmap(_scalar_interp_and_deriv)(x_eval)


# ---------- example ----------
if __name__ == "__main__":

    solve_nozzle_model_collocation = eqx.filter_jit(solve_nozzle_model_collocation)

    # Define model parameters
    backend = "perfect_gas"
    # backend = "jaxprop"
    fluid_name = "air"

    params = jax.tree_util.tree_map(
        jnp.asarray,
        {
            "Ma_target": 0.5,    # -
            "p0_in": 1.0e5,      # Pa
            "d0_in": 1.20,       # kg/m3
            "D_in": 0.050,       # m
            "length": 5.00,      # m
            "roughness": 10e-6,  # m
            "T_wall": 300.0,     # K
        },
    )

    # Numerical settings
    num_points = 50
    tolerance = 1e-8
    max_steps = 500
    jac_mode = "bwd"
    verbose = False

    # Define working fluid depending on backend selected
    if backend == "perfect_gas":
        from coolpropx.perfect_gas import get_props, get_perfect_gas_constants

        fluid = get_perfect_gas_constants(fluid_name, T_ref=300, P_ref=101325)

    elif backend == "jaxprop":
        from coolpropx.jaxprop import get_props

        fluid = cpx.Fluid(name=fluid_name, backend="HEOS")

    else:
        raise ValueError("Invalid fluid backend seclection")

    # Inlet Mach number sensitivity analysis
    print("\n" + "-" * 60)
    print("Running pressure ratio sweep (collocation)")
    print("-" * 60)
    # Ma_array = jnp.asarray(jnp.linspace(0.5, 0.7, 3))
    Ma_array = jnp.asarray([0.5, 0.9, 0.99, 0.999])
    colors = plt.cm.magma(jnp.linspace(0.2, 0.8, len(Ma_array)))
    z0 = initialize_flowfield(num_points, params, fluid)
    results = []
    for Ma, color in zip(Ma_array, colors):
        t0 = perf_counter()
        params = jax.tree_util.tree_map(jnp.asarray, {**params, "Ma_target": Ma})
        out, sol = solve_nozzle_model_collocation(
            z0,
            params,
            fluid,
            wall_friction=False,
            heat_transfer=False,
            num_points=num_points,
            max_steps=max_steps,
            jac_mode=jac_mode,
            verbose=verbose,
            rtol=tolerance,
            atol=tolerance,
        )

        # Substitute values, keep same array
        z0 = z0.at[:].set(sol.value)

        # Relative error diagnostics
        dt_ms = (perf_counter() - t0) * 1e3
        mdot_error = (out["m_dot"].max() - out["m_dot"].min()) / out["m_dot"][0]
        h0_error = (out["h0"].max() - out["h0"].min()) / out["h0"][0]
        s_error = (out["s"].max() - out["s"].min()) / out["s"][0]

        print(
            f"Ma_target = {Ma:0.4f} | Ma_crit = {out['Ma'][0]:0.5f} | Solver status {sol.result._value:2d} | "
            f"steps {int(sol.stats['num_steps']):3d} | "
            f"mdot error {mdot_error:0.2e} | h0 error {h0_error:0.2e} | "
            f"s_error {s_error:0.2e} | time {dt_ms:7.2f} ms"
        )

        results.append({"Ma": Ma, "color": color, "out": out, "sol": sol})

    # --- Plot the solutions ---
    fig, axs = plt.subplots(4, 1, figsize=(5, 9), sharex=True)
    x_dense = jnp.linspace(0.0, params["length"], 1000)

    # Pressure (bar)
    axs[0].set_ylabel("Pressure (bar)")
    for r in results:
        out = r["out"]
        x_nodes = out["x"]
        p_nodes = out["p"] * 1e-5
        p_dense = chebyshev_lobatto_interpolate(x_nodes, p_nodes, x_dense)
        axs[0].plot(x_dense, p_dense, color=r["color"])
        axs[0].plot(
            x_nodes,
            p_nodes,
            "o",
            color=r["color"],
            markersize=3,
            label=f"$Ma^*={r["Ma"]}$",
        )
    axs[0].legend(loc="lower right", fontsize=8)

    # Mach number
    axs[1].set_ylabel("Mach number (-)")
    for r in results:
        out = r["out"]
        Ma_nodes = out["Ma"]
        Ma_dense = chebyshev_lobatto_interpolate(out["x"], Ma_nodes, x_dense)
        axs[1].plot(x_dense, Ma_dense, color=r["color"])
        axs[1].plot(out["x"], Ma_nodes, "o", color=r["color"], markersize=3)

    # Static and stagnation enthalpy
    axs[2].set_ylabel("Enthalpy (J/kg)")
    for r in results:
        out = r["out"]
        h_dense = chebyshev_lobatto_interpolate(out["x"], out["h"], x_dense)
        h0_dense = chebyshev_lobatto_interpolate(out["x"], out["h0"], x_dense)
        axs[2].plot(x_dense, h_dense, color=r["color"], linestyle="-")
        axs[2].plot(out["x"], out["h"], "o", color=r["color"], markersize=3)
        axs[2].plot(x_dense, h0_dense, color=r["color"], linestyle="--")
        axs[2].plot(out["x"], out["h0"], "o", color=r["color"], markersize=3)

    # Entropy
    axs[3].set_ylabel("Entropy (J/kg/K)")
    for r in results:
        out = r["out"]
        s_dense = chebyshev_lobatto_interpolate(out["x"], out["s"], x_dense)
        axs[3].plot(x_dense, s_dense, color=r["color"])
        axs[3].plot(out["x"], out["s"], "o", color=r["color"], markersize=3)

    axs[3].set_xlabel("x (m)")
    fig.tight_layout(pad=1)
    plt.show()

    # # S metric
    # axs[3].set_ylabel("S metric (-)")
    # for r in results:
    #     out = r["out"]
    #     axs[3].plot(out["x"], out["s_metric"], color=r["color"])
    # axs[3].set_xlabel("x (m)")

    fig.tight_layout(pad=1)
    plt.show()
