# -----------------------------------------------------------------------------
# Overview
# -----------------------------------------------------------------------------
# This script implements a collocation-based solver for 1D quasi-1D nozzle flow
# using JAX, Optimistix, and Equinox. It supports two problem formulations:
#   1. "mach_in"    – solve with a prescribed inlet Mach number.
#   2. "mach_crit"  – solve with a prescribed maximum (critical) Mach number.
#
# Main components:
#
# 1. Data containers
#    - NozzleParams     : Stores all geometric, inlet, and target flow parameters.
#    - SolverSettings   : Stores numerical solver configuration (e.g., tolerances,
#                         solver type, collocation points, solve mode).
#    - ResidualParams   : Groups model, fluid, and discretization data for residuals.
#
# 2. Main solver
#    - solve_nozzle_model_collocation() :
#        Given an initial guess and problem parameters, builds the appropriate
#        residual function (based on solve_mode), sets up the nonlinear solver,
#        and returns the converged flowfield and solver statistics.
#
# 3. Residual functions
#    - build_residual_vector_mach_inlet()   : Residual formulation enforcing a
#                                             target inlet Mach number.
#    - build_residual_vector_mach_critical(): Residual formulation enforcing a
#                                             target maximum Mach number within
#                                             the domain, found via Newton search.
#
# 4. Flowfield utilities
#    - find_maximum_mach()    : Finds the location and value of the maximum Mach
#                               number in the domain from Chebyshev-Lobatto data.
#    - compute_static_state() : Computes static flow state from stagnation state
#                               and Mach number.
#    - split_z()              : Unpacks concatenated solution vector into velocity,
#                               log-density, and log-pressure arrays.
#    - evaluate_ode_rhs()     : Evaluates the nozzle right-hand-side model at all
#                               collocation points.
#    - initialize_flowfield() : Generates an initial guess for the solver using
#                               a parabolic Mach profile.
#
# 5. Chebyshev-Lobatto utilities
#    - chebyshev_lobatto_basis()                : Returns collocation nodes and
#                                                 differentiation matrix.
#    - chebyshev_lobatto_interpolate()          : Interpolates nodal data at given
#                                                 points (value only).
#    - chebyshev_lobatto_interpolate_and_derivative():
#                                                 Interpolates nodal data and
#                                                 returns derivative.
#
# 6. Helper
#    - replace_param() : Creates a copy of an Equinox module with one field replaced.
#
# Workflow:
#    (a) Define NozzleParams and SolverSettings.
#    (b) Generate an initial guess with initialize_flowfield().
#    (c) Call solve_nozzle_model_collocation() with chosen solve_mode.
#    (d) Post-process output data (e.g., interpolate for plotting).
# -----------------------------------------------------------------------------

from __future__ import annotations
import jax
import jax.numpy as jnp
import equinox as eqx
import optimistix as opx
import coolpropx as cpx

from typing import Any, Callable

jax.config.update("jax_enable_x64", True)

from coolpropx.perfect_gas import get_props
from nozzle_model_core import nozzle_single_phase_core


# shorthand factory for float64 arrays
def f64(value):
    return eqx.field(default_factory=lambda: jnp.array(value, dtype=jnp.float64))

class NozzleParams(eqx.Module):
    fluid: Any = eqx.field(static=True)
    geometry: Callable = eqx.field(static=True)
    p0_in: jnp.ndarray = f64(1.0e5)       # Pa
    d0_in: jnp.ndarray = f64(1.20)        # kg/m³
    D_in: jnp.ndarray = f64(0.050)        # m
    length: jnp.ndarray = f64(5.00)       # m
    roughness: jnp.ndarray = f64(1e-6)    # m
    T_wall: jnp.ndarray = f64(300.0)      # K
    Ma_in: jnp.ndarray = f64(0.1)
    Ma_low: jnp.ndarray = f64(0.95)
    Ma_high: jnp.ndarray = f64(1.05)
    Ma_target: jnp.ndarray = f64(1.0)
    heat_transfer: jnp.ndarray = f64(0.0)
    wall_friction: jnp.ndarray = f64(0.0)


class BVPSettings(eqx.Module):
    num_points: int = eqx.field(static=True)
    rtol: jnp.ndarray
    atol: jnp.ndarray
    max_steps: int = eqx.field(static=True)
    jac_mode: str = eqx.field(static=True)  # "bwd" or "fwd"
    verbose: bool = eqx.field(static=True)
    method: str = eqx.field(static=True, default="GaussNewton")
    warmup_method: str = eqx.field(static=True, default="LevenbergMarquardt")
    warmup_steps: int = eqx.field(static=True, default=0)
    solve_mode: str = eqx.field(static=True, default="mach_crit")  # or "mach_crit"

class IVPSettings(eqx.Module):
    """Settings for marching initial value problem solvers."""
    solver_name: str = eqx.field(static=True, default="Dopri5")
    adjoint_name: str = eqx.field(static=True, default="DirectAdjoint")
    number_of_points: int = eqx.field(static=True, default=50)
    rtol: float = 1e-6
    atol: float = 1e-6


class ResidualParams(eqx.Module):
    x: jnp.ndarray
    Dx: jnp.ndarray
    model: NozzleParams


def replace_param(obj, field, value):
    """Return a copy of obj with a single field replaced."""
    return eqx.tree_at(lambda o: getattr(o, field), obj, replace=jnp.asarray(value))


SOLVER_MAPPING = {
    "GaussNewton": opx.GaussNewton,
    "Dogleg": opx.Dogleg,
    "LevenbergMarquardt": opx.LevenbergMarquardt,
}


# ---------- Main function call ----------
@eqx.filter_jit
def solve_nozzle_model_collocation(
    initial_guess,
    params_model,
    params_solver,
):
    """
    Solves the collocation system using a warmup solver (e.g., Levenberg-Marquardt)
    followed by a main solver (e.g., Gauss-Newton).
    Returns the evaluated flowfield and the final solver result.
    """

    # Compute the Chebyshev basis (only once per call)
    x, D = chebyshev_lobatto_basis(params_solver.num_points, 0.0, params_model.length)

    # Build the function to compute the residual vector
    residual_args = ResidualParams(x=x, Dx=D, model=params_model)

    # Select correct residual function
    if params_solver.solve_mode == "mach_in":
        residual_fn = get_residual_mach_inlet
    elif params_solver.solve_mode == "mach_crit":
        residual_fn = get_residual_mach_critical
    else:
        raise ValueError(f"Unknown solve_mode: {params_solver.solve_mode}")

    # Configure the solvers
    def make_solver(name):
        return SOLVER_MAPPING[name](
            rtol=params_solver.rtol,
            atol=params_solver.atol,
            verbose=vset,
        )

    vars = {"step", "loss", "accepted", "step_size"}
    vset = frozenset(vars) if params_solver.verbose else frozenset()
    solver_warmup = make_solver(params_solver.warmup_method)
    solver_main   = make_solver(params_solver.method)

    # Use a robust solver for a fet iterations to warmup
    solution_warmup = opx.least_squares(
        fn=residual_fn,
        args=residual_args,
        y0=initial_guess,
        solver=solver_warmup,
        options={"jac": params_solver.jac_mode},
        max_steps=params_solver.warmup_steps,     
        throw=False,
    )

    # Solve the problem using the main solver
    solution = opx.least_squares(
        fn=residual_fn,
        args=residual_args,
        y0=solution_warmup.value,
        solver=solver_main,
        options={"jac": params_solver.jac_mode},
        max_steps=params_solver.max_steps,
    )

    # Evaluate the flowfield at the converged solution
    out_data = evaluate_ode_rhs(x, solution.value, params_model)

    return out_data, solution


# ---------- Create function handle for the residual vector ----------
def get_residual_mach_inlet(z, params: ResidualParams):

    # We solve for ln(d) and ln(p) instead of d and p directly.
    #   - This enforces strict positivity of density and pressure without explicit constraints.
    #   - The PDE in ln(d) or ln(p) form contains a 1/d or 1/p factor in the derivative term:
    #       d/dx[ln(d)] = (1/d) * d(d)/dx
    #       d/dx[ln(p)] = (1/p) * d(p)/dx
    #     so when forming the residuals, the nonlinear terms N_all[:,1] and N_all[:,2] must be divided
    #     by the current d and p values, respectively.
    #   - This scaling also normalizes the residual magnitude for variables with very different units
    #     and prevents density/pressure from dominating the Jacobian purely due to scale.

    # Unpack solution vector
    u, ln_d, ln_p = split_z(z, params.x.shape[0])
    d = jnp.exp(ln_d)
    p = jnp.exp(ln_p)

    # Compute right hand side of the ODE
    out = evaluate_ode_rhs(params.x, z, params.model)
    N_all = out["N"]
    D_tau = out["D"]

    # Evaluate residuals at collocation points
    # Multiply-through PDE residuals (no division by D)
    R_u = (params.Dx @ u) - N_all[:, 0] / D_tau
    R_d = (params.Dx @ ln_d) - N_all[:, 1] / D_tau / d
    R_p = (params.Dx @ ln_p) - N_all[:, 2] / D_tau / p

    # Evaluate residual at boundary condition
    R_u = R_u.at[0].set(params.model.Ma_in - out["Ma"][0])
    R_d = R_d.at[0].set(jnp.log(params.model.d0_in / out["d0"][0]))
    R_p = R_p.at[0].set(jnp.log(params.model.p0_in / out["p0"][0]))

    return jnp.concatenate([R_u, R_d, R_p])


# ---------- Create function handle for the residual vector ----------
def get_residual_mach_critical(z, params: ResidualParams):

    # We solve for ln(d) and ln(p) instead of d and p directly.
    #   - This enforces strict positivity of density and pressure without explicit constraints.
    #   - The PDE in ln(d) or ln(p) form contains a 1/d or 1/p factor in the derivative term:
    #       d/dx[ln(d)] = (1/d) * d(d)/dx
    #       d/dx[ln(p)] = (1/p) * d(p)/dx
    #     so when forming the residuals, the nonlinear terms N_all[:,1] and N_all[:,2] must be divided
    #     by the current d and p values, respectively.
    #   - This scaling also normalizes the residual magnitude for variables with very different units
    #     and prevents density/pressure from dominating the Jacobian purely due to scale.

    # Unpack solution vector
    u, ln_d, ln_p = split_z(z, params.x.shape[0])
    d = jnp.exp(ln_d)
    p = jnp.exp(ln_p)

    # Compute right hand side of the ODE
    out = evaluate_ode_rhs(params.x, z, params.model)
    N_all = out["N"]
    D_tau = out["D"]

    # Evaluate residuals at collocation points
    R_u = (params.Dx @ u) - N_all[:, 0] / D_tau
    R_d = (params.Dx @ ln_d) - N_all[:, 1] / D_tau / d
    R_p = (params.Dx @ ln_p) - N_all[:, 2] / D_tau / p

    # Evaluate residual at the boundary conditions
    x_star, Ma_max = find_maximum_mach(params.x, out["Ma"])
    R_u = R_u.at[0].set(params.model.Ma_target - Ma_max)
    R_d = R_d.at[0].set(jnp.log(params.model.d0_in / out["d0"][0]))
    R_p = R_p.at[0].set(jnp.log(params.model.p0_in / out["p0"][0]))

    return jnp.concatenate([R_u, R_d, R_p])



# ---------- Calculate maximum Mach number within the domain ----------


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
    solver = opx.Newton(rtol=rtol, atol=atol, cauchy_termination=False)
    sol = opx.root_find(resid, solver, y0=x0, args=None, max_steps=newton_steps, throw=False)
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

    return jax.vmap(per_node)(u, ln_d, ln_p, x)


# ---------- Generate flow field for initial guess ----------
def initialize_flowfield(num_points, params, Ma_min=0.1, Ma_max=0.2):
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
    fluid = params.fluid
    state0_in = get_props(cpx.DmassP_INPUTS, params.p0_in, params.d0_in, fluid)
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





# ---------- Compute static state from stagnation and Mach number ----------
def compute_static_state(p0, d0, Ma, fluid):
    """solve h0 - h(p,s0) - 0.5 a(p,s0)^2 Ma^2 = 0 for p"""
    st0 = get_props(cpx.DmassP_INPUTS, d0, p0, fluid)
    s0, h0 = st0["s"], st0["h"]

    def residual(p, _):
        st = get_props(cpx.PSmass_INPUTS, p, s0, fluid)
        a, h = st["a"], st["h"]
        v = a * Ma
        return h0 - h - 0.5 * v * v

    p_init = 0.9 * p0
    solver = opx.Newton(rtol=1e-10, atol=1e-10)
    sol = opx.root_find(residual, solver, y0=p_init, args=None)
    state = get_props(cpx.PSmass_INPUTS, sol.value, s0, fluid)
    return state