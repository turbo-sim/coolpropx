from __future__ import annotations
import jax
import jax.numpy as jnp
import optimistix as opx
import matplotlib.pyplot as plt
from jax.tree_util import tree_map
import equinox as eqx
import coolpropx as cpx

from time import perf_counter

jax.config.update("jax_enable_x64", True)


cpx.set_plot_options()
from diffrax_nozzle_single_phase_v3 import nozzle_single_phase_core


# This code works like a charm.
# It uses the collocation method calling the nozzle_single_phase right hand side
# The problem is solved in [u, ln(p), ln(d)] to improve stability
# The solution is, of course, a bit sensitive to the initial guess provided
# The boundary conditions given to the model are [p0_in, d0_in, Ma_in]
# The are converted into the corresponding values of [u, ln(p), ln(d)] at the beginning of the calculations in order to impose the latter in the BC residuals



# ---------- Main function call ----------
@eqx.filter_jit
def solve_nozzle_model(
    params,
    fluid,
    wall_friction=False,
    heat_transfer=False,
    initial_guess=None,
    num_points=50,
    rtol=1e-10,
    atol=1e-10,
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
    Ma_in = params["Ma_in"]
    p0_in = params["p0_in"]
    p0_in = params["p0_in"]
    d0_in = params["d0_in"]
    state_in = compute_static_state(p0_in, d0_in, Ma_in, fluid)
    inlet_bc = (Ma_in * state_in["a"], jnp.log(state_in["d"]), jnp.log(state_in["p"]))

    # Compute the Chebyshev basis” (only once per call)
    x, D = chebyshev_lobatto_basis(num_points, 0.0, params["length"])

    # Initialize constant flow field over x if not provided
    if initial_guess is None:
        z0 = initialize_flowfield(num_points, params, fluid)
    else:
        z0 = initial_guess

    # Build the residual vector function
    residual_fn = build_residual_vector(D, x)

    # Configure the solver and solve the problem
    vars = {"step", "loss", "accepted", "step_size"}
    vset = frozenset(vars) if verbose else frozenset()
    # solver = opx.GaussNewton(rtol=rtol, atol=atol, verbose=vset)
    solver = opx.Dogleg(rtol=rtol, atol=atol, verbose=vset)
    solution = opx.least_squares(
        residual_fn,
        solver,
        z0,
        args=(ode_args, inlet_bc),
        options={"jac": jac_mode},  # "bwd" or "fwd"
        max_steps=max_steps,
    )

    # Evaluate the flowfield at the converged solution
    out_data = evaluate_ode_rhs(x, solution.value, ode_args)

    return out_data, solution



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

# ---------- Generate uniform flow field for initial guess ----------
def initialize_flowfield(num_points, params, fluid):
    state0_in = get_props(cpx.DmassP_INPUTS, params["p0_in"], params["d0_in"], fluid)
    v_in = state0_in["a"] * params["Ma_in"]  # Approximation is fine for initial guess
    h_in = state0_in["h"] - 0.5 * v_in ** 2
    state_in = get_props(cpx.HmassSmass_INPUTS, h_in, state0_in["s"], fluid)
    d_in = state_in["rho"]
    p_in = state_in["p"]
    flowfield_v    = jnp.full((num_points+1,), v_in)
    flowfield_ln_d = jnp.full((num_points+1,), jnp.log(jnp.maximum(d_in, 1e-12)))
    flowfield_ln_p = jnp.full((num_points+1,), jnp.log(jnp.maximum(p_in, 1e-12)))
    return jnp.concatenate([flowfield_v, flowfield_ln_d, flowfield_ln_p])


# ---------- Create function handle for the residual vector ----------
# ---------- Problem is solved in z = [u, log_rho, log_p] ------------
def build_residual_vector(Dx, x):

    def residual(z, parameters):
        # Unpack parameters
        args, inlet_bc = parameters
        u_in, ln_d_in, ln_p_in = inlet_bc

        # Unpack solution vector
        u, ln_d, ln_p = split_z(z, x.shape[0])
        d = jnp.exp(ln_d)
        p = jnp.exp(ln_p)

        # Compute right hand side of the autonomous ODE
        out = evaluate_ode_rhs(x, z, args)
        N_all = out["N"]
        D_tau = out["D"]

        # Scalar normalization for stability (does not change the solution)
        sD = jnp.maximum(jnp.median(jnp.abs(D_tau)), 1e-12)
        Dn = D_tau / sD

        # Evaluate residuals at collocation points
        # Multiply-through PDE residuals (no division by D)
        R_u = Dn * (Dx @ u)    - N_all[:, 0] / sD
        R_d = Dn * (Dx @ ln_d) - N_all[:, 1] / (sD * jnp.maximum(d, 1e-12))
        R_p = Dn * (Dx @ ln_p) - N_all[:, 2] / (sD * jnp.maximum(p, 1e-12))

        # Evaluate residual at boundary condition
        R_u = R_u.at[0].set(u[0]    - u_in)
        R_d = R_d.at[0].set(ln_d[0] - ln_d_in)
        R_p = R_p.at[0].set(ln_p[0] - ln_p_in)

        return jnp.concatenate([R_u, R_d, R_p])

    return residual


def singularity_margin_from_out(out, alpha=60.0, eps=1e-12):
    # out["x"]: (N+1,), out["D"]: (N+1,), out["N"]: (N+1,3)
    D = out["D"]
    N = out["N"]
    sD = jnp.maximum(jnp.median(jnp.abs(D)), eps)
    sN = jnp.maximum(jnp.median(jnp.abs(N)), eps)
    S = jnp.sqrt((D / sD) ** 2 + jnp.sum((N / sN) ** 2, axis=1))  # (N+1,)
    w = jax.nn.softmax(-alpha * S)
    S_c = jnp.sum(w * S)
    return S_c, S, w

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
        Y = jnp.array([xi, ui, di, pi])
        return nozzle_single_phase_core(0.0, Y, args)

    return jax.vmap(per_node, in_axes=(0, 0, 0, 0))(u, ln_d, ln_p, x)


# ---------- Define Chebyshev–Lobatto nodes and differentiation matrix ----------
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
    # x_nodes: (N+1,), y_nodes: (N+1,), x_eval: (M,)
    # Compute the Chebyshev-Lobatto weights
    n = x_nodes.size - 1
    k = jnp.arange(n + 1)
    w = jnp.where((k == 0) | (k == n), 0.5, 1.0) * ((-1.0) ** k)

    # Broadcasted diffs: (N+1, M)
    diff = x_eval[None, :] - x_nodes[:, None]

    # Handle exact node hits to avoid division by zero
    is_node = diff == 0.0  # (N+1, M)
    any_node = jnp.any(is_node, axis=0)  # (M,)

    # Barycentric formula
    num = jnp.sum((w * y_nodes)[:, None] / diff, axis=0)
    den = jnp.sum((w)[:, None] / diff, axis=0)
    interp = num / den

    # Replace columns where x_eval coincides with a node
    # take the corresponding y_node for that column
    y_at_node = jnp.sum(jnp.where(is_node, y_nodes[:, None], 0.0), axis=0)
    return jnp.where(any_node, y_at_node, interp)


@eqx.filter_jit
def maximum_from_poly_nodes(x_nodes, y_nodes):
    """
    Find the location and value of the maximum of the degree-N polynomial interpolant
    through (x_nodes, y_nodes) on the closed interval [x1, x2].

    We:
      1) Map physical nodes x ∈ [x1, x2] to the canonical variable t ∈ [-1, 1].
      2) Build a Vandermonde matrix in powers of t and solve for monomial coefficients.
      3) Differentiate the polynomial (in descending-order coefficients).
      4) Compute all roots of the derivative with fixed output size (strip_zeros=False).
      5) Form a fixed-size candidate set = {endpoints} ∪ {all derivative roots},
         mask invalid candidates (complex or outside [-1, 1]), and select the argmax.
      6) Map the maximizing t back to x.

    Notes:
      - JIT-Friendly: All arrays have static shapes; we avoid data-dependent control flow.
      - roots(..., strip_zeros=False): Prevents data-dependent trimming of leading zeros,
        which would otherwise trigger a ConcretizationTypeError under JIT.
      - Numerical Considerations: A Vandermonde solve is fine for moderate N (e.g., ≤ 40)
        on Chebyshev/Lobatto-like nodes. For very large N, a Chebyshev-based routine
        (colleague matrix) is more stable.

    Args:
        x_nodes: (N+1,) coordinates of interpolation nodes in ascending order (x1, ..., x2).
        y_nodes: (N+1,) function values at those nodes.

    Returns:
        x_star: Argmax location in physical coordinates [x1, x2].
        y_star: Maximum value of the polynomial interpolant on [x1, x2].
    """

    # Function generated with chatGPT (not tested under all scenarios!!!)

    # Map x -> t in [-1, 1] using the affine change of variables.
    x1, x2 = x_nodes[0], x_nodes[-1]
    t_nodes = (2.0 * (x_nodes - x1) / (x2 - x1)) - 1.0

    # Keep N Static Under JIT (shape-derived, so compilation is stable if N is fixed).
    N = x_nodes.shape[0] - 1

    # Build Vandermonde Matrix In Ascending Powers: V[i, j] = t_i^j for j = 0..N.
    powers = jnp.arange(N + 1)
    V = t_nodes[:, None] ** powers[None, :]

    # Solve For Monomial Coefficients In Ascending Order: y ≈ sum_j c_asc[j] * t^j.
    c_asc = jnp.linalg.solve(V, y_nodes)

    # Convert To Descending Order For jnp.poly* APIs (polyder/roots/polyval expect this).
    c_desc = c_asc[::-1]

    # Differentiate Polynomial (Descending Coefficients → Descending Coefficients).
    dc_desc = jnp.polyder(c_desc)

    # Compute Roots With Fixed Output Size: Avoid Data-Dependent Zero-Stripping Under JIT.
    roots_c = jnp.roots(dc_desc, strip_zeros=False)  # Complex array of shape (N-1,)

    # Extract Real Parts And Build A Validity Mask For Real Roots In [-1, 1].
    roots_t = roots_c.real
    roots_ok = jnp.isclose(roots_c.imag, 0.0, atol=1e-12)
    roots_ok = roots_ok & (roots_t >= -1.0) & (roots_t <= 1.0)

    # Build Candidate Set With Fixed Length: Endpoints Are Always Valid.
    t_cand = jnp.concatenate([jnp.array([-1.0, 1.0], dtype=roots_t.dtype), roots_t])
    mask = jnp.concatenate([jnp.array([True, True]), roots_ok])

    # Evaluate Polynomial At All Candidates; Invalidate Non-Real/Out-Of-Range With -inf.
    y_cand_all = jnp.polyval(c_desc, t_cand)
    y_cand = jnp.where(mask, y_cand_all, -jnp.inf)

    # Select Argmax In Canonical Variable And Recover The True Value (Unmasked).
    idx = jnp.argmax(y_cand)
    t_star = t_cand[idx]
    y_star = y_cand_all[idx]

    # Map Back To Physical Coordinate: x = 0.5*(t+1)*(x2-x1) + x1.
    x_star = 0.5 * (t_star + 1.0) * (x2 - x1) + x1
    return x_star, y_star


# ---------- example ----------
if __name__ == "__main__":

    # Define model parameters
    backend = "perfect_gas"
    # backend = "jaxprop"
    fluid_name = "air"

    params = tree_map(
        jnp.asarray,
        {
            "Ma_in": 0.10,       # Pa
            "p0_in": 1.0e5,      # Pa
            "d0_in": 1.20,       # K
            "D_in": 0.050,       # m
            "length": 5.00,      # m
            "roughness": 10e-6,  # m
            "T_wall": 300.0,     # K
        },
    )

    # Numerical settings
    num_points = 35
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
    Ma_array = jnp.asarray(jnp.linspace(0.05, 0.20, 11))
    colors = plt.cm.magma(jnp.linspace(0.2, 0.8, len(Ma_array)))
    z0 = initialize_flowfield(num_points, params, fluid)
    results = []
    for Ma, color in zip(Ma_array, colors):
        t0 = perf_counter()
        params_current = tree_map(jnp.asarray,{**params, "Ma_in": Ma})
        out, sol = solve_nozzle_model(
            params_current,
            fluid,
            wall_friction=False,
            heat_transfer=False,
            # initial_guess=z0,
            num_points=num_points,
            max_steps=max_steps,
            jac_mode=jac_mode,
            verbose=verbose,
            rtol=tolerance,
            atol=tolerance,            
        )
        z0 = sol.value
        dt_ms = (perf_counter() - t0) * 1e3

        # Relative error diagnostics
        mdot_error = (out["m_dot"].max() - out["m_dot"].min()) / out["m_dot"][0] 
        h0_error = (out["h0"].max() - out["h0"].min()) / out["h0"][0]
        s_error = (out["s"].max() - out["s"].min()) / out["s"][0]

        print(
            f"p_in/p0 = {Ma:0.4f} | LM status {sol.result._value:2d} | "
            f"steps {int(sol.stats['num_steps']):3d} | "
            f"mdot error {mdot_error:0.2e} | h0 error {h0_error:0.2e} | "
            f"s_error {s_error:0.2e} | time {dt_ms:7.2f} ms"
        )

        results.append({"PR": Ma, "color": color, "out": out, "sol": sol})


    # --- Plot the solutions ---
    fig, axs = plt.subplots(4, 1, figsize=(5, 9), sharex=True)
    x_dense = jnp.linspace(0.0, params["length"], 500)

    # Pressure (bar)
    axs[0].set_ylabel("Pressure (bar)")
    for r in results:
        out = r["out"]
        x_nodes = out["x"]
        p_nodes = out["p"] * 1e-5
        p_dense = chebyshev_lobatto_interpolate(x_nodes, p_nodes, x_dense)
        axs[0].plot(x_dense, p_dense, color=r["color"])
        axs[0].plot(x_nodes, p_nodes, "o", color=r["color"], markersize=3)

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
