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


# This version is attemption to find the critical location by moving x*
# I did not manage to make it work nicely
# I am not sure which and what conditions I have to give to identify the critical point


# ---------- Main function call ----------
@eqx.filter_jit
def solve_nozzle_model(
    z0,
    params,
    fluid,
    wall_friction=False,
    heat_transfer=False,
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
    Ma_target = params["Ma_target"]

    # # Compute the inlet boundary condition iteratively
    # Ma_in = params["Ma_in"]
    p0_in = params["p0_in"]
    d0_in = params["d0_in"]
    # state_in = compute_static_state(p0_in, d0_in, Ma_in, fluid)
    inlet_bc = (p0_in, d0_in, Ma_target)

    # Compute the Chebyshev basis” (only once per call)
    x_hat, D_hat = chebyshev_lobatto_basis(num_points, -1, +1)
    L = params["x_star0"]
    x = 0.5 * L * (1.0 + x_hat)
    D_x = (2.0 / jnp.maximum(L, 1e-12)) * D_hat

    # Build the residual vector function
    residual_fn = build_residual_vector(D_hat, x_hat)

    inputs = (ode_args, inlet_bc)

    # Configure the solver and solve the problem
    vars = {"step", "loss", "accepted", "step_size"}
    vset = frozenset(vars) if verbose else frozenset()
    # solver = opx.LevenbergMarquardt(rtol=rtol, atol=atol, verbose=vset)
    # solver = opx.GaussNewton(rtol=rtol, atol=atol, verbose=vset)
    solver = opx.Dogleg(rtol=rtol, atol=atol, verbose=vset)
    solution = opx.least_squares(
        residual_fn,
        solver,
        z0,
        args=inputs,
        options={"jac": jac_mode},  # "bwd" or "fwd"
        max_steps=max_steps,
    )

    # Evaluate the flowfield at the converged solution
    _, _, _, x_star = split_z(solution.value, x_hat.shape[0])
    x = 0.5 * x_star * (1.0 + x_hat)
    out_data = evaluate_ode_rhs(x, solution.value, ode_args)

    return out_data, solution




# ---------- Generate uniform flow field for initial guess ----------
def initialize_solution(x, params, fluid):
    state0_in = get_props(cpx.DmassP_INPUTS, params["p0_in"], params["d0_in"], fluid)
    v_in = state0_in["a"] * params["Ma_in_guess"]  # Approximation is fine for initial guess
    h_in = state0_in["h"] - 0.5 * v_in ** 2
    state_in = get_props(cpx.HmassSmass_INPUTS, h_in, state0_in["s"], fluid)
    d_in = state_in["rho"]
    p_in = state_in["p"]
    v = jnp.full_like(x, v_in)
    ln_d = jnp.full_like(x, jnp.log(jnp.maximum(d_in, 1e-12)))
    ln_p = jnp.full_like(x, jnp.log(jnp.maximum(p_in, 1e-12)))
    x_star = jnp.atleast_1d(params["x_star0"])
    return jnp.concatenate([v, ln_d, ln_p, x_star])


# ---------- Create function handle for the residual vector ----------
# ---------- Problem is solved in z = [u, log_rho, log_p] ------------
def build_residual_vector(D_hat, x_hat):

    def residual(z, inputs):
        # Unpack parameters
        ode_args, boundary_conditions = inputs
        p0_in, d0_in, Ma_target = boundary_conditions

        # Unpack solution vector
        u, ln_d, ln_p, x_star = split_z(z, x_hat.shape[0])
        d = jnp.exp(ln_d)
        p = jnp.exp(ln_p)


        # # Known, fixed nozzle length
        # L_geo = ode_args[0]  # or pass it via ode_args
        # eps = 1e-6 * L_geo        # small guard to avoid zero length

        # # Squash eta to (eps, L_geo)
        # x_star = eps + (L_geo - eps) * jax.nn.sigmoid(x_star)

        # Compute right hand side of the autonomous ODE
        x = 0.5 * x_star * (1.0 + x_hat)
        Dx = (2.0 / jnp.maximum(x_star, 1e-12)) * D_hat
        out = evaluate_ode_rhs(x, z, ode_args)
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
        # R_u = R_u.at[-1].set(Dn[-1] * (Dx @ u)[-1])
        R_d = R_d.at[0].set(jnp.log(d0_in/out["d0"][0]))
        R_p = R_p.at[0].set(jnp.log(p0_in/out["p0"][0]))

        # TODO, no it does not make sense. we should impose the p0 and d0 here and let the Mach_in float

        # Evaluate residual for the x_star and Ma_target relation

        R_star = (out["Ma"][-1] - Ma_target)  # scale ~1


        R_full = jnp.concatenate([R_u, R_d, R_p, jnp.asarray([R_star])])
        R_norm = jnp.linalg.norm(R_full)

        jax.debug.print(
            "||R||={R_norm:.6e}, R_u0={R_u0:.6e}, R_d0={R_d0:.6e}, R_p0={R_p0:.6e}, "
            "R_star={R_star:.6e}, Ma_target={Ma_target:.2f}, Ma_last={Ma_last:.6f}, x_star={x_star:0.6f} ",
            R_norm=R_norm,
            R_u0=R_u[0],
            R_d0=R_d[0],
            R_p0=R_p[0],
            R_star=R_star,
            Ma_target=Ma_target,
            Ma_last=out["Ma"][-1],
            x_star=x_star

        )

        return R_full

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
    x_star = z[3 * num_points]  # last entry
    return u, ln_d, ln_p, x_star


def evaluate_ode_rhs(x, z, args):
    """Vectorized full-model eval at all nodes from z=[u, ln(rho), ln(p)]."""
    u, ln_d, ln_p, x_star = split_z(z, x.shape[0])

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



# ---------- example ----------
if __name__ == "__main__":

    # Define model parameters
    backend = "perfect_gas"
    # backend = "jaxprop"
    fluid_name = "air"

    params = tree_map(
        jnp.asarray,
        {
            "Ma_in_guess": 0.10,       # Pa
            "p0_in": 1.0e5,      # Pa
            "d0_in": 1.20,       # K
            "D_in": 0.050,       # m
            "length": 5.00,      # m
            "roughness": 10e-6,  # m
            "T_wall": 300.0,     # K
            "Ma_target": 0.5,     # target Mach at x*
            "x_star0":  .50,      # initial guess for x*
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
    

    # Numerical settings
    num_points = 40
    tolerance = 1e-4
    max_steps = 50
    jac_mode = "bwd"
    verbose = False

    # Inlet Mach number sensitivity analysis
    # Ma_array = jnp.asarray(jnp.linspace(0.1, 0.99, 4))
    Ma_array = jnp.asarray([0.3])
    colors = plt.cm.magma(jnp.linspace(0.2, 0.8, len(Ma_array)))

    results = []
    print("\n" + "-" * 60)
    print("Running pressure ratio sweep (collocation)")
    print("-" * 60)
    # num_points = 50
    z0 = initialize_solution(jnp.linspace(0, 1, num_points+1), params, fluid)
    for Ma, color in zip(Ma_array, colors):
        t0 = perf_counter()
        params_current = tree_map(jnp.asarray, {**params, "Ma_target": Ma})
        out, sol = solve_nozzle_model(
            z0,
            params_current,
            fluid,
            wall_friction=False,
            heat_transfer=False,
            num_points=num_points,
            max_steps=max_steps,
            jac_mode=jac_mode,  # keep as-is
            verbose=verbose,
            rtol=tolerance,
            atol=tolerance,            
        )
        z0 = z0.at[:].set(sol.value) 
        dt_ms = (perf_counter() - t0) * 1e3

        # diagnostics (absolute errors, same as before)
        mdot_error = out["m_dot"].max() - out["m_dot"].min()
        h0_error   = out["h0"].max()   - out["h0"].min()
        s_error    = out["s"].max()    - out["s"].min()

        print(
            f"p_in/p0 = {Ma:0.4f} | LM status {sol.result._value:2d} | "
            f"steps {int(sol.stats['num_steps']):3d} | "
            f"mdot error {mdot_error:0.2e} | h0 error {h0_error:0.2e} | "
            f"s_error {s_error:0.2e} | time {dt_ms:7.2f} ms"
        )

        results.append({"Ma": Ma, "color": color, "out": out, "sol": sol})


    # --- Plot the solutions ---
    fig, axs = plt.subplots(4, 1, figsize=(5, 9), sharex=True)

    # Pressure (bar)
    axs[0].set_ylabel("Area (m2)")
    for r in results:
        out = r["out"]
        x_nodes = out["x"]
        # p_nodes = out["p"] * 1e-5s
        p_nodes = out["A"]
        x_dense = jnp.linspace(out["x"][0], out["x"][-1], 500)
        p_dense = chebyshev_lobatto_interpolate(x_nodes, p_nodes, x_dense)
        axs[0].plot(x_dense, p_dense, color=r["color"])
        axs[0].plot(x_nodes, p_nodes, "o", color=r["color"], markersize=3, label=f"$Ma^*={r["Ma"]}$")
    axs[0].legend(loc="lower right", fontsize=8)

    # Mach number
    axs[1].set_ylabel("Mach number (-)")
    for r in results:
        out = r["out"]
        Ma_nodes = out["Ma"]
        x_dense = jnp.linspace(out["x"][0], out["x"][-1], 500)
        Ma_dense = chebyshev_lobatto_interpolate(out["x"], Ma_nodes, x_dense)
        axs[1].plot(x_dense, Ma_dense, color=r["color"])
        axs[1].plot(out["x"], Ma_nodes, "o", color=r["color"], markersize=3)

    # Static and stagnation enthalpy
    axs[2].set_ylabel("Enthalpy (J/kg)")
    for r in results:
        out = r["out"]
        x_dense = jnp.linspace(out["x"][0], out["x"][-1], 500)
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
        x_dense = jnp.linspace(out["x"][0], out["x"][-1], 500)
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

