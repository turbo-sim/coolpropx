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
    d0_in = params["d0_in"]
    state_in = compute_static_state(p0_in, d0_in, Ma_in, fluid)
    inlet_bc = (Ma_in * state_in["a"], jnp.log(state_in["d"]), jnp.log(state_in["p"]))

    # Compute the Chebyshev basis” (only once per call)
    x, D = chebyshev_lobatto(num_points, 0.0, params["length"])

    # Initialize constant flow field over x if not provided
    if initial_guess is None:
        z0 = initialize_flowfield(x, params, fluid)
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
def initialize_flowfield(x, params, fluid):
    state0_in = get_props(cpx.DmassP_INPUTS, params["p0_in"], params["d0_in"], fluid)
    v_in = state0_in["a"] * params["Ma_in"]  # Approximation is fine for initial guess
    h_in = state0_in["h"] - 0.5 * v_in ** 2
    state_in = get_props(cpx.HmassSmass_INPUTS, h_in, state0_in["s"], fluid)
    d_in = state_in["rho"]
    p_in = state_in["p"]
    flowfield_v = jnp.full_like(x, v_in)
    flowfield_ln_d = jnp.full_like(x, jnp.log(jnp.maximum(d_in, 1e-12)))
    flowfield_ln_p = jnp.full_like(x, jnp.log(jnp.maximum(p_in, 1e-12)))
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
def chebyshev_lobatto(N: int, x1: float, x2: float):
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
    Ma_array = jnp.asarray(jnp.linspace(0.05, 0.30, 11))
    colors = plt.cm.magma(jnp.linspace(0.2, 0.8, len(Ma_array)))

    results = []
    print("\n" + "-" * 60)
    print("Running pressure ratio sweep (collocation)")
    print("-" * 60)
    z0 = None
    for Ma, color in zip(Ma_array, colors):
        t0 = perf_counter()
        cur = {**params, "Ma_in": Ma}
        out, sol = solve_nozzle_model(
            cur,
            fluid,
            wall_friction=False,
            heat_transfer=False,
            num_points=50,
            max_steps=300,
            jac_mode="fwd",  # keep as-is
            initial_guess=z0,
            verbose=False,
            rtol=1e-8,
            atol=1e-8,            
        )
        z0 = sol.value
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

        results.append({"PR": Ma, "color": color, "out": out, "sol": sol})


    # --- Plot the solutions ---
    fig, axs = plt.subplots(4, 1, figsize=(5, 9), sharex=True)

    # Pressure (bar)
    axs[0].set_ylabel("Pressure (bar)")
    for r in results:
        out = r["out"]
        axs[0].plot(out["x"], out["p"] * 1e-5, color=r["color"], label=f"p/p0={float(r['PR']):.2f}")
    axs[0].legend(loc="lower right", fontsize=7)

    # Mach number
    axs[1].set_ylabel("Mach number (-)")
    for r in results:
        out = r["out"]
        Ma = out["Ma"] if "Ma" in out else out["u"] / jnp.sqrt(out["p"] / out["rho"])  # fallback
        axs[1].plot(out["x"], Ma, color=r["color"])

    # Static and stagnation enthalpy
    axs[2].set_ylabel("Enthalpy (J/kg)")
    for r in results:
        out = r["out"]
        axs[2].plot(out["x"], out["h"],  color=r["color"], linestyle="-")
        axs[2].plot(out["x"], out["h0"], color=r["color"], linestyle="--")

    # Entropy
    axs[3].set_ylabel("Entropy (J/kg/K)")
    for r in results:
        out = r["out"]
        axs[3].plot(out["x"], out["s"], color=r["color"])
    axs[3].set_xlabel("x (m)")

    # # S metric
    # axs[3].set_ylabel("S metric (-)")
    # for r in results:
    #     out = r["out"]
    #     axs[3].plot(out["x"], out["s_metric"], color=r["color"])
    # axs[3].set_xlabel("x (m)")


    fig.tight_layout(pad=1)
    plt.show()
