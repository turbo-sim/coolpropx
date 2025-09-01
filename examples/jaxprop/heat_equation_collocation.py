# steady 1d heat conduction via spectral collocation (JAX + Optimistix LM)
# first-order system:
#   dT/dx = v
#   dv/dx = 0
# Dirichlet BCs:
#   T(0) = T_left
#   T(L) = T_right
#
# This file:
# - builds Chebyshev–Lobatto nodes and a spectral differentiation matrix D
# - assembles a square residual R(z) for collocation + BCs
# - solves R(z)=0 with Levenberg–Marquardt from Optimistix
# - compares to the exact linear solution
#
# Everything is JAX-pure and differentiable end-to-end.

from __future__ import annotations
import jax
import jax.numpy as jnp
import optimistix as opx


# use float64 for spectral accuracy with Chebyshev
jax.config.update("jax_enable_x64", True)


def solve_steady_heat_collocation(
    params,
    N: int = 32,
    rtol=1e-9,
    atol=1e-9,
    max_steps=200,
    verbose=True,
    print_final=True,
):

    # Create basis and differentiation matrix
    x, D = chebyshev_lobatto(N, params["x1"], params["x2"])

    # Initial guess: null field
    T0 = jnp.full_like(x, 0.0)
    v0 = jnp.full_like(x, 0.0)
    z0 = jnp.concatenate([T0, v0])

    # Create residual function
    residual = build_residual_heat_equation(D, params)

    # Solve system of equations
    solution = solve_residual_heat_equation(
        residual,
        z0,
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
        jac="fwd",
        verbose=verbose,
    )

    if print_final:
        print_optimistix_summary(solution, residual)

    # unpack solution
    z_star = solution.value
    T_star = z_star[: N + 1]
    v_star = z_star[N + 1 :]
    return x, T_star, solution


# ---------- chebyshev–lobatto nodes and differentiation matrix ----------
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


# ---------- residual builder for first-order collocation ----------
def build_residual_heat_equation(D, params):
    """
    Residual function for steady 1D heat conduction with constant k, q_gen,
    and *general Robin boundary conditions*.

    params must include:
        x1      : location of the left boundary
        x2      : location of the right boundary
        k       : thermal conductivity
        q_gen   : uniform volumetric heat generation
        bc_left : (alpha0, beta0, gamma0) for alpha*T + beta*dTdx = gamma at x=x1
        bc_right: (alphaL, betaL, gammaL) for alpha*T + beta*dTdx = gamma at x=x2

    Unknowns: z = [T_nodes (N+1), v_nodes (N+1)]
    Equations:
        dT/dx = v                at all nodes
        dv/dx = -q_gen/k         at interior nodes
        Robin BCs at both ends
    """

    # Rename parameters
    k = params["k"]
    q_gen = params["q_gen"]
    alpha1, beta1, gamma1 = params["bc_left"]
    alpha2, beta2, gamma2 = params["bc_right"]

    # Get the index of the interior nodes
    Np1 = D.shape[0]
    interior = jnp.arange(1, Np1 - 1)

    # Define the residuals function
    def residual(z, _args):

        # PDE residuals
        T = z[:Np1]
        v = z[Np1:]
        R1_all = (D @ T) - v
        R2_int = (D @ v)[interior] + (q_gen / k)

        # Robin BCs (use v=dT/dx for derivative)
        R2_bc1 = alpha1 * T[0] + beta1 * v[0] - gamma1
        R2_bc2 = alpha2 * T[-1] + beta2 * v[-1] - gamma2

        return jnp.concatenate([R1_all, R2_int, jnp.array([R2_bc1, R2_bc2])])

    return residual


def solve_residual_heat_equation(
    residual_fun,
    z0,
    rtol: float = 1e-12,
    atol: float = 1e-12,
    max_steps: int = 200,
    jac: str = "fwd",
    verbose: bool = False,
):
    """
    LM solve of R(z)=0 by minimizing 0.5*||R(z)||^2.

    residual_fun: callable (z, args) -> residual pytree/array  (args can be None)
    z0          : initial guess (pytree/array)
    """
    vset = (
        frozenset({"step", "loss", "accepted", "step_size"}) if verbose else frozenset()
    )
    solver = opx.GaussNewton(rtol=rtol, atol=atol, verbose=vset)

    sol = opx.least_squares(
        residual_fun,
        solver,
        z0,
        args=None,
        options={"jac": jac},  # "fwd" or "bwd"
        max_steps=max_steps,
        has_aux=False,
        # throw=True  # keep default; set False if you want a non-throwing failure path
    )
    return sol  # raw Solution; contains .value, .result, .stats, etc.


def print_optimistix_summary(solution, residual_fun, *, args=None):

    def _leaf_sqsum(tree):
        return sum(jnp.vdot(a, a).real for a in jax.tree.leaves(tree))

    def _dg(x):
        try:
            y = jax.device_get(x)
        except Exception:
            y = x
        try:
            if getattr(y, "shape", ()) == ():
                return y.item()
        except Exception:
            pass
        return y

    def _get(obj, dotted, default=None):
        cur = obj
        for name in dotted.split("."):
            if not hasattr(cur, name):
                return default
            cur = getattr(cur, name)
        return cur

    z_star = solution.value
    r = residual_fun(z_star, args)
    ssq = _leaf_sqsum(r)
    rnorm = jnp.sqrt(ssq)
    loss = 0.5 * ssq

    status = solution.result._value
    num_steps = _dg(solution.stats.get("num_steps", None))
    max_steps = _dg(solution.stats.get("max_steps", None))
    step_size = _dg(_get(solution, "state.search_state.step_size", None))
    num_acc = _dg(_get(solution, "state.num_accepted_steps", None))

    print("-" * 80)
    print(f"Exit flag    : {status}")
    if num_steps is not None and max_steps is not None:
        line = f"steps        : {num_steps}/{max_steps}"
        if num_acc is not None:
            line += f"  accepted: {num_acc}"
        print(line)
    print(f"||r(z*)||2   : {float(_dg(rnorm)):.6e}")
    print(f"loss         : {float(_dg(loss)):.6e}   (0.5 * ||r||^2)")
    if step_size is not None:
        print(f"trust radius : {float(step_size):.6e}")
    print("-" * 80)


def solve_steady_state_exact(x, params):
    """
    Exact steady 1D conduction with uniform q_gen and constant k on [x1, x2].

    params MUST include:
      x1, x2, k, q_gen,
      bc_left  = (alpha1, beta1, gamma1)  for  alpha1*T + beta1*T' = gamma1 at x=x1
      bc_right = (alpha2, beta2, gamma2)  for  alpha2*T + beta2*T' = gamma2 at x=x2
    """
    # Rename parameters
    x1 = params["x1"]
    x2 = params["x2"]
    k = params["k"]
    q_gen = params["q_gen"]
    a1, b1, g1 = params["bc_left"]
    a2, b2, g2 = params["bc_right"]

    # General form:  T(x) = -(q/(2k)) x^2 + C1 x + C2,  T'(x) = -(q/k) x + C1
    # Apply Robin at x1 and x2 → 2×2 linear system for [C1, C2]
    A = jnp.array([[a1 * x1 + b1, a1], [a2 * x2 + b2, a2]], dtype=jnp.float64)
    b1 = (g1 + 0.5 * a1 * (q_gen / k) * x1**2 + b1 * (q_gen / k) * x1,)
    b2 = (g2 + 0.5 * a2 * (q_gen / k) * x2**2 + b2 * (q_gen / k) * x2,)
    b = jnp.array([b1, b2], dtype=jnp.float64)
    C1, C2 = jnp.linalg.solve(A, b)
    return -0.5 * (q_gen / k) * x**2 + C1 * x + C2


# ---------- example usage, exact comparison, and a quick gradient check ----------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Define problem
    params = {
        "x1": 0.0,
        "x2": 5,
        "k": 5.0,
        "q_gen": 100.0,
        "bc_left": (1.0, 0.0, 100.0),
        "bc_right": (1.0, 0.0, 0.0),
    }

    params = jax.tree.map(jnp.asarray, params)

    #
    x, T_numeric, sol = solve_steady_heat_collocation(params, N=50)

    T_exact = solve_steady_state_exact(x, params)

    # plot numerical vs exact
    plt.figure()
    plt.plot(x, T_exact, label="Analytic solution")
    plt.plot(x, T_numeric, marker="o", linestyle="--", label="Numerical solution")
    plt.xlabel("x")
    plt.ylabel("T")
    plt.legend()
    plt.tight_layout()

    # # differentiability sanity check: d T_exit / d T_right ≈ 1
    # def T_exit(TR):
    #     _, Tsol, _, _ = solve_steady_heat_collocation(L, T_left, TR, N=N)
    #     return Tsol[-1]

    # dTexit_dTR = jax.grad(T_exit)(T_right)
    # print(f"d T_exit / d T_right ≈ {float(dTexit_dTR):.6f}")

    plt.show()
