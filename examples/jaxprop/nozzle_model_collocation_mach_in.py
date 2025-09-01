# -----------------------------------------------------------------------------
# Example goal
# -----------------------------------------------------------------------------
# This example demonstrates the solution of the steady-state Euler equations
# in a quasi-1D converging–diverging nozzle with a collocation method for 
# different values of the inlet mach number
# 
# -----------------------------------------------------------------------------
# Approach
# -----------------------------------------------------------------------------
# 1. Governing equations:
#    - The steady, quasi-1D Euler equations are written in terms of the state
#      vector [u, ln(p), ln(rho)], where u is the velocity, p is the static
#      pressure, and rho is the static density.
#    - Using logarithmic pressure and density ensures strict positivity of
#      these variables without requiring explicit inequality constraints.
#    - The PDE in logarithmic form contains 1/p and 1/rho factors in the
#      derivatives, which appear explicitly in the residual equations.
#
# 2. Discretization:
#    - Spatial derivatives are discretized using a Chebyshev–Lobatto
#      collocation method.
#    - This method uses N+1 non-uniform grid points clustered near the
#      boundaries, which improves resolution of steep gradients near the inlet
#      and outlet.
#    - The differentiation matrix `Dx` acts directly on nodal values to
#      produce nodal derivatives.
#
# 3. Boundary conditions:
#    - The inlet boundary condition is given in terms of stagnation pressure
#      p0_in, stagnation density d0_in, and inlet Mach number
#    - Boundary conditions are enforced by replacing the first residual
#      entries with the difference between the computed and target inlet
#      values.
#
# 4. Nonlinear solver:
#    - The ODE residuals for all collocation nodes are assembled into a single
#      vector and passed to an Optimistix nonlinear least-squares solver.
#    - Available solver methods include Gauss–Newton, Dogleg, and
#      Levenberg–Marquardt.
#    - Solver settings (rtol, atol, jacobian mode, etc.) are provided via a
#      `SolverSettings` dataclass, where some parameters are marked as static
#      for JAX compilation efficiency.
#
# 5. Post-processing:
#    - The converged nodal results (pressure, Mach number, enthalpy, entropy,
#      mass flow rate, etc.) are interpolated onto a dense grid for plotting
#      using barycentric interpolation with Chebyshev–Lobatto nodes.
#    - Key diagnostics such as conservation of mass flow, total enthalpy, and
#      entropy uniformity are computed to assess numerical accuracy.
#
# -----------------------------------------------------------------------------
# Why use collocation?
# -----------------------------------------------------------------------------
# - It provides spectral accuracy for smooth solutions, meaning very high
#   accuracy with relatively few nodes.
# - The formulation integrates seamlessly with JAX for automatic
#   differentiation, making it suitable for optimization and sensitivity
#   analysis tasks.
#
# -----------------------------------------------------------------------------


import time
import jax
import jax.numpy as jnp
import optimistix as opx
import matplotlib.pyplot as plt
import coolpropx as cpx

from examples.jaxprop.nozzle_model_solver import (
    solve_nozzle_model_collocation,
    initialize_flowfield,
    NozzleParams,
    BVPSettings,
    replace_param,
    chebyshev_lobatto_interpolate,
)

from examples.jaxprop.nozzle_model_core import symmetric_nozzle_geometry

cpx.set_plot_options()


if __name__ == "__main__":

    # Define model parameters
    fluid_name = "air"
    fluid = cpx.perfect_gas.get_constants(fluid_name, T_ref=300, P_ref=101325)

    params_model = NozzleParams(
        Ma_in=0.25,
        p0_in=5.0e5,  # Pa
        d0_in=1.20,  # kg/m³
        D_in=0.050,  # m
        length=5.00,  # m
        roughness=1e-6,  # m
        T_wall=300.0,  # K
        heat_transfer=0.0,
        wall_friction=0.0,
        Ma_low=0.95,
        Ma_high=1.05,
        fluid=fluid,
        geometry=symmetric_nozzle_geometry,
    )

    params_solver = BVPSettings(
        solve_mode="mach_in",
        num_points=50,
        rtol=1e-8,
        atol=1e-8,
        max_steps=50,
        jac_mode="bwd",
        verbose=False,
        method="GaussNewton",
        warmup_method="Dogleg",
        warmup_steps=5,
    )

    # Inlet Mach number sensitivity analysis
    print("\n" + "-" * 60)
    print("Running Mach number sweep (collocation)")
    print("-" * 60)
    Ma_array = jnp.asarray(jnp.linspace(0.05, 0.25, 5))
    colors = plt.cm.magma(jnp.linspace(0.2, 0.8, len(Ma_array)))
    initial_guess = initialize_flowfield(params_solver.num_points, params_model)
    results = []
    for Ma, color in zip(Ma_array, colors):
        t0 = time.perf_counter()
        params_model = replace_param(params_model, "Ma_in", Ma)
        out, sol = solve_nozzle_model_collocation(
            initial_guess,
            params_model,
            params_solver,
        )

        # Continuation strategy
        # z0 = z0.at[:].set(sol.value)

        # Relative error diagnostics
        dt_ms = (time.perf_counter() - t0) * 1e3
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
    x_dense = jnp.linspace(0.0, params_model.length, 1000)

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
            label=f"$Ma_{{in}}={r["Ma"]:0.3f}$",
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
