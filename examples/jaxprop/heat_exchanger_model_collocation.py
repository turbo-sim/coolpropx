
import time
import jax
import jax.numpy as jnp
import optimistix as opx
import equinox as eqx
import matplotlib.pyplot as plt
import coolpropx as cpx

from typing import Any, Callable

from coolpropx.perfect_gas import get_props


from examples.jaxprop.nozzle_model_solver import (
    nozzle_single_phase_core,
    NozzleParams,
    replace_param,
    f64,
    split_z,
    chebyshev_lobatto_basis,
    ResidualParams,
    SOLVER_MAPPING,

)


cpx.set_plot_options()



class HXParams(eqx.Module):
    """Parameters for a counterflow heat exchanger with two nozzle-like streams."""

    # --- Hot stream ---
    hot_fluid: Any = eqx.field(static=True)
    hot_geometry: Callable = eqx.field(static=True)
    hot_p0_in: jnp.ndarray = f64(1.0e5)   # Pa
    hot_d0_in: jnp.ndarray = f64(1.20)    # kg/m³
    hot_mdot_in: jnp.ndarray = f64(0.5)   # kg/s
    hot_roughness: jnp.ndarray = f64(1e-6)
    hot_heat_transfer: jnp.ndarray = f64(0.0)
    hot_wall_friction: jnp.ndarray = f64(0.0)

    # --- Cold stream ---
    cold_fluid: Any = eqx.field(static=True)
    cold_geometry: Callable = eqx.field(static=True)
    cold_p0_in: jnp.ndarray = f64(1.0e5)  # Pa
    cold_d0_in: jnp.ndarray = f64(1.20)   # kg/m³
    cold_mdot_in: jnp.ndarray = f64(0.5)  # kg/s
    cold_roughness: jnp.ndarray = f64(1e-6)
    cold_heat_transfer: jnp.ndarray = f64(0.0)
    cold_wall_friction: jnp.ndarray = f64(0.0)

    # --- Coupling ---
    U: jnp.ndarray = f64(100.0)  # W/m²/K overall heat transfer coefficient
    L: jnp.ndarray = f64(1.00)   # Length of the heat exchanger


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
    residual_fn = get_residual_hx

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

    # # Evaluate the flowfield at the converged solution
    # out_data = evaluate_ode_rhs(x, solution.value, params_model)

    # TODO: I am thinking that it might be wiser and more computationally efficient to have a single ODE RHS

    # return out_data, solution



def get_residual_hx(z, x, Dx, hx: HXParams):
    n = x.shape[0]
    z_hot, z_cold = split_z_two(z, n)

    # temperatures from state only (no T_wall needed)
    T_hot_nodes  = temperature_nodes_from_z(x, z_hot, hx.hot)
    T_cold_nodes = temperature_nodes_from_z(x, z_cold, hx.cold)

    # one-pass coupled RHS (each uses the other's T as wall)
    out_hot  = evaluate_ode_rhs_hot(x,  z_hot,  T_cold_nodes, hx)
    out_cold = evaluate_ode_rhs_cold(x, z_cold, T_hot_nodes,  hx)

    # unpack for residuals
    N_hot,  D_hot  = out_hot["N"],  out_hot["D"]
    N_cold, D_cold = out_cold["N"], out_cold["D"]

    u_h, ln_d_h, ln_p_h = split_z(z_hot, n)
    u_c, ln_d_c, ln_p_c = split_z(z_cold, n)
    d_h, p_h = jnp.exp(ln_d_h), jnp.exp(ln_p_h)
    d_c, p_c = jnp.exp(ln_d_c), jnp.exp(ln_p_c)

    # PDE residuals
    R_uh = (Dx @ u_h) - N_hot[:, 0]  / D_hot
    R_dh = (Dx @ ln_d_h) - N_hot[:, 1] / D_hot / d_h
    R_ph = (Dx @ ln_p_h) - N_hot[:, 2] / D_hot / p_h

    R_uc = (Dx @ u_c) - N_cold[:, 0]  / D_cold
    R_dc = (Dx @ ln_d_c) - N_cold[:, 1] / D_cold / d_c
    R_pc = (Dx @ ln_p_c) - N_cold[:, 2] / D_cold / p_c

    # boundary conditions
    # hot inlet at x[0]: enforce mdot and stagnations p0,d0
    A_h0, *_ = hx.hot.geometry(x[0], hx.hot.length)
    R_mh0 = hx.hot.mdot_in - d_h[0] * u_h[0] * A_h0
    R_d0h = jnp.log(hx.hot.d0_in / out_hot["d0"][0])
    R_p0h = jnp.log(hx.hot.p0_in / out_hot["p0"][0])

    # cold inlet at x[-1] (counterflow): enforce mdot and stagnations
    A_cL, *_ = hx.cold.geometry(x[-1], hx.cold.length)
    R_mcL = hx.cold.mdot_in - d_c[-1] * u_c[-1] * A_cL
    R_d0c = jnp.log(hx.cold.d0_in / out_cold["d0"][-1])
    R_p0c = jnp.log(hx.cold.p0_in / out_cold["p0"][-1])

    # inject BCs
    R_uh = R_uh.at[0].set(R_mh0)
    R_dh = R_dh.at[0].set(R_d0h)
    R_ph = R_ph.at[0].set(R_p0h)

    R_uc = R_uc.at[-1].set(R_mcL)
    R_dc = R_dc.at[-1].set(R_d0c)
    R_pc = R_pc.at[-1].set(R_p0c)

    return jnp.concatenate([R_uh, R_dh, R_ph, R_uc, R_dc, R_pc])



# you already have get_props(cpx.DmassP_INPUTS, d, p, fluid)
def temperature_nodes_from_z(x_nodes, z_side, params_side):
    n = x_nodes.shape[0]
    _, ln_d, ln_p = split_z(z_side, n)
    d = jnp.exp(ln_d)
    p = jnp.exp(ln_p)

    def per_node(di, pi):
        st = get_props(cpx.DmassP_INPUTS, di, pi, params_side.fluid)
        return st["T"]

    return jax.vmap(per_node)(d, p)

def evaluate_ode_rhs_hot(x, z_hot, T_cold_nodes, hx: HXParams):
    n = x.shape[0]
    u, ln_d, ln_p = split_z(z_hot, n)

    def per_node(ui, ln_di, ln_pi, xi, T_cold_i):
        di = jnp.exp(ln_di)
        pi = jnp.exp(ln_pi)
        yi = jnp.array([ui, di, pi])
        hot_mod = replace_param(hx.hot, "T_wall", T_cold_i)
        return nozzle_single_phase_core(xi, yi, hot_mod)

    return jax.vmap(per_node)(u, ln_d, ln_p, x, T_cold_nodes)


def evaluate_ode_rhs_cold(x, z_cold, T_hot_nodes, hx: HXParams):
    n = x.shape[0]
    u, ln_d, ln_p = split_z(z_cold, n)

    def per_node(ui, ln_di, ln_pi, xi, T_hot_i):
        di = jnp.exp(ln_di)
        pi = jnp.exp(ln_pi)
        yi = jnp.array([ui, di, pi])
        cold_mod = replace_param(hx.cold, "T_wall", T_hot_i)
        return nozzle_single_phase_core(xi, yi, cold_mod)

    return jax.vmap(per_node)(u, ln_d, ln_p, x, T_hot_nodes)


def split_z_two(z, num_points):
    """Split z into hot and cold state subvectors."""
    n = num_points
    z_hot = z[0:3*n]
    z_cold = z[3*n:6*n]
    return z_hot, z_cold


def get_residual_hx(z, params, x, Dx):
    """Coupled collocation residual for counterflow HX."""

    n = x.shape[0]
    z_hot, z_cold = split_z_two(z, n)

    # evaluate both sides
    out_hot = evaluate_ode_rhs_hot(x, z_hot, T_cold_nodes=None, params=params)  # we’ll fix below
    out_cold = evaluate_ode_rhs_cold(x, z_cold, T_hot_nodes=None, params=params)

    # Extract temperatures at collocation nodes
    T_hot_nodes = out_hot["T"]
    T_cold_nodes = out_cold["T"]

    # Re-evaluate with coupling (since each needs the other’s T)
    out_hot = evaluate_ode_rhs_hot(x, z_hot, T_cold_nodes, params)
    out_cold = evaluate_ode_rhs_cold(x, z_cold, T_hot_nodes, params)

    N_hot = out_hot["N"]; D_hot = out_hot["D"]
    N_cold = out_cold["N"]; D_cold = out_cold["D"]

    u_hot, ln_d_hot, ln_p_hot = split_z(z_hot, n)
    u_cold, ln_d_cold, ln_p_cold = split_z(z_cold, n)
    d_hot = jnp.exp(ln_d_hot); p_hot = jnp.exp(ln_p_hot)
    d_cold = jnp.exp(ln_d_cold); p_cold = jnp.exp(ln_p_cold)

    # Hot side residuals
    R_uh = (Dx @ u_hot) - N_hot[:,0]/D_hot
    R_dh = (Dx @ ln_d_hot) - N_hot[:,1]/D_hot/d_hot
    R_ph = (Dx @ ln_p_hot) - N_hot[:,2]/D_hot/p_hot

    # Cold side residuals
    R_uc = (Dx @ u_cold) - N_cold[:,0]/D_cold
    R_dc = (Dx @ ln_d_cold) - N_cold[:,1]/D_cold/d_cold
    R_pc = (Dx @ ln_p_cold) - N_cold[:,2]/D_cold/p_cold

    # TODO: boundary conditions for hot and cold inlets/outlets
    # For example:
    R_uh = R_uh.at[0].set(params.hot.Ma_in - out_hot["Ma"][0])
    R_dh = R_dh.at[0].set(jnp.log(params.hot.d0_in/out_hot["d0"][0]))
    R_ph = R_ph.at[0].set(jnp.log(params.hot.p0_in/out_hot["p0"][0]))

    R_uc = R_uc.at[-1].set(params.cold.Ma_in - out_cold["Ma"][-1])  # counterflow: inlet is at outlet end
    R_dc = R_dc.at[-1].set(jnp.log(params.cold.d0_in/out_cold["d0"][-1]))
    R_pc = R_pc.at[-1].set(jnp.log(params.cold.p0_in/out_cold["p0"][-1]))

    return jnp.concatenate([R_uh,R_dh,R_ph,R_uc,R_dc,R_pc])




if __name__ == "__main__":




    def inner_tube(x, r_out=0.30):
        """
        Return A (m^2), dA/dx (m), perimeter (m), diameter (m) for a symmetric parabolic CD nozzle.
        x: position in m (scalar or array)
        L: total length in m (scalar)
        """
        A = jnp.pi * r_out**2 
        dAdx = jnp.asarray(0.0)
        diameter = 2.0 * r_out
        perimeter = jnp.pi * diameter
        return A, dAdx, perimeter, diameter

    def outer_shell(x, r_in=0.01, r_out=0.02):
        """
        Return A (m^2), dA/dx (m), perimeter (m), diameter (m) for a symmetric parabolic CD nozzle.
        x: position in m (scalar or array)
        L: total length in m (scalar)
        """
        A = jnp.pi * (r_out**2 - r_in**2)
        dAdx = jnp.asarray(0.0)
        diameter_in = 2.0 * r_in
        perimeter = jnp.pi * diameter_in
        return A, dAdx, perimeter, diameter_in



    # Define model parameters
    fluid_name = "air"
    fluid = cpx.perfect_gas.get_constants(fluid_name, T_ref=300, P_ref=101325)



    hot_stream = NozzleParams(
        fluid="air",
        geometry=inner_tube,
        p0_in=1e5,
        d0_in=1.2,
        mdot_in=0.5,
        length=5.0,
    )

    cold_stream = NozzleParams(
        fluid="water",
        geometry=annulus,
        p0_in=2e5,
        d0_in=1000.0,
        mdot_in=0.5,
        length=5.0,
    )

    hx_params = HXParams(hot=hot_stream, cold=cold_stream, U=200.0)


    params_model = NozzleParams(
        Ma_in=0.01,
        p0_in=1.0e5,  # Pa
        d0_in=1.20,  # kg/m³
        D_in=0.050,  # m
        length=100.00,  # m
        roughness=1e-6,  # m
        T_wall=300.0,  # K
        heat_transfer=1.0,
        wall_friction=0.0,
        fluid=fluid,
        geometry=inner_tube,
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
        warmup_steps=0,
    )


    # Inlet Mach number sensitivity analysis
    print("\n" + "-" * 60)
    print("Running Mach number sweep (collocation)")
    print("-" * 60)
    # T_wall = jnp.asarray(jnp.linspace(300, 400, 5))
    T_wall = jnp.asarray([400])
    colors = plt.cm.magma(jnp.linspace(0.2, 0.8, len(T_wall)))
    initial_guess = initialize_flowfield(params_solver.num_points, params_model)
    results = []
    for T, color in zip(T_wall, colors):
        t0 = time.perf_counter()
        params_model = replace_param(params_model, "T_wall", T)
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
            f"Ma_target = {T:0.4f} | Ma_crit = {out['Ma'][0]:0.5f} | Solver status {sol.result._value:2d} | "
            f"steps {int(sol.stats['num_steps']):3d} | "
            f"mdot error {mdot_error:0.2e} | h0 error {h0_error:0.2e} | "
            f"s_error {s_error:0.2e} | time {dt_ms:7.2f} ms"
        )
        results.append({"Ma": T, "color": color, "out": out, "sol": sol})

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
    axs[2].set_ylabel("Temperature (K)")
    for r in results:
        out = r["out"]
        T_dense = chebyshev_lobatto_interpolate(out["x"], out["T"], x_dense)
        T0_dense = chebyshev_lobatto_interpolate(out["x"], out["T0"], x_dense)
        axs[2].plot(x_dense, T_dense, color=r["color"], linestyle="-")
        axs[2].plot(out["x"], out["T"], "o", color=r["color"], markersize=3)
        # axs[2].plot(x_dense, T0_dense, color=r["color"], linestyle="--")
        # axs[2].plot(out["x"], out["T0"], "o", color=r["color"], markersize=3)

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
