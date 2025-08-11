"""
Demonstration script: JAX-compatible thermodynamic ODE + gradient verification

This script shows how to:
  1. Wrap CoolProp property calls with a JAX-safe bridge (`jax.pure_callback`) so they
     can be used inside JAX transformations (`jacfwd`, `jacrev`, `jit`, etc.).
  2. Define a custom JVP rule for a property function (`rho_of`) so forward-mode AD
     works, even though the property evaluation is done via an external (non-JAX) library.
  3. Implement a polytropic expansion process model using `diffrax.diffeqsolve` to
     integrate the ODE dh/dp = eta / rho(h, p).
     - The ODE right-hand side calls the JAX-safe `rho_of` bridge.
     - We use `dfx.DirectAdjoint()` so the ODE solution supports *both* forward-mode
       and reverse-mode autodiff through the integration.
  4. Postprocess the solution by reconstructing the full thermodynamic states along
     the integration path (outside the JAX trace, so these are concrete Python objects).
  5. Define a scalar output of interest — the exit temperature — as a function of four
     scalar inputs: inlet enthalpy h_in, inlet pressure p_in, outlet pressure p_out,
     and efficiency η.
  6. Compute the gradient of the scalar output with respect to the four inputs using:
       - Forward-mode autodiff (`jax.jacfwd`)
       - Reverse-mode autodiff (`jax.jacrev`)
       - Finite-difference approximation (`scipy.optimize._numdiff.approx_derivative`)
  7. Compare the three results to validate correctness of the JAX gradients.

Key points:
- `rho_of` uses `jax.pure_callback` because CoolProp cannot be traced by JAX.
- The custom JVP for `rho_of` uses simple relative-step finite differences to
  propagate tangents in forward mode.
- `dfx.DirectAdjoint` is computationally less efficient than
  `RecursiveCheckpointAdjoint` or `ForwardMode` but allows both fwd/rev autodiff.
- The script serves as a working example for coupling thermodynamic property
  libraries with JAX and diffrax for differentiable process modeling.
"""

# diffrax_polytropic_ode_jax_only.py
import time
import jax
import jax.numpy as jnp
import diffrax as dfx
import equinox as eqx
import coolprop as cpx
import matplotlib.pyplot as plt

cpx.set_plot_options(grid=False)

# -----------------------------------------------------------------------------
# Main API to the polytropic expansion model
# -----------------------------------------------------------------------------
def polytropic_expansion(
    params,
    fluid,
    num_steps=None,
):
    """
    Calculate properties along a compression process defined by a isentropic or polytropic efficiency

    Parameters
    ----------
    fluid : Fluid
        The fluid object used to evaluate thermodynamic properties
    h_in : float
        Enthalpy at the start of the compression process.
    p_in : float
        Pressure at the start of the compression process.
    p_out : float
        Pressure at the end of the compression process.
    efficiency : float
        The efficiency of the compression process.
    efficiency_type : str, optional
        The type of efficiency to be used in the process ('isentropic' or 'polytropic'). Default is 'isentropic'.
    num_steps : int, optional
        The number of steps for the polytropic process calculation. Default is 50.

    Returns
    -------
    tuple
        Tuple containing (state_out, states) where states is a list with all intermediate states

    Raises
    ------
    ValueError
        If an invalid 'efficiency_type' is provided.

    """

    solver  = dfx.Dopri5()
    adjoint = dfx.DirectAdjoint()  # works for fwd+rev when using callbacks
    term = dfx.ODETerm(_polytropic_odefun)
    ctrl = dfx.PIDController(rtol=1e-6, atol=1e-9)
    if num_steps is not None:
        ts = jnp.linspace(p_in, p_out, num_steps)
        saveat = dfx.SaveAt(ts=ts, dense=False, fn=postprocess_ode)
    else:
        saveat = dfx.SaveAt(t1=True, fn=postprocess_ode)

    sol = dfx.diffeqsolve(
        terms=term,
        solver=solver,
        t0=params["p_in"],
        t1=params["p_out"],
        dt0=None,
        y0=params["h_in"],
        args=(params["efficiency"], fluid),
        saveat=saveat,
        stepsize_controller=ctrl,
        adjoint=adjoint,
        max_steps=10_000,
    )
    return sol


# -----------------------------------------------------------------------------
# Right hand side of the diffuser ODE system
# -----------------------------------------------------------------------------

def _polytropic_odefun(t, y, args):
    p, h = t, y
    eta, fluid = args
    rho = get_props(cpx.HmassP_INPUTS, h, p, fluid)["rho"]
    return eta / rho


def postprocess_ode(t, y, args):
    """compute derived outputs at save times"""
    p, h = t, y
    state = get_props(cpx.HmassP_INPUTS, h, p, fluid)
    return state

# -----------------------------------------------------------------------------
# Define one scalar objective function
# -----------------------------------------------------------------------------

def get_exit_temperature(params, fluid):
    sol = polytropic_expansion(params, fluid)
    return sol.ys["T"][-1]


def get_exit_temperature_gradient(params, fluid, rel_eps=1e-6):
    """Finite-diff gradient of exit temperature w.r.t. params (dict values)."""
    keys = list(params.keys())
    x0 = jnp.array([params[k] for k in keys], dtype=jnp.float64)

    def fun_vec(x_vec):
        p = params.copy()
        for k, val in zip(keys, x_vec):
            p[k] = val
        return get_exit_temperature(p, fluid)

    grad_vec = finite_diff_grad(fun_vec, x0, rel_eps=rel_eps)
    return dict(zip(keys, grad_vec))


def finite_diff_grad(fun, x, rel_eps=1e-6):
    """forward 2-point finite-diff gradient of scalar fun: R^n -> R"""
    x = jnp.asarray(x, dtype=jnp.float64)
    f0 = fun(x)
    def one_comp(i, val):
        eps = rel_eps * (jnp.abs(val) + 1.0)
        x_pert = x.at[i].add(eps)
        return (fun(x_pert) - f0) / eps
    g = [one_comp(i, x[i]) for i in range(x.size)]
    return jnp.array(g, dtype=jnp.float64)




# -----------------------------------------------------------------------------
# Main code start
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    # Define the case parameters
    fluid_name = "CO2"
    T_in = 400.0
    p_in = 200e5 
    p_out = 50e5
    efficiency = 0.8
    
    # Initialize fluid and plot depending on backend
    # backend = "perfect_gas"
    backend = "jaxprop"
    x_prop, y_prop = "s", "T"
    if backend == "perfect_gas":
        from coolprop.perfect_gas import get_props, get_perfect_gas_constants
        fluid = get_perfect_gas_constants(fluid_name, T_in, p_in)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.set_xlabel(cpx.LABEL_MAPPING.get(x_prop, x_prop))
        ax.set_ylabel(cpx.LABEL_MAPPING.get(y_prop, y_prop))

    elif backend == "jaxprop":
        from coolprop.jaxprop import get_props
        fluid = cpx.Fluid(name=fluid_name, backend="HEOS")
        fig, ax = fluid.plot_phase_diagram(x_prop, y_prop)

    else:
        raise ValueError("Invalid fluid backend seclection")

    # Group model parameters
    params = {
        "h_in": get_props(cpx.PT_INPUTS, p_in, T_in, fluid)["h"],
        "p_in": p_in,
        "p_out": p_out,
        "efficiency": efficiency,
    }
    params = {k: jnp.asarray(v) for k, v in params.items()}


    # Polytropic efficiency sensitivity analysis
    eff_array = jnp.linspace(1.0, 0.1, 10)
    colors = plt.cm.magma(jnp.linspace(0.2, 0.8, len(eff_array)))
    print("\n" + "-" * 42)
    print("Polytropic efficiency sensitivity analysis")
    print("-" * 42)
    print(f"{'efficiency':>12s} {'T_out [K]':>15s} {'time [ms]':>12s}")

    for i, eff in enumerate(eff_array):
        t0 = time.perf_counter()
        params["efficiency"] = eff
        sol = polytropic_expansion(params, fluid, num_steps=50)
        elapsed_ms = (time.perf_counter() - t0) * 1e3
        s, T, T_out = sol.ys["s"], sol.ys["T"], sol.ys["T"][-1]
        print(f"{eff:12.3f} {T_out:15.8f} {elapsed_ms:12.3f}")
        ax.plot(s, T, label=rf"\\eta_{{p}}={eff:.2f}", color=colors[i])

    fig.tight_layout(pad=1)


    # Prepare gradient functions
    fwd_grad_fn = eqx.filter_jit(jax.jacfwd(get_exit_temperature))
    rev_grad_fn = eqx.filter_jit(jax.jacrev(get_exit_temperature))

    # Vector of parameter names for printing
    keys = list(params.keys())

    # --- Forward-mode timings ---
    print("\n" + "-" * 42)
    print("Gradient timings (forward-mode)")
    print("-" * 42)
    print(f"{'efficiency':>12s} {'grad_norm':>15s} {'time [ms]':>12s}")
    grad_fwd_all = []
    for eff in eff_array:
        params["efficiency"] = eff
        t0 = time.perf_counter()
        g_fwd = fwd_grad_fn(params, fluid)
        elapsed_ms = (time.perf_counter() - t0) * 1e3
        grad_fwd_all.append(g_fwd)
        norm = jnp.linalg.norm(jnp.asarray([val for _, val in g_fwd.items()]))
        print(f"{eff:12.3f} {norm:15.8f} {elapsed_ms:12.3f}")

    # --- Reverse-mode timings ---
    print("\n" + "-" * 42)
    print("Gradient timings (reverse-mode)")
    print("-" * 42)
    print(f"{'efficiency':>12s} {'grad_norm':>15s} {'time [ms]':>12s}")
    grad_rev_all = []
    for eff in eff_array:
        params["efficiency"] = eff
        t0 = time.perf_counter()
        g_rev = rev_grad_fn(params, fluid)
        elapsed_ms = (time.perf_counter() - t0) * 1e3
        grad_rev_all.append(g_rev)
        norm = jnp.linalg.norm(jnp.asarray([val for _, val in g_rev.items()]))
        print(f"{eff:12.3f} {norm:15.8f} {elapsed_ms:12.3f}")

    # --- Finite-difference timings ---
    print("\n" + "-" * 42)
    print("Gradient timings (finite-difference)")
    print("-" * 42)
    print(f"{'efficiency':>12s} {'grad_norm':>15s} {'time [ms]':>12s}")
    grad_fd_all = []
    for eff in eff_array:
        params["efficiency"] = eff
        t0 = time.perf_counter()
        g_fd = get_exit_temperature_gradient(params, fluid, rel_eps=1e-6)
        elapsed_ms = (time.perf_counter() - t0) * 1e3
        grad_fd_all.append(g_fd)
        norm = jnp.linalg.norm(jnp.asarray([val for _, val in g_fd.items()]))
        print(f"{eff:12.3f} {norm:15.8f} {elapsed_ms:12.3f}")

    # --- Gradient calculation accuracy ---
    def alignment_metric(g_ad, g_fd):
        return (jnp.dot(g_ad, g_fd) /
                (jnp.linalg.norm(g_ad) * jnp.linalg.norm(g_fd))) - 1

    def relative_L2_error(g_ad, g_fd):
        return jnp.linalg.norm(g_ad - g_fd) / jnp.linalg.norm(g_fd)

    print("\n" + "-" * 42)
    print("Verification of JAX gradients vs finite differences")
    print("-" * 42)
    print(f"{'efficiency':>12s} {'fwd vs FD':>16s} {'rev vs FD':>16s}")


    for eff, g_fwd_dict, g_rev_dict, g_fd_dict in zip(eff_array, grad_fwd_all, grad_rev_all, grad_fd_all):
        # Convert dicts to arrays in consistent key order
        g_fwd = jnp.asarray([g_fwd_dict[k] for k in keys])
        g_rev = jnp.asarray([g_rev_dict[k] for k in keys])
        g_fd  = jnp.asarray([g_fd_dict[k]  for k in keys])

        align_fwd = relative_L2_error(g_fwd, g_fd)
        align_rev = relative_L2_error(g_rev, g_fd)
        print(f"{eff:12.3f} {align_fwd:16.6e} {align_rev:16.6e}")

    print()
    print("Gradient values")
    print(g_rev)
    print(g_fwd)
    print(g_fd)
    

    # Show figures
    plt.show()