import numpy as np
import scipy.integrate
import scipy.optimize
import matplotlib.pyplot as plt
import coolpropx as cpx

from time import perf_counter

cpx.set_plot_options(grid=False)


# -----------------------------------------------------------------------------
# Main API to the vaneless diffuser model
# -----------------------------------------------------------------------------
def evaluate_vaneless_diffuser_1d(
    params,
    fluid,
    number_of_points=None,
    tol=1e-6,
):
    """Evaluate one-dimensional flow in a generic annular duct"""

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
    p_in = state["p"]
    a_in = state["a"]
    v_in = Ma_in * a_in
    v_m_in = v_in * np.cos(alpha_in)
    v_t_in = v_in * np.sin(alpha_in)
    y0 = np.array([v_m_in, v_t_in, d_in, p_in, 0.0, 0.0])

    # Define ODE function
    def odefun(t, y):
        # Rename from ODE terminology to physical variables
        length = t
        v_m, v_t, d, p, s_gen, theta = y

        # Calculate velocity
        v = np.sqrt(v_t**2 + v_m**2)
        alpha = np.arctan2(v_t, v_m)

        # Calculate local geometry
        r = r_fun(r_in, phi, length)
        b = b_fun(b_in, div, length)
        diff_br = br_grad(length, b_in, div, r_in, phi)

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
        M = np.array(
            [
                [d, 0.0, v_m, 0.0],
                [d * v_m, 0.0, 0.0, 1.0],
                [0.0, d * v_m, 0.0, 0.0],
                [0.0, 0.0, -(a**2), d],
            ]
        )

        # Compute source term
        S = np.array(
            [
                -d * v_m / (b * r) * diff_br,
                d * v_t**2 / r * np.sin(phi) - 2 * tau_w / b * np.cos(alpha),
                -d * v_t * v_m / r * np.sin(phi) - 2 * tau_w / b * np.sin(alpha),
                (2 / b) * (G / v_m) * (tau_w * v + q_w),
            ]
        )

        # Compute solution
        core = np.linalg.solve(M, S)  # dy = [dv_m, dv_t, d, p]
        sigma = 2.0 / b * (tau_w * v)
        s_gen_dot = sigma / (d * v_m) / T  # Entropy generation
        theta_dot = (v_t / v_m) / r  # Streamline wrapping angle

        # Store data in dictionary
        out = {"v_t" : v_t,
               "v_m" : v_m,
               "v" : v,
               "Ma" : v / a,
               "Ma_m" : v_m / a,
               "Ma_t" : v_t / a,
               "alpha" : alpha,
               "d" : d,
               "p" : p,
               "s" : s,
               "s_gen" : s_gen,
               "h" : h,
               "h0" : h0,
               "theta" : theta,
               "r": r,
               "b": b,
               "L": L,
               "radius_ratio": r/r_in,
               "area_ratio": (b*r)/(b_in*r_in),
               "Cp": (p-p_in)/(p0_in-p_in),
               }
        
        return np.concatenate([core, np.array([s_gen_dot, theta_dot])]), out

    # Prepare integration points
    t_eval = np.linspace(0.0, L, number_of_points) if number_of_points else None

    # Solve ODE using SciPy RK45
    ode_sol = scipy.integrate.solve_ivp(
        fun=lambda t,y: odefun(t,y)[0],
        t_span=[0.0, L],
        y0=y0,
        method="RK45",
        rtol=tol,
        atol=tol,
        t_eval=t_eval,
    )


    out = postprocess_ode(ode_sol.t, ode_sol.y, odefun)

    return out, ode_sol



def postprocess_ode(t, y, ode_handle):
    """
    Post-processes the output of an ordinary differential equation (ODE) solver.

    This function takes the time points and corresponding ODE solution matrix,
    and for each time point, it calls a user-defined ODE handling function to
    process the state of the ODE system. It collects the results into a
    dictionary where each key corresponds to a state variable and the values
    are numpy arrays of that state variable at each integration step

    Parameters
    ----------
    t : array_like
        Integration points at which the ODE was solved, as a 1D numpy array.
    y : array_like
        The solution of the ODE system, as a 2D numpy array with shape (n,m) where
        n is the number of points and m is the number of state variables.
    ode_handle : callable
        A function that takes in a integration point and state vector and returns a tuple,
        where the first element is ignored (can be None) and the second element
        is a dictionary representing the processed state of the system.

    Returns
    -------
    ode_out : dict
        A dictionary where each key corresponds to a state variable and each value
        is a numpy array containing the values of that state variable at each integration step.
    """
    # Initialize ode_out as a dictionary
    ode_out = {}
    for t_i, y_i in zip(t, y.T):
        _, out = ode_handle(t_i, y_i)

        for key, value in out.items():
            # Initialize with an empty list
            if key not in ode_out:
                ode_out[key] = []
            # Append the value to list of current key
            ode_out[key].append(value)

    # Convert lists to numpy arrays
    for key in ode_out:
        ode_out[key] = np.array(ode_out[key])

    return ode_out


# -----------------------------------------------------------------------------
# Geometry helper functions (NumPy version)
# -----------------------------------------------------------------------------
def r_fun(r_in, phi, m):
    """Calculate the radius from the meridional coordinate"""
    return r_in + np.sin(phi) * m


def b_fun(b_in, div, m):
    """Calculate the channel width from the meridional coordinate"""
    return b_in + 2 * np.tan(div) * m


def br_fun(m, b_in, div, r_in, phi):
    return b_fun(b_in, div, m) * r_fun(r_in, phi, m)


def br_grad(m, b_in, div, r_in, phi, h=1e-8):
    """Finite-difference gradient of br_fun w.r.t. m"""
    return (br_fun(m + h, b_in, div, r_in, phi) - br_fun(m - h, b_in, div, r_in, phi)) / (2 * h)


# -----------------------------------------------------------------------------
# Inlet state helper
# -----------------------------------------------------------------------------
def compute_inlet_static_state(p0, T0, Ma, fluid):
    """
    Calculate the static pressure from stagnation conditions and Mach number.
    """
    state0 = get_props(cpx.PT_INPUTS, p0, T0, fluid)
    s0 = state0["s"]
    h0 = state0["h"]

    def stagnation_definition_error(p):
        state = get_props(cpx.PSmass_INPUTS, p, s0, fluid)
        a = state["a"]
        h = state["h"]
        v = a * Ma
        return h0 - h - v**2 / 2

    p_static = scipy.optimize.fsolve(stagnation_definition_error, p0)[0]
    return p_static, s0


# -----------------------------------------------------------------------------
# Example usage / plotting
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    # Define model parameters (plain floats)
    params = {
        "p0_in": 101325.0,
        "T0_in": 273.15 + 20.0,
        "Ma_in": 0.75,
        "alpha_in": 65 * np.pi / 180,
        "Cf": 0.0,
        "q_w": 0.0,
        "r_in": 1.0,
        "r_out": 3.0,
        "b_in": 0.25,
        "phi": 90 * np.pi / 180,  # pi/2 for radial channel
        "div": 0 * np.pi / 180,   # 0 for constant width channel
    }

    # Convert all parameter values to NumPy arrays
    params = {k: np.array(v) for k, v in params.items()}

    # Define fluid
    from coolpropx.perfect_gas import get_props, get_perfect_gas_constants
    fluid_name = "air"
    fluid = get_perfect_gas_constants(fluid_name, params["T0_in"], params["p0_in"])

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
    ax_3.axis(1.1 * params["r_out"] * np.array([-1, 1, -1, 1]))
    theta = np.linspace(0, 2 * np.pi, 100)
    x_in = params["r_in"] * np.cos(theta)
    y_in = params["r_in"] * np.sin(theta)
    x_out = params["r_out"] * np.cos(theta)
    y_out = params["r_out"] * np.sin(theta)
    ax_3.plot(x_in, y_in, "k", label=None)  # HandleVisibility='off'
    ax_3.plot(x_out, y_out, "k", label=None)  # HandleVisibility='off'
    theta = np.linspace(0, 2 * np.pi, number_of_streamlines + 1)

    # Compute diffuser performance for different friction factors
    Cf_array = np.asarray([0.0, 0.01, 0.02, 0.03])
    colors = plt.cm.magma(np.linspace(0, 1, len(Cf_array)+1))  # Generate colors
    for i, Cf in enumerate(Cf_array):
        params["Cf"] = Cf
        t0 = perf_counter()
        out, odesol = evaluate_vaneless_diffuser_1d(params, fluid, number_of_points=100)
        print(f"Call {i+1}: Model evaluation time: {(perf_counter()-t0)*1e3:.2f} ms")

        # Plot the pressure recovery coefficient distribution
        ax_1.plot(out['radius_ratio'], out['Cp'], label=f"$C_f = {Cf:0.3f}$", color=colors[i])
        ax_1.legend(loc='lower right')

        # Plot the Mach number distribution
        ax_2.plot(out['radius_ratio'], out['Ma'], label=f"$C_f = {Cf:0.3f}$", color=colors[i])
        ax_2.legend(loc='upper right')

        # Plot streamlines
        for j in range(len(theta)):
            x = out['r'] * np.cos(out['theta'] + theta[j])
            y = out['r'] * np.sin(out['theta'] + theta[j])
            if j == 0:
                ax_3.plot(x, y, label=f"$C_f = {Cf:0.3f}$", color=colors[i])
            else:
                ax_3.plot(x, y, color=colors[i])

    # Adjust pad
    for fig in [fig_1, fig_2, fig_3]:
        fig.tight_layout(pad=1)

    # Show plot
    plt.show()








