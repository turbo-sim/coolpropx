
import scipy.linalg
import scipy.integrate
import scipy.optimize
import numpy as np
import CoolProp as cp
import matplotlib.pyplot as plt


def perfect_gas_mach(number_of_points=None, tol=1e-6):
    """Evaluate one-dimensional flow in an generic annular duct"""


    # Define integration interval
    x_in = 0.30
    x_out = 1.00
    gamma = 1.40
    
    def odefun(t, y):

        # Rename from ODE terminology to physical variables
        x = t     
        Ma, = y

        # Calculate local area
        A = A_fun(x)
        diff_A = dA_fun(x)

        # # # Calculate derivative of area change
        # delta = 1e-6
        # diff_Abis = (A_fun(x+0.5*delta) - A_fun(x-0.5*delta))/delta

        # Compute rate of change of Mach number
        dy = - (1.0 + 0.5 * (gamma - 1.0) * Ma ** 2) / (1 - Ma**2) * Ma / A * diff_A
        dy = np.atleast_1d(dy)

        print(x, Ma, diff_A)



        # Store data in dictionary
        out = {"Ma" : Ma,
               "x" : x,
               "A" : A,
               "diff_A" : diff_A,
               }

        return dy, out

    m_eval = np.linspace(x_in, x_out, number_of_points) if number_of_points else None
    ode_sol = scipy.integrate.solve_ivp(
        fun=lambda t,y: odefun(t,y)[0],
        t_span=[x_in, x_out],
        t_eval=m_eval,
        y0=[2.00],
        method = "LSODA",
        rtol = tol,
        atol = tol,
    )

    # Postprocess solution
    states = postprocess_ode(ode_sol.t, ode_sol.y, odefun)

    return states, ode_sol



def A_fun(x):
    """Area of a nozzle given by a parabola with a throat at x=1/2"""
    A_inlet = 0.20
    A_throat = 0.10
    return A_inlet - 4 * (A_inlet - A_throat) * x * (1.0 - x)

def dA_fun(x):
    """Area of a nozzle given by a parabola with a throat at x=1/2"""
    A_inlet = 0.20
    A_throat = 0.10
    return - 4 * (A_inlet - A_throat) * (1.0 - 2.0 * x)

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




if __name__ == '__main__':

    # Evaluate diffuser model
    states, ode_sol = perfect_gas_mach(
        # number_of_points=200,
    )

    # Plot the pressure recovery coefficient distribution
    fig_2, ax_2 = plt.subplots(figsize=(6, 5))
    ax_2.set_aspect('equal')
    ax_2.grid(True)
    ax_2.set_xlabel('Nozzle length')
    ax_2.set_ylabel('Mach number')
    ax_2.set_xlim([0, 1])
    ax_2.plot(states["x"], states["Ma"], "ko-")

    # Plot the pressure recovery coefficient distribution
    fig_1, ax_1 = plt.subplots(figsize=(6, 5))
    ax_1.set_aspect('equal')
    ax_1.grid(True)
    ax_1.set_xlabel('Nozzle length')
    ax_1.set_ylabel('Nozzle area')
    ax_1.set_xlim([0, 1])
    ax_1.set_ylim([0, 0.5])
    x = np.linspace(0, 1, 100)
    A = A_fun(x)
    ax_1.plot(x, A, "k-")
    ax_1.plot(states["x"], states["A"], "o")

    # Show the figure
    plt.show()


