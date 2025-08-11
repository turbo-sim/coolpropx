import scipy.linalg
import scipy.integrate
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import coolpropx as cpx


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


def get_geometry(length, total_length, area_in, area_ratio):
    """
    Calculates the cross-sectional area, area slope, perimeter, and diameter
    of a pipe or nozzle at a given length along its axis.

    This function is useful for analyzing variable area pipes or nozzles, where the
    area changes linearly from the inlet to the outlet. The area slope is calculated
    based on the total change in area over the total length, assuming a linear variation.

    Parameters
    ----------
    length : float
        The position along the pipe or nozzle from the inlet (m).
    total_length : float
        The total length of the pipe or nozzle (m).
    area_in : float
        The cross-sectional area at the inlet of the pipe or nozzle (m^2).
    area_ratio : float
        The ratio of the area at the outlet to the area at the inlet.

    Returns
    -------
    area : float
        The cross-sectional area at the specified length (m^2).
    area_slope : float
        The rate of change of the area with respect to the pipe or nozzle's length (m^2/m).
    perimeter : float
        The perimeter of the cross-section at the specified length (m).
    diameter : float
        The diameter of the cross-section at the specified length (m).
    """
    area_slope = (area_ratio - 1.0) * area_in / total_length
    area = area_in + area_slope * length
    radius = np.sqrt(area / np.pi)
    diameter = 2 * radius
    perimeter = np.pi * diameter
    return area, area_slope, perimeter, diameter


def get_wall_friction(velocity, density, viscosity, roughness, diameter):
    """
    Computes the frictional stress at the wall of a pipe due to viscous effects.

    The function first calculates the Reynolds number to characterize the flow.
    It then uses the Haaland equation to find the Darcy-Weisbach friction factor.
    Finally, it calculates the wall shear stress using the Darcy-Weisbach equation.

    Parameters
    ----------
    velocity : float
        The flow velocity of the fluid in the pipe (m/s).
    density : float
        The density of the fluid (kg/m^3).
    viscosity : float
        The dynamic viscosity of the fluid (Pa·s or N·s/m^2).
    roughness : float
        The absolute roughness of the pipe's internal surface (m).
    diameter : float
        The inner diameter of the pipe (m).

    Returns
    -------
    stress_wall : float
        The shear stress at the wall due to friction (Pa or N/m^2).
    friction_factor : float
        The Darcy-Weisbach friction factor, dimensionless.
    reynolds : float
        The Reynolds number, dimensionless, indicating the flow regime.
    """
    reynolds = velocity * density * diameter / viscosity
    friction_factor = get_friction_factor_haaland(reynolds, roughness, diameter)
    stress_wall = (1 / 8) * friction_factor * density * velocity**2
    return stress_wall, friction_factor, reynolds


def get_friction_factor_haaland(reynolds, roughness, diameter):
    """
    Computes the Darcy-Weisbach friction factor using the Haaland equation.

    The Haaland equation provides an explicit formulation for the friction factor
    that is simpler to use than the Colebrook equation, with an acceptable level
    of accuracy for most engineering applications.
    This function implements the Haaland equation as it is presented in many fluid
    mechanics textbooks, such as "Fluid Mechanics Fundamentals and Applications"
    by Cengel and Cimbala (equation 12-93).

    Parameters
    ----------
    reynolds : float
        The Reynolds number, dimensionless, indicating the flow regime.
    roughness : float
        The absolute roughness of the pipe's internal surface (m).
    diameter : float
        The inner diameter of the pipe (m).

    Returns
    -------
    f : float
        The computed friction factor, dimensionless.
    """
    f = (-1.8 * np.log10(6.9 / reynolds + (roughness / diameter / 3.7) ** 1.11)) ** -2
    return f


def get_heat_transfer_coefficient(
    velocity, density, heat_capacity, darcy_friction_factor
):
    """
    Estimates the heat transfer using the Reynolds analogy.

    This function is an adaptation of the Reynolds analogy which relates the heat transfer
    coefficient to the product of the Fanning friction factor, velocity, density, and heat
    capacity of the fluid.

    Parameters
    ----------
    velocity : float
        Velocity of the fluid (m/s).
    density : float
        Density of the fluid (kg/m^3).
    heat_capacity : float
        Specific heat capacity of the fluid at constant pressure (J/kg·K).
    darcy_friction_factor : float
        Darcy friction factor, dimensionless.

    Returns
    -------
    float
        Estimated heat transfer coefficient (W/m^2·K).

    Notes
    -----
    The Fanning friction factor used here is a quarter of the Darcy friction factor.
    """
    fanning_friction_factor = darcy_friction_factor / 4
    return 0.5 * fanning_friction_factor * velocity * density * heat_capacity


def pipeline_steady_state_1D(
    fluid_name,
    pressure_in,
    temperature_in,
    diameter_in,
    length,
    roughness,
    area_ratio=1.00,
    mass_flow=None,
    mach_in=None,
    include_friction=True,
    include_heat_transfer=True,
    temperature_external=None,
    number_of_points=None,
):
    """
    Simulates steady-state flow in a 1D pipeline system.

    This function integrates mass, momentum, and energy equations along the length
    of the pipeline. It models friction using the Darcy-Weisbach equation and the Haaland
    correlation for the friction factor. Heat transfer at the walls is calculated using
    an overall heat transfer coefficient based on the Reynolds analogy. Fluid properties
    are obtained using the CoolProp library.

    Parameters
    ----------
    fluid_name : str
        Name of the fluid as recognized by the CoolProp library.
    pressure_in : float
        Inlet pressure of the fluid (Pa).
    temperature_in : float
        Inlet temperature of the fluid (K).
    diameter_in : float
        Inner diameter of the pipeline (m).
    length : float
        Length of the pipeline (m).
    roughness : float
        Surface roughness of the pipeline (m).
    area_ratio : float, optional
        Ratio of the outlet area to the inlet area (default is 1.00).
    mass_flow : float, optional
        Mass flow rate of the fluid (kg/s). Either mass_flow or mach_in must be specified.
    mach_in : float, optional
        Inlet Mach number. Either mass_flow or mach_in must be specified.
    include_friction : bool, optional
        Whether to include friction in calculations (default is True).
    include_heat_transfer : bool, optional
        Whether to include heat transfer in calculations (default is True).
    temperature_external : float, optional
        External temperature for heat transfer calculations (K).
    number_of_points : int, optional
        Number of points for spatial discretization.

    Returns
    -------
    dict
        A dictionary containing the solution of the pipeline flow, with keys for distance,
        velocity, density, pressure, temperature, and other relevant flow properties.

    Raises
    ------
    ValueError
        If neither or both of mass_flow and mach_in are specified.
    """
    # Check that exactly one of mass_flow or mach_in is provided
    if (mass_flow is None and mach_in is None) or (
        mass_flow is not None and mach_in is not None
    ):
        raise ValueError(
            "Exactly one of 'mass_flow' or 'mach_in' must be specified, but not both."
        )

    # Define geometry
    radius_in = 0.5 * diameter_in
    area_in = np.pi * radius_in**2
    # perimeter_in = 2 * np.pi * radius_in

    # Create Fluid object
    fluid = cpx.Fluid(fluid_name, backend="HEOS", exceptions=True)

    # Calculate inlet density
    state_in = fluid.get_state(cpx.PT_INPUTS, pressure_in, temperature_in)
    density_in = state_in.rho

    # Calculate velocity based on specified parameter
    if mass_flow is not None:
        velocity_in = mass_flow / (area_in * density_in)
    elif mach_in is not None:
        velocity_in = mach_in * state_in.a

    # System of ODEs describing the flow equations
    def odefun(t, y):
        # Rename from ODE terminology to physical variables
        x = t
        v, rho, p = y

        # Calculate thermodynamic state
        state = fluid.get_state(cpx.DmassP_INPUTS, rho, p)

        # Calculate area
        area, area_slope, perimeter, diameter = get_geometry(
            length=x, total_length=length, area_in=area_in, area_ratio=area_ratio
        )

        # Compute friction at the walls
        stress_wall, friction_factor, reynolds = get_wall_friction(
            velocity=v,
            density=rho,
            viscosity=state.mu,
            roughness=roughness,
            diameter=diameter,
        )
        if not include_friction:
            stress_wall = 0.0
            friction_factor = 0.0

        # Calculate heat transfer
        if include_heat_transfer:
            U = get_heat_transfer_coefficient(v, rho, state.cp, friction_factor)
            heat_in = U * (temperature_external - fluid.T)
        else:
            U = 0.0
            heat_in = 0

        # Compute coefficient matrix
        M = np.asarray(
            [
                [rho, v, 0.0],
                [rho * v, 0.0, 1.0],
                [0.0, -state.a**2, 1.0],
            ]
        )

        # Compute right hand side
        G = state.isobaric_expansion_coefficient * state.a**2 / state.cp
        b = np.asarray(
            [
                -rho * v / area * area_slope,
                -perimeter / area * stress_wall,
                +perimeter / area * G / v * (stress_wall * v + heat_in),
            ]
        )

        # Solve the linear system of equations
        dy = scipy.linalg.solve(M, b)

        # Save all relevant variables in dictionary
        out = {
            "distance": x,
            "velocity": v,
            "density": rho,
            "pressure": p,
            "temperature": state.T,
            "speed_of_sound": state.a,
            "viscosity": state.mu,
            "compressibility_factor": state.Z,
            "enthalpy": state.h,
            "entropy": state.s,
            "total_enthalpy": state.h + 0.5 * v**2,
            "mach_number": v / state.a,
            "mass_flow": v * rho * area,
            "area": area,
            "area_slope": area_slope,
            "perimeter": perimeter,
            "diameter": diameter,
            "stress_wall": stress_wall,
            "friction_factor": friction_factor,
            "reynolds": reynolds,
            "source_1": b[0],
            "source_2": b[1],
            "source_3": b[2],
        }

        return dy, out

    # Solve polytropic compression differential equation

    solution = scipy.integrate.solve_ivp(
        lambda t, y: odefun(t, y)[0],  # Give only 'dy' to solver
        [0.0, length],
        [velocity_in, density_in, pressure_in],
        t_eval=np.linspace(0, length, number_of_points) if number_of_points else None,
        method="RK45",
        rtol=1e-9,
        atol=1e-9,
    )

    solution = postprocess_ode(solution.t, solution.y, odefun)

    return solution


def get_choke_length_fanno(Ma, k, f, D):
    """
    Computes the dimensionless choke length for Fanno flow of a perfect gas.

    The dimensionless choke length is calculated using the formula:

        (fL*)/D = (1 - Ma^2)/(kMa^2) + (k+1)/(2k) * ln([(k + 1)Ma^2]/[2 + (k - 1)Ma^2])

    The formula is applicable for adiabatic flow with no heat transfer and friction in
    constant-area ducts.

    Parameters
    ----------
    Ma : float
        Mach number of the flow.
    k : float
        Specific heat ratio of the gas (cp/cv).
    f : float
        Darcy friction factor.
    D : float
        Diameter of the duct.

    Returns
    -------
    float
        The dimensionless choke length (fL*/D) for the given Fanno flow conditions.
    """
    term1 = (1 - Ma**2) / (k * Ma**2)
    term2 = (k + 1) / (2 * k)
    term3 = np.log(((k + 1) * Ma**2) / (2 + (k - 1) * Ma**2))

    dimensionless_length = term1 + term2 * term3

    return dimensionless_length * D / f


def get_critical_area_ratio_isentropic(Ma, k):
    """
    Calculates the critical area ratio for isentropic flow of a perfect gas.

    This function computes the ratio of the area of the flow passage (A) to the
    area at the throat (A*) where the flow is sonic (Mach number, Ma = 1), for a
    given Mach number and specific heat ratio (k) of a perfect gas.

    The formula used for the calculation is given by:

        A/A* = (1/Ma) * ((2/(k + 1)) * (1 + (k - 1)/2 * Ma^2))**((k + 1)/(2*(k - 1)))

    The formula is applicable for isentropic flow with no heat transfer and friction in
    variable-area ducts.

    Parameters
    ----------
    Ma : float
        Mach number of the flow at the area A where the ratio is being calculated.
    k : float
        Specific heat ratio (cp/cv) of the perfect gas.

    Returns
    -------
    float
        The critical area ratio (A/A*) for the specified Mach number and specific
        heat ratio.
    """
    term1 = 2 / (k + 1)
    term2 = 1 + (k - 1) / 2 * Ma**2
    exponent = (k + 1) / (2 * (k - 1))

    area_ratio = (1 / Ma) * (term1 * term2) ** exponent

    return area_ratio
