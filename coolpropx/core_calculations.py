import numpy as np
import CoolProp.CoolProp as CP
import pysolver_view as psv

from . import math
from . import utils
# from .. import pysolver_view as psv

# Universal molar gas constant
GAS_CONSTANT = 8.3144598

# Define property aliases
PROPERTY_ALIAS = {
    "P": "p",
    "rho": "rhomass",
    "density": "rhomass",
    "d": "rhomass",
    "rhomass": "rho",
    "dmass": "rhomass",
    "u": "umass",
    "h": "hmass",
    "s": "smass",
    "cv": "cvmass",
    "cp": "cpmass",
    "a": "speed_sound",
    "Z": "compressibility_factor",
    "mu": "viscosity",
    "k": "conductivity",
    "vapor_quality": "quality_mass",
    "void_fraction": "quality_volume",
}

# Define valid phase change types and their aliases
PHASE_CHANGE_ALIASES = {
    "condensation": "condensation",
    "evaporation": "evaporation",
    "flashing": "evaporation",
    "cavitation": "evaporation",
}

# Define NaN fallbacks for viscosity and thermal conductivity
def get_conductivity(AS):
    # AS.conductivity()
    try:
        return AS.conductivity()
    except ValueError:
        return np.nan

def get_viscosity(AS):
    # AS.viscosity()
    try:
        return AS.viscosity()
    except ValueError:
        return np.nan


# ------------------------------------------------------------------------------------ #
# Equilibrium property calculations in the single-phase region
# ------------------------------------------------------------------------------------ #

def compute_properties_1phase(
    abstract_state,
    generalize_quality=False,
    supersaturation=False,
):
    """Extract single-phase properties from CoolProp abstract state"""

    # Fluid properties available from CoolProp
    props = {}
    AS = abstract_state
    props["T"] = AS.T()
    props["p"] = AS.p()
    props["rhomass"] = AS.rhomass()
    props["umass"] = AS.umass()
    props["hmass"] = AS.hmass()
    props["smass"] = AS.smass()
    # props["gibbsmass"] = AS.gibbsmass()
    props["cvmass"] = AS.cvmass()
    props["cpmass"] = AS.cpmass()
    props["gamma"] = props["cpmass"] / props["cvmass"]
    # props["compressibility_factor"] = AS.compressibility_factor()
    M = AS.molar_mass()
    Z = props["p"] / (props["rhomass"] * (GAS_CONSTANT / M) * props["T"])
    props["compressibility_factor"] = Z
    props["speed_sound"] = AS.speed_sound()
    props["isentropic_bulk_modulus"] = props["rhomass"] * props["speed_sound"] ** 2
    props["isentropic_compressibility"] = 1 / props["isentropic_bulk_modulus"]
    props["isothermal_bulk_modulus"] = 1 / AS.isothermal_compressibility()
    props["isothermal_compressibility"] = AS.isothermal_compressibility()
    isobaric_expansion_coefficient = AS.isobaric_expansion_coefficient()
    props["isobaric_expansion_coefficient"] = isobaric_expansion_coefficient
    props["viscosity"] = get_viscosity(AS)
    props["conductivity"] = get_conductivity(AS)
    props["surface_tension"] = np.nan

    # Compute Gruneisen parameter
    temp = props["isobaric_expansion_coefficient"] / props["isothermal_compressibility"]
    props["gruneisen"] = temp / (props["rhomass"] * props["cvmass"])

    # Generalized quality outside the two-phase region
    if generalize_quality:
        props["Q"] = calculate_generalized_quality(AS)
        props["quality_mass"] = props["Q"]
        props["quality_volume"] = np.nan
    else:
        props["Q"] = AS.Q()
        props["quality_mass"] = props["Q"]
        props["quality_volume"] = props["Q"]

    # Calculate departure from saturation properties
    if supersaturation:
        props = calculate_supersaturation(AS, props)
        props["subcooling"] = calculate_subcooling(AS)
        props["superheating"] = calculate_superheating(AS)

    # Add properties as aliases
    for key, value in PROPERTY_ALIAS.items():
        props[key] = props[value]

    return props


# ------------------------------------------------------------------------------------ #
# Equilibrium property calculations in the two-phase region
# ------------------------------------------------------------------------------------ #
def compute_dsdp_q(AS, pressure, quality, rel_dp=1e-4):
    """
    Compute ds/dp|q using analytic method if available, otherwise fall back to finite difference.

    Parameters:
        AS : CoolProp.AbstractState
            AbstractState instance (already configured with fluid and backend)
        pressure : float
            Pressure [Pa]
        quality : float
            Vapor quality. 0.0 for saturated liquid, 1.0 for saturated vapor
        rel_dp : float
            Relative pressure perturbation for finite difference (default: 1e-4)

    Returns:
        dsdp : float or np.nan
            Derivative of entropy with respect to pressure at constant quality
    """
    try:
        # Try analytic saturation derivative
        AS.update(CP.PQ_INPUTS, pressure, quality)
        dsdp = AS.first_saturation_deriv(CP.iSmass, CP.iP)
        return dsdp
    except ValueError:
        try:
            # Fallback: finite difference
            dp = rel_dp * pressure
            AS.update(CP.PQ_INPUTS, pressure, quality)
            s0 = AS.smass()
            AS.update(CP.PQ_INPUTS, pressure + dp, quality)
            s1 = AS.smass()
            return (s1 - s0) / dp
        except Exception:
            return np.nan



def compute_properties_2phase(abstract_state, supersaturation=False):
    """Compute two-phase fluid properties from CoolProp abstract state

    Get two-phase properties from mixing rules and single-phase CoolProp properties

    Homogeneous equilibrium model

    State formulas for T=T, p=p, mfrac/vfrac(rho), h-s-g-u-cp-cv, mu-k, a

    """

    # Instantiate new AbstractState to compute saturation properties without changing the state of the class
    AS = abstract_state
    fluids = AS.fluid_names()
    if len(fluids) != 1:
        raise ValueError(f"Expected one fluid, got {fluids}")
    cloned_AS = CP.AbstractState(AS.backend_name(), fluids[0])

    # Basic properties of the two-phase mixture
    T_mix = AS.T()
    p_mix = AS.p()
    rho_mix = AS.rhomass()
    u_mix = AS.umass()
    h_mix = AS.hmass()
    s_mix = AS.smass()
    surface_tension = AS.surface_tension()
    # gibbs_mix = AS.gibbsmass()

    # Saturated liquid properties
    cloned_AS.update(CP.QT_INPUTS, 0.00, T_mix)
    rho_L = cloned_AS.rhomass()
    cp_L = cloned_AS.cpmass()
    cv_L = cloned_AS.cvmass()
    k_L = get_conductivity(cloned_AS)
    mu_L = get_viscosity(cloned_AS)
    a_L = cloned_AS.speed_sound()
    # dsdp_L = cloned_AS.first_saturation_deriv(CP.iSmass, CP.iP)
    dsdp_L = compute_dsdp_q(cloned_AS, p_mix, quality=0.0)

    # Saturated vapor properties
    cloned_AS.update(CP.QT_INPUTS, 1.00, T_mix)
    rho_V = cloned_AS.rhomass()
    cp_V = cloned_AS.cpmass()
    cv_V = cloned_AS.cvmass()
    k_V = get_conductivity(cloned_AS)
    mu_V = get_viscosity(cloned_AS)
    a_V = cloned_AS.speed_sound()
    # dsdp_V = cloned_AS.first_saturation_deriv(CP.iSmass, CP.iP)
    dsdp_V = compute_dsdp_q(cloned_AS, p_mix, quality=1.0)

    # Volume fractions of vapor and liquid
    vfrac_V = (rho_mix - rho_L) / (rho_V - rho_L)
    vfrac_L = 1.00 - vfrac_V

    # Mass fractions of vapor and liquid
    mfrac_V = (1 / rho_mix - 1 / rho_L) / (1 / rho_V - 1 / rho_L)
    mfrac_L = 1.00 - mfrac_V

    # Heat capacities of the two-phase mixture
    cp_mix = mfrac_L * cp_L + mfrac_V * cp_V
    cv_mix = mfrac_L * cv_L + mfrac_V * cv_V

    # Transport properties of the two-phase mixture
    k_mix = vfrac_L * k_L + vfrac_V * k_V
    mu_mix = vfrac_L * mu_L + vfrac_V * mu_V

    # Compressibility factor of the two-phase mixture
    M = AS.molar_mass()
    Z_mix = p_mix / (rho_mix * (GAS_CONSTANT / M) * T_mix)

    # Speed of sound of the two-phase mixture
    B1 = vfrac_L / (rho_L * a_L**2) + vfrac_V / (rho_V * a_V**2)
    B2 = vfrac_L * rho_L / cp_L * dsdp_L**2 + vfrac_V * rho_V / cp_V * dsdp_V**2
    compressibility_HEM = B1 + T_mix * B2
    if mfrac_V < 1e-6:  # Avoid discontinuity when Q_v=0
        a_HEM = a_L
    elif mfrac_V > 1.0 - 1e-6:  # Avoid discontinuity when Q_v=1
        a_HEM = a_V
    else:
        a_HEM = (1 / rho_mix / compressibility_HEM) ** 0.5

    # Store properties in dictionary
    props = {}
    props["T"] = T_mix
    props["p"] = p_mix
    props["rhomass"] = rho_mix
    props["umass"] = u_mix
    props["hmass"] = h_mix
    props["smass"] = s_mix
    # props["gibbsmass"] = gibbs_mix
    props["cvmass"] = cv_mix
    props["cpmass"] = cp_mix
    props["gamma"] = props["cpmass"] / props["cvmass"]
    props["compressibility_factor"] = Z_mix
    props["speed_sound"] = a_HEM
    props["isentropic_bulk_modulus"] = rho_mix * a_HEM**2
    props["isentropic_compressibility"] = (rho_mix * a_HEM**2) ** -1
    props["isothermal_bulk_modulus"] = np.nan
    props["isothermal_compressibility"] = np.nan
    props["isobaric_expansion_coefficient"] = np.nan
    props["viscosity"] = mu_mix
    props["conductivity"] = k_mix
    props["Q"] = mfrac_V
    props["quality_mass"] = mfrac_V
    props["quality_volume"] = vfrac_V
    props["surface_tension"] = surface_tension

    if supersaturation:
        props["subcooling"] = calculate_subcooling(AS)
        props["superheating"] = calculate_superheating(AS)
        props = calculate_supersaturation(AS, props)

    # Add properties as aliases
    for key, value in PROPERTY_ALIAS.items():
        props[key] = props[value]

    # Add saturation properties as subdictionaries
    props["saturation_liquid"] = {
        "rhomass": rho_L,
        "cpmass": cp_L,
        "cvmass": cv_L,
        "conductivity": k_L,
        "viscosity": mu_L,
        "speed_sound": a_L,
        "dsdp": dsdp_L,
    }

    props["saturation_vapor"] = {
        "rhomass": rho_V,
        "cpmass": cp_V,
        "cvmass": cv_V,
        "conductivity": k_V,
        "viscosity": mu_V,
        "speed_sound": a_V,
        "dsdp": dsdp_V,
    }

    for key, value in PROPERTY_ALIAS.items():
        if value in props["saturation_liquid"]:
            props["saturation_liquid"][key] = props["saturation_liquid"][value]
        if value in props["saturation_vapor"]:
            props["saturation_vapor"][key] = props["saturation_vapor"][value]
            
    return props



# ------------------------------------------------------------------------------------ #
# Property calculations using Helmholtz equations of state
# ------------------------------------------------------------------------------------ #


def compute_properties_metastable_rhoT(
    abstract_state, rho, T, supersaturation=False, generalize_quality=False
):
    r"""
    Compute the thermodynamic properties of a fluid using temperature-density calls to the Helmholtz energy equation of state (HEOS).

    Parameters
    ----------
    abstract_state : CoolProp.AbstractState
        The abstract state of the fluid for which the properties are to be calculated.
    rho : float
        Density of the fluid (kg/m³).
    T : float
        Temperature of the fluid (K).
    supersaturation : bool, optional
        Whether to evaluate supersaturation properties. Default is False.

    Returns
    -------
    dict
        Thermodynamic properties computed at the given density and temperature.

    Notes
    -----
    The Helmholtz energy equation of state (HEOS) expresses the Helmholtz energy as an explicit function
    of temperature and density:

    .. math::
        \Phi = \Phi(\rho, T)

    In dimensionless form, the Helmholtz energy is given by:

    .. math::
        \phi(\delta, \tau) = \frac{\Phi(\delta, \tau)}{R T}

    where:

    - :math:`\phi` is the dimensionless Helmholtz energy
    - :math:`R` is the gas constant of the fluid
    - :math:`\delta = \rho / \rho_c` is the reduced density
    - :math:`\tau = T_c / T` is the inverse of the reduced temperature

    Thermodynamic properties can be derived from the Helmholtz energy and its partial derivatives.
    The following table summarizes the expressions for various properties:

    .. list-table:: Helmholtz energy thermodynamic relations
        :widths: 30 70
        :header-rows: 1

        * - Property name
          - Mathematical relation
        * - Pressure
          - .. math:: Z = \frac{p}{\rho R T} = \delta \cdot \phi_{\delta}
        * - Entropy
          - .. math:: \frac{s}{R} = \tau \cdot \phi_{\tau} - \phi
        * - Internal energy
          - .. math:: \frac{u}{R T} = \tau \cdot \phi_{\tau}
        * - Enthalpy
          - .. math:: \frac{h}{R T} = \tau \cdot \phi_{\tau} + \delta \cdot \phi_{\delta}
        * - Isochoric heat capacity
          - .. math:: \frac{c_v}{R} = -\tau^2 \cdot \phi_{\tau \tau}
        * - Isobaric heat capacity
          - .. math:: \frac{c_p}{R} = -\tau^2 \cdot \phi_{\tau \tau} + \frac{(\delta \cdot \phi_{\delta} - \tau \cdot \delta \cdot \phi_{\tau \delta})^2}{2 \cdot \delta \cdot \phi_{\delta} + \delta^2 \cdot \phi_{\delta \delta}}
        * - Isobaric expansivity
          - .. math:: \alpha \cdot T = \frac{\delta \cdot \phi_{\delta} - \tau \cdot \delta \cdot \phi_{\tau \delta}}{2 \cdot \delta \cdot \phi_{\delta} + \delta^2 \cdot \phi_{\delta \delta}}
        * - Isothermal compressibility
          - .. math:: \rho R T \beta = \left(2 \cdot \delta \cdot \phi_{\delta} + \delta^2 \cdot \phi_{\delta \delta} \right)^{-1}
        * - Isothermal bulk modulus
          - .. math:: \frac{K_T}{\rho R T} = 2 \cdot \delta \cdot \phi_{\delta} + \delta^2 \cdot \phi_{\delta \delta}
        * - Isentropic bulk modulus
          - .. math:: \frac{K_s}{\rho R T} = 2 \cdot \delta \cdot \phi_{\delta} + \delta^2 \ \cdot \phi_{\delta \delta} - \frac{(\delta \cdot \phi_{\delta} - \tau \cdot \delta \cdot \phi_{\tau \delta})^2}{\tau^2 \cdot \phi_{\tau \tau}}
        * - Joule-Thompson coefficient
          - .. math:: \rho R \mu_{\mathrm{JT}} = - \frac{\delta \cdot \phi_{\delta} + \tau \cdot \delta \cdot \phi_{\tau \delta} + \delta^2 \cdot \phi_{\delta \delta}}{(\delta \cdot \phi_{\delta} - \tau \cdot \delta \cdot \phi_{\tau \delta})^2 - \tau^2 \cdot \phi_{\tau \tau} \cdot (2 \cdot \delta \cdot \phi_{\delta} + \delta^2 \cdot \phi_{\delta \delta})}

    Where the following abbreviations are used:

    - :math:`\phi_{\delta} = \left(\frac{\partial \phi}{\partial \delta}\right)_{\tau}`
    - :math:`\phi_{\tau} = \left(\frac{\partial \phi}{\partial \tau}\right)_{\delta}`
    - :math:`\phi_{\delta \delta} = \left(\frac{\partial^2 \phi}{\partial \delta^2}\right)_{\tau, \tau}`
    - :math:`\phi_{\tau \tau} = \left(\frac{\partial^2 \phi}{\partial \tau^2}\right)_{\delta, \delta}`
    - :math:`\phi_{\tau \delta} = \left(\frac{\partial^2 \phi}{\partial \tau \delta}\right)_{\delta, \tau}`

    This function can be used to estimate metastable properties using the equation of state beyond the saturation lines.
    """

    # Update thermodynamic state
    AS = abstract_state
    if isinstance(rho, np.ndarray):
        rho = rho.item()
    if isinstance(T, np.ndarray):
        T = T.item()
    AS.update(CP.DmassT_INPUTS, rho, T)

    # Get fluid constant properties
    R = GAS_CONSTANT
    M = AS.molar_mass()
    T_crit = AS.T_critical()
    rho_crit = AS.rhomass_critical()

    # Compute reduced variables
    tau = T_crit / T
    delta = rho / rho_crit

    # Compute from the Helmholtz energy derivatives
    alpha = AS.alpha0() + AS.alphar()
    dalpha_dTau = AS.dalpha0_dTau() + AS.dalphar_dTau()
    dalpha_dDelta = AS.dalpha0_dDelta() + AS.dalphar_dDelta()
    d2alpha_dTau2 = AS.d2alpha0_dTau2() + AS.d2alphar_dTau2()
    d2alpha_dDelta2 = AS.d2alpha0_dDelta2() + AS.d2alphar_dDelta2()
    d2alpha_dDelta_dTau = AS.d2alpha0_dDelta_dTau() + AS.d2alphar_dDelta_dTau()

    # Compute thermodynamic properties from Helmholtz energy EOS
    props = {}
    props["T"] = T
    props["p"] = (R / M) * T * rho * delta * dalpha_dDelta
    props["rhomass"] = rho
    props["umass"] = (R / M) * T * (tau * dalpha_dTau)
    props["hmass"] = (R / M) * T * (tau * dalpha_dTau + delta * dalpha_dDelta)
    props["smass"] = (R / M) * (tau * dalpha_dTau - alpha)
    # props["gibbsmass"] = (R / M) * T * (alpha + delta * dalpha_dDelta)
    props["cvmass"] = (R / M) * (-(tau**2) * d2alpha_dTau2)
    props["cpmass"] = (R / M) * (
        -(tau**2) * d2alpha_dTau2
        + (delta * dalpha_dDelta - delta * tau * d2alpha_dDelta_dTau) ** 2
        / (2 * delta * dalpha_dDelta + delta**2 * d2alpha_dDelta2 + 1e-12)
    )
    props["gamma"] = props["cpmass"] / props["cvmass"]
    props["compressibility_factor"] = delta * dalpha_dDelta
    a_square = (R / M * T) * (
        (2 * delta * dalpha_dDelta + delta**2 * d2alpha_dDelta2)
        - (delta * dalpha_dDelta - delta * tau * d2alpha_dDelta_dTau) ** 2
        / (tau**2 * d2alpha_dTau2)
    )
    props["speed_sound"] = np.sqrt(a_square) if a_square > 0 else np.nan
    props["isentropic_bulk_modulus"] = (rho * R / M * T) * (
        (2 * delta * dalpha_dDelta + delta**2 * d2alpha_dDelta2)
        - (delta * dalpha_dDelta - delta * tau * d2alpha_dDelta_dTau) ** 2
        / (tau**2 * d2alpha_dTau2)
    )
    props["isentropic_compressibility"] = 1 / props["isentropic_bulk_modulus"]
    props["isothermal_bulk_modulus"] = (
        R / M * T * rho * (2 * delta * dalpha_dDelta + delta**2 * d2alpha_dDelta2)
    )
    props["isothermal_compressibility"] = 1 / (
        R
        / M
        * T
        * rho
        * (2 * delta * dalpha_dDelta + delta**2 * d2alpha_dDelta2 + 1e-12)
    )
    props["isobaric_expansion_coefficient"] = (
        (1 / T)
        * (delta * dalpha_dDelta - delta * tau * d2alpha_dDelta_dTau)
        / (2 * delta * dalpha_dDelta + delta**2 * d2alpha_dDelta2 + 1e-12)
    )
    props["viscosity"] = get_viscosity(AS)
    props["conductivity"] = get_conductivity(AS)
    props["surface_tension"] = np.nan


    # Compute Gruneisen parameter
    temp = props["isobaric_expansion_coefficient"] / props["isothermal_compressibility"]
    props["gruneisen"] = temp / (props["rhomass"] * props["cvmass"])

    if supersaturation:
        props = calculate_supersaturation(AS, props)

    # Generalized quality outside the two-phase region
    if generalize_quality:
        props["Q"] = calculate_generalized_quality(AS)
        props["quality_mass"] = props["Q"]
        props["quality_volume"] = 1.00 if props["Q"] >= 1 else 0.00
    else:
        props["Q"] = np.nan
        props["quality_mass"] = np.nan
        props["quality_volume"] = np.nan


    # Add properties as aliases
    for key, value in PROPERTY_ALIAS.items():
        props[key] = props[value]

    return props


# ------------------------------------------------------------------------------------ #
# Property calculations using CoolProp solver
# ------------------------------------------------------------------------------------ #


def compute_properties_coolprop(
    abstract_state,
    input_type,
    prop_1,
    prop_2,
    generalize_quality=False,
    supersaturation=False,
):
    r"""
    Set the thermodynamic state of the fluid based on input properties.

    This method updates the thermodynamic state of the fluid in the CoolProp ``abstractstate`` object
    using the given input properties. It then calculates either single-phase or two-phase
    properties based on the current phase of the fluid.

    If the calculation of properties fails, `converged_flag` is set to False, indicating an issue with
    the property calculation. Otherwise, it's set to True.

    Aliases of the properties are also added to the ``Fluid.properties`` dictionary for convenience.

    Parameters
    ----------
    input_type : int
        The variable pair used to define the thermodynamic state. This should be one of the
        predefined input pairs in CoolProp, such as ``PT_INPUTS`` for pressure and temperature.

        The valid options for the argument 'input_type' are summarized below.

                 .. list-table::
                        :widths: 50 30
                        :header-rows: 1

                        * - Input pair name
                          - Input pair mapping
                        * - QT_INPUTS
                          - 1
                        * - PQ_INPUTS
                          - 2
                        * - QSmolar_INPUTS
                          - 3
                        * - QSmass_INPUTS
                          - 4
                        * - HmolarQ_INPUTS
                          - 5
                        * - HmassQ_INPUTS
                          - 6
                        * - DmolarQ_INPUTS
                          - 7
                        * - DmassQ_INPUTS
                          - 8
                        * - PT_INPUTS
                          - 9
                        * - DmassT_INPUTS
                          - 10
                        * - DmolarT_INPUTS
                          - 11
                        * - HmolarT_INPUTS
                          - 12
                        * - HmassT_INPUTS
                          - 13
                        * - SmolarT_INPUTS
                          - 14
                        * - SmassT_INPUTS
                          - 15
                        * - TUmolar_INPUTS
                          - 16
                        * - TUmass_INPUTS
                          - 17
                        * - DmassP_INPUTS
                          - 18
                        * - DmolarP_INPUTS
                          - 19
                        * - HmassP_INPUTS
                          - 20
                        * - HmolarP_INPUTS
                          - 21
                        * - PSmass_INPUTS
                          - 22
                        * - PSmolar_INPUTS
                          - 23
                        * - PUmass_INPUTS
                          - 24
                        * - PUmolar_INPUTS
                          - 25
                        * - HmassSmass_INPUTS
                          - 26
                        * - HmolarSmolar_INPUTS
                          - 27
                        * - SmassUmass_INPUTS
                          - 28
                        * - SmolarUmolar_INPUTS
                          - 29
                        * - DmassHmass_INPUTS
                          - 30
                        * - DmolarHmolar_INPUTS
                          - 31
                        * - DmassSmass_INPUTS
                          - 32
                        * - DmolarSmolar_INPUTS
                          - 33
                        * - DmassUmass_INPUTS
                          - 34
                        * - DmolarUmolar_INPUTS
                          - 35

    prop_1 : float
        The first property value corresponding to the input type.
    prop_2 : float
        The second property value corresponding to the input type.


    Returns
    -------
    dict
        A dictionary object containing the fluid properties

    """

    # Update Coolprop thermodynamic state
    abstract_state.update(input_type, prop_1, prop_2)

    # Retrieve single-phase properties
    is_two_phase = abstract_state.phase() == CP.iphase_twophase
    if not is_two_phase:
        properties = compute_properties_1phase(
            abstract_state,
            generalize_quality=generalize_quality,
            supersaturation=supersaturation,
        )
    else:
        properties = compute_properties_2phase(
            abstract_state,
            supersaturation=supersaturation,
        )

    # Define flag to check if state is within the two-phase region
    properties["is_two_phase"] = is_two_phase
    return properties


# ------------------------------------------------------------------------------------ #
# Property calculations using custom solver
# ------------------------------------------------------------------------------------ #

def compute_properties(
    abstract_state,
    prop_1,
    prop_1_value,
    prop_2,
    prop_2_value,
    calculation_type,
    rhoT_guess_metastable=None,
    rhoT_guess_equilibrium=None,
    phase_change=None,
    blending_variable=None,
    blending_onset=None,
    blending_width=None,
    supersaturation=False,
    generalize_quality=False,
    solver_algorithm="hybr",
    solver_tolerance=1e-6,
    solver_max_iterations=100,
    print_convergence=False,
):
    r"""
    .. _compute_properties:

    Determine the thermodynamic state for the given fluid property pair by iterating on the 
    density-temperature native inputs,

    This function uses a non-linear root finder to determine the solution of the nonlinear system given by:

    .. math::

        R_1(\rho,\, T) = y_1 - y_1(\rho,\, T) = 0 \\
        R_2(\rho,\, T) = y_2 - y_2(\rho,\, T) = 0

    where :math:`(y_1,\, y_2)` is the given thermodynamic property pair (e.g., enthalpy and pressure),
    while density and temperature :math:`(\rho,\, T)` are the independent variables that the solver 
    iterates until the residual of the problem is driven to zero.
     
      
    
    TODO    The calculations :math:`y_1(\rho,\, T)` and
    :math:`y_2(\rho,\, T)` are performed by [dependns on input]
    
    equilibrium calculations (coolprop)
    evaluating the Helmholtz energy equation of state.
    blending of both


    Parameters
    ----------
    prop_1 : str
        The the type of the first thermodynamic property.
    prop_1_value : float
        The the numerical value of the first thermodynamic property.
    prop_2 : str
        The the type of the second thermodynamic property.
    prop_2_value : float
        The the numerical value of the second thermodynamic property.
    rho_guess : float
        Initial guess for density
    T_guess : float
        Initial guess for temperature
    calculation_type : str
        The type of calculation to perform. Valid options are 'equilibrium', 'metastable', and 'blending'.
    supersaturation : bool, optional
        If True, calculates supersaturation variables. Defaults to False.
    generalize_quality : bool, optional
        If True, generalizes quality outside two-phase region. Defaults to False.
    blending_variable : str, optional
        The variable used for blending properties. Required if `calculation_type` is 'blending'.
    blending_onset : float, optional
        The onset value for blending. Required if `calculation_type` is 'blending'.
    blending_width : float, optional
        The width value for blending. Required if `calculation_type` is 'blending'.
    solver_algorithm : str, optional
        Method to be used for solving the nonlinear system. Defaults to 'hybr'.
    solver_tolerance : float, optional
        Tolerance for the solver termination. Defaults to 1e-6.
    solver_max_iterations : integer, optional
        Maximum number of iterations of the solver. Defaults to 100.
    print_convergence : bool, optional
        If True, displays the convergence progress. Defaults to False.

    Returns
    -------
    dict       
        Thermodynamic properties corresponding to the given input pair.
    """

    # Perform calculations according to specified type
    if calculation_type == "equilibrium":
        rho_guess, T_guess = np.asarray(rhoT_guess_equilibrium)
        return _perform_flash_calculation(
            abstract_state=abstract_state,
            prop_1=prop_1,
            prop_1_value=prop_1_value,
            prop_2=prop_2,
            prop_2_value=prop_2_value,
            rho_guess=rho_guess,
            T_guess=T_guess,
            calculation_type="equilibrium",
            supersaturation=supersaturation,
            generalize_quality=generalize_quality,
            solver_algorithm=solver_algorithm,
            solver_tolerance=solver_tolerance,
            solver_max_iterations=solver_max_iterations,
            print_convergence=print_convergence,
        )

    elif calculation_type == "metastable":
        rho_guess, T_guess = np.asarray(rhoT_guess_metastable)
        return _perform_flash_calculation(
            abstract_state=abstract_state,
            prop_1=prop_1,
            prop_1_value=prop_1_value,
            prop_2=prop_2,
            prop_2_value=prop_2_value,
            rho_guess=rho_guess,
            T_guess=T_guess,
            calculation_type="metastable",
            supersaturation=supersaturation,
            generalize_quality=generalize_quality,
            solver_algorithm=solver_algorithm,
            solver_tolerance=solver_tolerance,
            solver_max_iterations=solver_max_iterations,
            print_convergence=print_convergence,
        )

    elif calculation_type == "blending":

        if (
            phase_change is None
            or blending_variable is None
            or blending_onset is None
            or blending_width is None
        ):
            msg = (
                f"The following variables must be specified when calculation_type='{calculation_type}':\n"
                f"   1. phase_change: {phase_change}\n"
                f"   2. blending_variable: {blending_variable}\n"
                f"   3. blending_onset: {blending_onset}\n"
                f"   4. blending_width: {blending_width}\n"
            )
            raise ValueError(msg)

        # Equilibrium state
        rho_guess, T_guess = np.asarray(rhoT_guess_equilibrium)
        props_eq = _perform_flash_calculation(
            abstract_state=abstract_state,
            prop_1=prop_1,
            prop_1_value=prop_1_value,
            prop_2=prop_2,
            prop_2_value=prop_2_value,
            rho_guess=rho_guess,
            T_guess=T_guess,
            calculation_type="equilibrium",
            supersaturation=supersaturation,
            generalize_quality=generalize_quality,
            solver_algorithm=solver_algorithm,
            solver_tolerance=solver_tolerance,
            solver_max_iterations=solver_max_iterations,
            print_convergence=print_convergence,
        )

        # Metastable properties
        rho_guess, T_guess = np.asarray(rhoT_guess_metastable)
        props_meta = _perform_flash_calculation(
            abstract_state=abstract_state,
            prop_1=prop_1,
            prop_1_value=prop_1_value,
            prop_2=prop_2,
            prop_2_value=prop_2_value,
            rho_guess=rho_guess,
            T_guess=T_guess,
            calculation_type="metastable",
            supersaturation=supersaturation,
            generalize_quality=False,
            solver_algorithm=solver_algorithm,
            solver_tolerance=solver_tolerance,
            solver_max_iterations=solver_max_iterations,
            print_convergence=print_convergence,
        )
        props_meta["Q"] = 1.00 if phase_change == "condensation" else 0.00

        # Blend properties
        props_blended = blend_properties(
            phase_change=phase_change,
            props_equilibrium=props_eq,
            props_metastable=props_meta,
            blending_variable=blending_variable,
            blending_onset=blending_onset,
            blending_width=blending_width,
        )

        return props_blended, props_eq, props_meta

    raise ValueError(f"Unknown calculation type: {calculation_type}")


# Helper functions
def _perform_flash_calculation(
    abstract_state,
    prop_1,
    prop_1_value,
    prop_2,
    prop_2_value,
    rho_guess,
    T_guess,
    calculation_type,
    supersaturation,
    generalize_quality,
    solver_algorithm,
    solver_tolerance,
    solver_max_iterations,
    print_convergence,
):

    # Ensure prop_1_value and prop_2_value are scalar numbers
    if not utils.is_float(prop_1_value) or not utils.is_float(prop_2_value):
        msg = f"Both prop_1_value and prop_2_value must be scalar numbers. Received: prop_1_value={prop_1_value}, prop_2_value={prop_2_value}"
        raise ValueError(msg)

    # Validate initial guesses for rho and T
    if not utils.is_float(rho_guess) or not utils.is_float(T_guess):
        msg = f"A valid initial guess must be provided for density and temperature. Received: rho_guess={rho_guess}, T_guess={T_guess}."
        raise ValueError(msg)

    # Define problem (find root of temperature-density residual)
    if calculation_type == "equilibrium":
        function_handle = lambda rho, T: compute_properties_coolprop(
            abstract_state=abstract_state,
            input_type=CP.DmassT_INPUTS,
            prop_1=rho,
            prop_2=T,
            generalize_quality=generalize_quality,
            supersaturation=supersaturation,
        )
    elif calculation_type == "metastable":
        function_handle = lambda rho, T: compute_properties_metastable_rhoT(
            abstract_state=abstract_state,
            rho=rho,
            T=T,
            generalize_quality=generalize_quality,
            supersaturation=supersaturation,
        )
    else:
        msg = f"Invalid calculation type '{calculation_type}'. Valid options are: 'equilibrium' and 'metastable'"
        raise ValueError(msg)

    # Define the problem (find root of temperature-density residual)
    rho_crit = abstract_state.rhomass_critical()
    T_crit = abstract_state.T_critical()
    problem = _FlashCalculationResidual(
        prop_1=prop_1,
        prop_1_value=prop_1_value,
        prop_2=prop_2,
        prop_2_value=prop_2_value,
        function_handle=function_handle,
        rho_scale=rho_crit,
        T_scale=T_crit,
    )

    # Define root-finding solver object
    solver = psv.NonlinearSystemSolver(
        problem,
        method=solver_algorithm,
        tolerance=solver_tolerance,
        max_iterations=solver_max_iterations,
        print_convergence=print_convergence,
        update_on="function",
    )

    # Define initial guess and solve the problem
    # x0 = np.asarray([rho_guess, T_guess])
    x0_reduced = np.asarray([rho_guess / rho_crit, T_guess / T_crit])
    xf_reduced = solver.solve(x0_reduced)
    rho, T = xf_reduced[0] * rho_crit, xf_reduced[1] * T_crit

    # Check if solver converged
    if not solver.success:
        msg = f"Property calculation did not converge for calculation_type={calculation_type}.\n{solver.message}"
        raise ValueError(msg)

    props = problem.compute_properties(rho, T)
    props["residual"]  = np.linalg.norm(problem.residual(xf_reduced))
    return props


class _FlashCalculationResidual(psv.NonlinearSystemProblem):
    """Class to compute the residual of property calculations"""

    def __init__(self, prop_1, prop_1_value, prop_2, prop_2_value, function_handle, rho_scale, T_scale):
        self.prop_1 = prop_1
        self.prop_2 = prop_2
        self.prop_1_value = prop_1_value
        self.prop_2_value = prop_2_value
        self.compute_properties = function_handle
        self.rho_scale = rho_scale
        self.T_scale = T_scale

    def residual(self, x):
        # Ensure x can be indexed and contains exactly two elements
        if not hasattr(x, "__getitem__") or len(x) != 2:
            msg = f"Input x={x} must be a list, tuple or numpy array containing exactly two elements: density and temperature."
            raise ValueError(msg)

        # Unscale rho-T and compute properties
        rho = x[0] * self.rho_scale
        T = x[1] * self.T_scale
        props = self.compute_properties(rho, T)

        # Compute residual
        def compute_residual(prop_name, target_value):
            value = props[prop_name]
            if prop_name == "Q":
                return value - target_value
            else:
                return 1.0 - value / target_value

        res_1 = compute_residual(self.prop_1, self.prop_1_value)
        res_2 = compute_residual(self.prop_2, self.prop_2_value)

        return np.asarray([res_1, res_2])


def blend_properties(
    phase_change,
    props_equilibrium,
    props_metastable,
    blending_variable,
    blending_onset,
    blending_width,
):
    """
    Calculate blending between equilibrum and metastable fluid properties

    Parameters
    ----------
    phase_change : str
        The type of phase change (e.g., 'condensation', 'evaporation', 'flashing', 'cavitation'). Cavitation, flashing, and evaporation do the same calculations, they are aliases added for convenience.
    props_equilibrium : dict
        The equilibrium properties.
    props_metastable : dict
        The metastable properties.
    blending_variable : str
        The variable used for blending.
    blending_onset : float
        The onset value for blending.
    blending_width : float
        The width value for blending.

    Returns
    -------
    dict
        Blended thermodynamic properties.
    """

    # Map aliases to their respective phase change
    normalized_phase_change = PHASE_CHANGE_ALIASES.get(phase_change)

    # Validate the normalized phase change
    if normalized_phase_change is None:
        msg = (
            f"Invalid value for phase_change='{phase_change}'. "
            f"Valid values are: {', '.join(PHASE_CHANGE_ALIASES.keys())}."
        )
        raise ValueError(msg)

    # Calculate the value of "x" based on the normalized phase change
    if normalized_phase_change == "condensation":
        x = 1 + (props_equilibrium[blending_variable] - blending_onset) / blending_width
    elif normalized_phase_change == "evaporation":
        x = 1 - (props_equilibrium[blending_variable] - blending_onset) / blending_width

    # Calculate the blending factor sigma
    sigma = math.sigmoid_smoothstep(x)
    # sigma = math.sigmoid_smootherstep(x)

    # Blend properties
    props_blended = {}
    for key in props_equilibrium.keys():
        prop_equilibrium = props_equilibrium.get(key, np.nan)
        prop_metastable = props_metastable.get(key, np.nan)
        if utils.is_numeric(prop_equilibrium) and utils.is_numeric(prop_metastable):
            props_blended[key] = prop_equilibrium * (1 - sigma) + prop_metastable * sigma
        else:
            props_blended[key] = None
            
    # Add additional properties
    props_blended["x"] = x
    props_blended["sigma"] = sigma

    return props_blended


# ------------------------------------------------------------------------------------ #
# Additional property calculations
# ------------------------------------------------------------------------------------ #

# def calculate_generalized_quality(abstract_state, alpha=10):
#     r"""
#     Calculate the generalized quality of a fluid, extending the quality calculation beyond the two-phase region if necessary.

#     Below the critical temperature, the quality is calculated from the specific volume of the saturated liquid and vapor states.
#     Above the critical temperature, an artificial two-phase region is defined around the critical density line to allow for a finite-width quality computation.

#     The quality, :math:`Q`, is given by:

#     .. math::

#         Q = \frac{v - v(T, Q=0)}{v(T, Q=1) - v(T, Q=0)}

#     where :math:`v=1/\rho` is the specific volume and :math:`T` is the temperature.

#     Additionally, this function applies smoothing limiters to ensure the quality is bounded between -1 and 2.
#     These limiters prevent the quality value from taking arbitrarily large values, enhancing stability and robustness of downstream calculation.
#     The limiters use the `logsumexp` method for smooth transitions, with a specified alpha value controlling the smoothness.

#     Parameters
#     ----------
#     abstract_state : CoolProp.AbstractState
#         CoolProp abstract state of the fluid.

#     alpha : float
#         Smoothing factor of the quality-calculation limiters.

#     Returns
#     -------
#     float
#         The calculated quality of the fluid.
#     """
#     # Instantiate new abstract state to compute saturation properties without changing the state of the class
#     AS = abstract_state
#     fluids = AS.fluid_names()  # AS.name does not work well for REFPROP backend
#     if len(fluids) != 1:
#         raise ValueError(f"Expected one fluid, got {fluids}")
#     cloned_AS = CP.AbstractState(AS.backend_name(), fluids[0])

#     # Extend quality calculation beyond the two-phase region
#     # Checking if subcritical using temperature works better than with pressure
#     if abstract_state.T() < abstract_state.T_critical():
#         # Saturated liquid
#         cloned_AS.update(CP.QT_INPUTS, 0.00, abstract_state.T())
#         rho_liq = cloned_AS.rhomass()

#         # Saturated vapor
#         cloned_AS.update(CP.QT_INPUTS, 1.00, abstract_state.T())
#         rho_vap = cloned_AS.rhomass()

#     else:
#         # For states at or above the critical temperature, the concept of saturation states is not applicable
#         # Instead, an artificial two-phase region is created around the pseudo-critical density line (line of critical density)
#         # The width of the artificial two-phase region is assumed to increase linearly with temperature

#         # Rename properties
#         T = abstract_state.T()
#         T_crit = abstract_state.T_critical()
#         rho_crit = abstract_state.rhomass_critical()

#         # Define pseudocritical region
#         T_hat = 1.5 * T_crit
#         rho_hat_liq = 1.1 * rho_crit
#         rho_hat_vap = 0.9 * rho_crit
#         rho_liq = rho_crit + (rho_hat_liq - rho_crit) * (T - T_crit) / (T_hat - T_crit)
#         rho_vap = rho_crit + (rho_hat_vap - rho_crit) * (T - T_crit) / (T_hat - T_crit)

#     # Compute quality according to definition
#     rho = abstract_state.rhomass()
#     quality = (1 / rho - 1 / rho_liq) / (1 / rho_vap - 1 / rho_liq + 1e-6)

#     # Apply smoothing limiters so that the quality is bounded between [-1, 2]
#     # The margin is defined as delta_Q=1 to the left of Q=0 and to the right of Q=1
#     quality = math.smooth_minimum(quality, +2, method="logsumexp", alpha=alpha)
#     quality = math.smooth_maximum(quality, -1, method="logsumexp", alpha=alpha)

#     return quality.item()

def calculate_generalized_quality(abstract_state, alpha=10):
    r"""
    Calculate the generalized quality of a fluid using specific entropy, extending the quality computation beyond
    the two-phase region by extrapolating entropy values.

    The quality, :math:`Q`, is given by:

    .. math::

        Q = \frac{s - s(T, Q=0)}{h(T, Q=1) - s(T, Q=0)}

    where :math:`s` is the specific enthalpy and :math:`T` is the temperature.

    Smoothing limiters keep the quality in the range [-1, 2] using a `logsumexp` formulation.

    Parameters
    ----------
    abstract_state : CoolProp.AbstractState
        CoolProp abstract state of the fluid.

    alpha : float
        Smoothing factor of the quality-calculation limiters.

    Returns
    -------
    float
        The calculated quality of the fluid.
    """
    # Instantiate new abstract state to compute saturation properties without changing the state of the class
    AS = abstract_state
    fluids = AS.fluid_names()  # AS.name does not work well for REFPROP backend
    if len(fluids) != 1:
        raise ValueError(f"Expected one fluid, got {fluids}")
    cloned_AS = CP.AbstractState(AS.backend_name(), fluids[0])

    # Extend quality calculation beyond the two-phase region
    # Checking if subcritical using temperature works better than with pressure
    if AS.T() < AS.T_critical():
        # Subcritical: use actual saturation properties
        cloned_AS.update(CP.QT_INPUTS, 0.0, AS.T())
        s_liq = cloned_AS.smass()

        cloned_AS.update(CP.QT_INPUTS, 1.0, AS.T())
        s_vap = cloned_AS.smass()
    else:
        # Supercritical: extrapolate enthalpy bounds from critical point
        # For states at or above the critical temperature, the concept of saturation states is not applicable
        # Instead, an artificial two-phase region is created around the pseudo-critical density line (line of critical density)
        # The width of the artificial two-phase region is assumed to increase linearly with temperature

        T = AS.T()
        rho_crit, T_crit = AS.rhomass_critical(), AS.T_critical()
        cloned_AS.update(CP.DmassT_INPUTS, rho_crit, T_crit)
        s_crit = cloned_AS.smass()

        T_hat = 1.5 * T_crit
        s_hat_liq = 1.1 * s_crit
        s_hat_vap = 0.9 * s_crit
        s_liq = s_crit + (s_hat_liq - s_crit) * (T - T_crit) / (T_hat - T_crit)
        s_vap = s_crit + (s_hat_vap - s_crit) * (T - T_crit) / (T_hat - T_crit)

    # Compute quality from enthalpy
    s = AS.smass()
    quality = (s - s_liq) / (s_vap - s_liq + 1e-6)

    # Apply smooth limiting between [-1, 2]
    quality = math.smooth_minimum(quality, 2, method="logsumexp", alpha=alpha)
    quality = math.smooth_maximum(quality, -1, method="logsumexp", alpha=alpha)

    return quality.item()



def calculate_superheating(abstract_state):
    r"""
    Calculate the degree of superheating for a given CoolProp abstract state.

    This function computes the superheating of a fluid using the CoolProp library's
    abstract state class. It handles both subcritical and supercritical conditions
    to provide a measure of superheating for any thermodynamic state. This results in
    a continuous variation of superheating in the two-phase region, which is necessary
    to achieve in reliable convergence of systems of equations and optimization problems
    involving the degree of superheating.

    Calculation cases:
        - Physically meaningful superheating for subcritical states in the vapor region:

        .. math::

          \Delta T = T - T(p, Q=1) \quad \text{for} \quad h > h(p, Q=1)

        - Artificial superheating for subcritical states in the liquid and two-phase regions:

        .. math::

          \Delta T = \frac{h - h(p, Q=1)}{c_p(p, Q=1)}

        - Artificial superheating for supercritical states defined using the critical density line:

        .. math::

          \Delta T = T - T(p, \rho_{\text{crit}})

    Parameters
    ----------
    abstract_state : CoolProp.AbstractState
        The abstract state of the fluid for which the superheating is to be calculated.

    Returns
    -------
    float
        The degree of superheating in Kelvin.

    Examples
    --------
    >>> import CoolProp as CP
    >>> abstract_state = CP.AbstractState("HEOS", "water")
    >>> abstract_state.update(CP.PT_INPUTS, 101325, 120 + 273.15)
    >>> superheating = calculate_superheating(abstract_state)
    >>> print(f"Superheating is {superheating:+0.3f} K")
    Superheating is +20.026 K

    >>> abstract_state = CP.AbstractState("HEOS", "water")
    >>> abstract_state.update(CP.PQ_INPUTS, 101325, 0.95)
    >>> superheating = calculate_superheating(abstract_state)
    >>> print(f"Superheating is {superheating:+0.3f} K")
    Superheating is -54.244 K
    """
    # Instantiate new abstract state to compute saturation properties without changing the state of the class
    AS = abstract_state
    fluids = AS.fluid_names()  # AS.name does not work well for REFPROP backend
    if len(fluids) != 1:
        raise ValueError(f"Expected one fluid, got {fluids}")
    sat_AS = CP.AbstractState(AS.backend_name(), fluids[0])

    # Compute triple pressure
    sat_AS.update(CP.QT_INPUTS, 1.00, AS.Ttriple())
    p_triple = sat_AS.p()

    # Check if the pressure is below the critical pressure of the fluid
    if AS.p() < AS.p_critical():

        # Compute triple pressure (needed to avoid error at low pressure)
        sat_AS.update(CP.QT_INPUTS, 1.00, AS.Ttriple())
        p_triple = sat_AS.p()

        # Set the saturation state of the fluid at the given pressure
        sat_AS.update(CP.PQ_INPUTS, max(p_triple, AS.p()), 1.00)

        # Check if the fluid is in the two-phase or liquid regions
        if AS.hmass() < sat_AS.hmass():
            # Below the vapor saturation enthalpy, define superheating as the normalized difference in enthalpy
            # The normalization is done using the specific heat capacity at saturation (cp)
            # This provides a continuous measure of superheating, even in the two-phase region
            superheating = (AS.hmass() - sat_AS.hmass()) / sat_AS.cpmass()
        else:
            # Outside the two-phase region, superheating is the difference in temperature
            # from the saturation temperature at the same pressure
            superheating = AS.T() - sat_AS.T()
    else:
        # For states at or above the critical pressure, the concept of saturation temperature is not applicable
        # Instead, use a 'pseudo-critical' state for comparison, where the density is set to the critical density
        # but the pressure is the same as the state of interest
        rho_crit = AS.rhomass_critical()
        sat_AS.update(CP.DmassP_INPUTS, rho_crit, AS.p())

        # Define superheating as the difference in enthalpy from this 'pseudo-critical' state
        # This approach extends the definition of superheating to conditions above the critical pressure
        superheating = AS.T() - sat_AS.T()

    return superheating


def calculate_subcooling(abstract_state):
    r"""
    Calculate the degree of subcooling for a given CoolProp abstract state.

    This function computes the subcooling of a fluid using the CoolProp library's
    abstract state class. It handles both subcritical and supercritical conditions
    to provide a measure of subcooling for any thermodynamic state. This results in
    a continuous variation of subcooling in the two-phase region, which is necessary
    to achieve reliable convergence of systems of equations and optimization problems
    involving the degree of subcooling.

    Calculation cases:
        - Physically meaningful subcooling for subcritical states in the liquid region:

        .. math::

          \Delta T = T(p, Q=0) - T \quad \text{for} \quad h < h(p, Q=0)

        - Artificial subcooling for subcritical states in the vapor and two-phase regions:

        .. math::

          \Delta T = \frac{h(p, Q=0) - h}{c_p(p, Q=0)}

        - Artificial subcooling for supercritical states defined using the critical density line:

        .. math::

          \Delta T = T(p, \rho_{\text{crit}}) - T

    Parameters
    ----------
    abstract_state : CoolProp.AbstractState
        The abstract state of the fluid for which the subcooling is to be calculated.

    Returns
    -------
    float
        The degree of subcooling in Kelvin.

    Examples
    --------
    >>> import CoolProp as CP
    >>> abstract_state = CP.AbstractState("HEOS", "water")
    >>> abstract_state.update(CP.PT_INPUTS, 101325, 25+273.15)
    >>> subcooling = bpy.calculate_subcooling(abstract_state)
    >>> print(f"Subcooling is {subcooling:+0.3f} K")
    Subcooling is +74.974 K

    >>> abstract_state = CP.AbstractState("HEOS", "water")
    >>> abstract_state.update(CP.PQ_INPUTS, 101325, 0.05)
    >>> subcooling = bpy.calculate_subcooling(abstract_state)
    >>> print(f"Subcooling is {subcooling:+0.3f} K")
    Subcooling is -26.763 K
    """

    # Instantiate new abstract state to compute saturation properties without changing the state of the class
    AS = abstract_state
    fluids = AS.fluid_names()  # AS.name does not work well for REFPROP backend
    if len(fluids) != 1:
        raise ValueError(f"Expected one fluid, got {fluids}")
    sat_AS = CP.AbstractState(AS.backend_name(), fluids[0])

    # Check if the pressure is below the critical pressure of the fluid
    if AS.p() < AS.p_critical():

        # Compute triple pressure (needed to avoid error at low pressure)
        sat_AS.update(CP.QT_INPUTS, 0.00, AS.Ttriple())
        p_triple = sat_AS.p()

        # Set the saturation state of the fluid at the given pressure
        sat_AS.update(CP.PQ_INPUTS, max(p_triple, AS.p()), 0.00)

        # Check if the fluid is in the two-phase or vapor regions
        if AS.hmass() > sat_AS.hmass():
            # Above the liquid saturation enthalpy, define superheating as the normalized difference in enthalpy
            # The normalization is done using the specific heat capacity at saturation (cp)
            # This provides a continuous measure of superheating, even in the two-phase region
            subcooling = (sat_AS.hmass() - AS.hmass()) / sat_AS.cpmass()
        else:
            # Outside the two-phase region, superheating is the difference in temperature
            # from the saturation temperature at the same pressure
            subcooling = sat_AS.T() - AS.T()
    else:
        # For states at or above the critical pressure, the concept of saturation temperature is not applicable
        # Instead, use a 'pseudo-critical' state for comparison, where the density is set to the critical density
        # but the pressure is the same as the state of interest
        rho_crit = AS.rhomass_critical()
        sat_AS.update(CP.DmassP_INPUTS, rho_crit, AS.p())

        # Define superheating as the difference in enthalpy from this 'pseudo-critical' state
        # This approach extends the definition of superheating to conditions above the critical pressure
        subcooling = sat_AS.T() - AS.T()

    return subcooling


def calculate_supersaturation(abstract_state, props):
    r"""
    Evaluate degree of supersaturation and supersaturation ratio.

    The supersaturation degree is defined as the actual temperature minus the saturation temperature at the corresponding pressure:

    .. math::
        \Delta T = T - T_{\text{sat}}(p)

    The supersaturation ratio is defined as the actual pressure divided by the saturation pressure at the corresponding temperature:

    .. math::
        S = \frac{p}{p_{\text{sat}}(T)}

    The metastable liquid and metastable vapor regions are illustrated in the pressure-temperature diagram.
    In the metastable liquid region, :math:`\Delta T > 0` and :math:`S < 1`, indicating that the liquid temperature
    is higher than the equilibrium temperature and will tend to evaporate. Conversely, in the metastable vapor region,
    :math:`\Delta T < 0` and :math:`S > 1`, indicating that the vapor temperature is lower than the equilibrium temperature
    and will tend to condense.

    .. image:: /_static/metastable_regions_CO2.svg
        :alt: Pressure-density diagram and spinodal points for carbon dioxide.


    .. note::

        Supersaturation properties are only computed for subcritical pressures and temperatures.
        If the fluid is in the supercritical region, the function will return NaN for the supersaturation properties.


    Parameters
    ----------
    abstract_state : CoolProp.AbstractState
        The abstract state of the fluid for which the properties are to be calculated.
    props : dict
        Dictionary to store the computed properties.

    Returns
    -------
    dict
        Thermodynamic properties including supersaturation properties

    """
    # Compute triple pressure
    AS = abstract_state
    fluids = AS.fluid_names()  # AS.name does not work well for REFPROP backend
    if len(fluids) != 1:
        raise ValueError(f"Expected one fluid, got {fluids}")
    AS = CP.AbstractState(AS.backend_name(), fluids[0])
    AS.update(CP.QT_INPUTS, 1.00, AS.Ttriple())
    p_triple = AS.p()

    # Compute supersaturation for subcritical states
    if AS.Ttriple() < props["T"] < AS.T_critical():
        AS.update(CP.QT_INPUTS, 0.00, props["T"])
        props["p_saturation"] = AS.p()
        props["supersaturation_ratio"] = props["p"] / AS.p()
    else:
        props["p_saturation"] = np.nan
        props["supersaturation_ratio"] = np.nan

    if p_triple < props["p"] < AS.p_critical():
        AS.update(CP.PQ_INPUTS, props["p"], 0.00)
        props["T_saturation"] = AS.T()
        props["supersaturation_degree"] = props["T"] - AS.T()
    else:
        props["T_saturation"] = np.nan
        props["supersaturation_degree"] = np.nan

    return props


# ------------------------------------------------------------------------------------ #
# Properties of a two-component mixture (e.g., water and nitrogen)
# ------------------------------------------------------------------------------------ #

def calculate_mixture_properties(props_1, props_2, y_1, y_2):
    """
    Calculate the thermodynamic properties of the mixture.

    TODO: add model equations and explanation

    Parameters
    ----------
    state_1 : dict
        Thermodynamic properties of fluid 1.
    state_2 : dict
        Thermodynamic properties of fluid 2.
    y_1 : float
        Mass fraction of fluid 1.
    y_2 : float
        Mass fraction of fluid 2.

    Returns
    -------
    mixture_properties : dict
        A dictionary containing the mixture's properties.
    """

    # Validate mass fractions
    if not utils.is_float(y_1) or not utils.is_float(y_2) or np.abs(y_1 + y_2 - 1) > 1e-12:
        raise ValueError(f"The mass fractions must be floats and their sum must equal 1. Received y_1={y_1} and y_2={y_2}.")

    # Mass-averaged properties
    cv = y_1 * props_1['cvmass'] + y_2 * props_2['cvmass']
    cp = y_1 * props_1['cpmass'] + y_2 * props_2['cpmass']
    umass = y_1 * props_1['umass'] + y_2 * props_2['umass']
    hmass = y_1 * props_1['hmass'] + y_2 * props_2['hmass']
    smass = y_1 * props_1['smass'] + y_2 * props_2['smass']    
    
    # Enthalpy derivative for the two-component barotropic model
    alphaP_1 = props_1['isobaric_expansion_coefficient']
    alphaP_2 = props_2['isobaric_expansion_coefficient']
    dhdp_T_1 = (1 - props_1['T'] * alphaP_1) / props_1['rho']
    dhdp_T_2 = (1 - props_2['T'] * alphaP_2) / props_2['rho']
    dhdp_T = y_1 * dhdp_T_1 + y_2 * dhdp_T_2

    # Compute volume fractions and void fraction
    rho = 1 / (y_1 / props_1['rho'] + y_2 / props_2['rho'])
    vol_1 = y_1 * rho/props_1['rhomass']
    vol_2 = y_2 * rho/props_2['rhomass']
    quality_mass = y_1 if props_1['rho'] < props_2['rho'] else vol_2
    quality_volume = vol_1 if props_1['rho'] < props_2['rho'] else vol_2
    
    # Isothermal compressibility
    betaT_1 = props_1["isothermal_compressibility"]
    betaT_2 = props_2["isothermal_compressibility"]
    isothermal_compressibility = vol_1 * betaT_1 + vol_2 * betaT_2
    isothermal_bulk_modulus = 1 / isothermal_compressibility

    # Isentropic compressibility and speed of sound (Wood's formula)
    betaS_1 = props_1["isentropic_compressibility"]
    betaS_2 = props_2["isentropic_compressibility"]
    isentropic_compressibility = vol_1 * betaS_1 + vol_2 * betaS_2
    isentropic_bulk_modulus = 1 / isentropic_compressibility
    speed_sound = np.sqrt(isentropic_bulk_modulus / rho)

    # Transport properties as volume averages
    conductivity = vol_1 * props_1['conductivity'] + vol_2 * props_2['conductivity']
    viscosity = vol_1 * props_1['viscosity'] + vol_2 * props_2['viscosity']

    # Group up mixture properties
    props = {
        "mass_frac_1": y_1,
        "mass_frac_2": y_2,
        "vol_frac_1": vol_1,
        "vol_frac_2": vol_2,
        "mixture_ratio": y_1 / y_2,
        "quality_mass": quality_mass,
        "quality_volume": quality_volume,
        "p": props_1['p'],
        "T": props_1['T'],
        "rhomass": rho,
        "hmass": hmass,
        "umass": umass,
        "smass": smass,
        "cvmass": cv,
        "cpmass": cp,
        "isothermal_compressibility": isothermal_compressibility,
        "isothermal_bulk_modulus": isothermal_bulk_modulus,
        "isentropic_compressibility": isentropic_compressibility,
        "isentropic_bulk_modulus": isentropic_bulk_modulus,
        "speed_sound": speed_sound,
        "conductivity": conductivity,
        "viscosity": viscosity,
        "viscosity_kinematic": viscosity / rho,
        "dhdp_T": dhdp_T,
    }

    # Add properties as aliases (if property exists)
    for key, value in PROPERTY_ALIAS.items():
        props[key] = props.get(value, np.nan)

    return props
