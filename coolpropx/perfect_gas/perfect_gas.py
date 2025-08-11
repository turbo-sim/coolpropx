from .. import fluid_properties as fp
from ..core_calculations import PROPERTY_ALIAS
from ..jax_import import jax, jnp, JAX_AVAILABLE
from functools import partial

# ----------------------------------------------------------------------------- #
# Constant generation and Sutherland estimation
# ----------------------------------------------------------------------------- #
def get_perfect_gas_constants(fluid_name, T_ref, P_ref, dT=100.0, display=False):
    """
    Compute perfect gas constants from a real-fluid model at a reference state,
    and estimate Sutherland constants using offset temperatures.

    Parameters
    ----------
    Fluid : object
        Fluid object with a method `get_state(input_type: str, val1: float, val2: float) -> dict`
        returning thermodynamic properties including 'R', 'cp', 'cv', 'mu', 'k', and optionally 's'.
    fluid_name : str
        Name of the fluid (for logging or selection).
    T_ref : float
        Reference temperature in kelvin [K].
    P_ref : float
        Reference pressure in pascal [Pa].
    dT : float, optional
        Temperature offset used to evaluate `mu` and `k` at T_ref ± dT for
        Sutherland constant estimation [K]. Default is 100.0.

    Returns
    -------
    dict
        Dictionary containing perfect gas constants:
        - 'R': specific gas constant [J/(kg·K)]
        - 'gamma': specific heat ratio
        - 'T_ref': reference temperature [K]
        - 'P_ref': reference pressure [Pa]
        - 's_ref': reference entropy [J/(kg·K)]
        - 'mu_ref': dynamic viscosity at T_ref [Pa·s]
        - 'S_mu': Sutherland constant for viscosity [K]
        - 'k_ref': thermal conductivity at T_ref [W/(m·K)]
        - 'S_k': Sutherland constant for thermal conductivity [K]

    Raises
    ------
    ValueError
        If required properties are missing or invalid.
    ZeroDivisionError
        If the Sutherland estimation becomes numerically unstable.
    """
    # Calculate fluid constants at the reference pressure and temperature
    fluid = fp.Fluid(name=fluid_name, backend="HEOS")
    state = fluid.get_state(fp.PT_INPUTS, P_ref, T_ref)
    R = fp.GAS_CONSTANT/fluid.abstract_state.molar_mass()
    cp = state["cp"]
    cv = state["cv"]
    gamma = cp / cv
    s_ref = 0.0
    mu_ref = state["mu"]
    k_ref = state["k"]

    # Estimate Sutherland constants from values at T_ref ± dT
    T2 = T_ref + dT
    state2 = fluid.get_state(fp.PT_INPUTS, P_ref, T2)
    S_mu = estimate_sutherland_constant(T_ref, mu_ref, T2, state2["mu"])
    S_k = estimate_sutherland_constant(T_ref, k_ref, T2, state2["k"])

    constants = {
        "R": R,
        "gamma": gamma,
        "T_ref": T_ref,
        "P_ref": P_ref,
        "s_ref": s_ref,
        "k_ref": k_ref,
        "mu_ref": mu_ref,
        "S_k": S_k,
        "S_mu": S_mu,
    }

    if display:
        varname = f"GAS_CONSTANTS_{fluid_name.upper()}"
        print()
        print(f"# Perfect gas properties for fluid \"{fluid_name}\" at pressure = {P_ref} Pa and temperature = {T_ref} K")
        print(f"{varname} = {{")
        for key, value in constants.items():
            print(f"    \"{key}\": {value:0.12e},")
        print("}")

    return constants


def estimate_sutherland_constant(T1, mu1, T2, mu2):
    """
    Estimate the Sutherland constant [K] from two values of a transport property 
    (viscosity or thermal conductivity) at two temperatures, assuming a Sutherland-like law:
    
        mu(T) = mu_ref * (T / T_ref)^{3/2} * (T_ref + S) / (T + S)

    Parameters
    ----------
    T1, T2 : float
        Temperatures in kelvin [K].
    mu1, mu2 : float
        Transport property values at T1 and T2 (e.g. viscosity in Pa·s or conductivity in W/m·K).

    Returns
    -------
    float
        Estimated Sutherland constant in kelvin [K].
    """
    r = (mu2 / mu1) * (T2 / T1) ** (-1.5)
    return (r * T2 - T1) / (1 - r)


# ----------------------------------------------------------------------------- #
# Example constant sets (code-generated)
# ----------------------------------------------------------------------------- #

# Perfect gas properties for fluid "air" at pressure = 101325 Pa and temperature = 298.15 K
GAS_CONSTANTS_AIR = {
    "R": 2.870473936889e+02,
    "gamma": 1.401766304726e+00,
    "T_ref": 2.981500000000e+02,
    "P_ref": 1.013250000000e+05,
    "s_ref": 0.000000000000e+00,
    "k_ref": 2.624693131891e-02,
    "mu_ref": 1.844808216200e-05,
    "S_k": 1.663154111359e+02,
    "S_mu": 1.202103108299e+02,
}

# Perfect gas properties for fluid "water" at pressure = 1000.0 Pa and temperature = 298.15 K
GAS_CONSTANTS_WATER = {
    "R": 4.615229593032e+02,
    "gamma": 1.329030966936e+00,
    "T_ref": 2.981500000000e+02,
    "P_ref": 1.000000000000e+03,
    "s_ref": 0.000000000000e+00,
    "k_ref": 1.843390523303e-02,
    "mu_ref": 9.706485314005e-06,
    "S_k": 9.116455179041e+02,
    "S_mu": 4.849654253407e+02,
}


# ----------------------------------------------------------------------------- #
# Helper functions to calculate individual fluid properties
# ----------------------------------------------------------------------------- #

def specific_heat(constants):
    R = constants["R"]
    gamma = constants["gamma"]
    cp = (gamma * R) / (gamma - 1)
    cv = cp / gamma
    return cp, cv

def temperature_from_h(h, constants):
    cp, _ = specific_heat(constants)
    return jnp.maximum(1.0, h / cp)

def temperature_from_Ps(P, s, constants):
    R = constants["R"]
    gamma = constants["gamma"]
    T_ref = constants["T_ref"]
    P_ref = constants["P_ref"]
    s_ref = constants["s_ref"]
    cp, _ = specific_heat(constants)
    exponent = ((s - s_ref) + (R * jnp.log(P / P_ref))) / cp
    return jnp.maximum(1.0, T_ref * jnp.exp(exponent))

def temperature_from_rhoP(rho, p, constants):
    R = constants["R"]
    return jnp.maximum(1.0, p / (rho * R))

def viscosity_from_T(T, constants):
    mu_ref = constants["mu_ref"]
    T_ref = constants["T_ref"]
    S_mu = constants["S_mu"]
    return mu_ref * ((T / T_ref) ** 1.5) * ((T_ref + S_mu) / (T + S_mu))

def conductivity_from_T(T, constants):
    k_ref = constants["k_ref"]
    T_ref = constants["T_ref"]
    S_k = constants["S_k"]
    return k_ref * ((T / T_ref) ** 1.5) * ((T_ref + S_k) / (T + S_k))

def speed_of_sound_from_T(T, constants):
    R = constants["R"]
    gamma = constants["gamma"]
    return jnp.sqrt(gamma * R * T)

def entropy_from_PT(P, T, constants):
    R = constants["R"]
    T_ref = constants["T_ref"]
    P_ref = constants["P_ref"]
    s_ref = constants["s_ref"]
    cp, _ = specific_heat(constants)
    return s_ref + (cp * jnp.log(T / T_ref)) - (R * jnp.log(P / P_ref))

def entropy_from_rhoP(rho, P, constants):
    R = constants["R"]
    T = P / (rho * R)
    return entropy_from_PT(P, T, constants)


# ----------------------------------------------------------------------------- #
# Helper functions to calculate full fluid states
# ----------------------------------------------------------------------------- #

def calculate_properties_PT(P, T, constants):
    T = jnp.maximum(T, 0.1)
    P = jnp.maximum(P, 0.1)
    R = constants["R"]
    rho = P / (R * T)
    cp, _ = specific_heat(constants)
    h = cp * T
    s = entropy_from_PT(P, T, constants)
    return assemble_properties(T, P, rho, h, s, constants)

def calculate_properties_hs(h, s, constants):
    cp, _ = specific_heat(constants)
    T = temperature_from_h(h, constants)
    R = constants["R"]
    T_ref = constants["T_ref"]
    P_ref = constants["P_ref"]
    s_ref = constants["s_ref"]
    P = P_ref * jnp.exp(((cp * jnp.log(T / T_ref)) - (s - s_ref)) / R)
    rho = P / (R * T)
    return assemble_properties(T, P, rho, h, s, constants)

def calculate_properties_hP(h, P, constants):
    T = temperature_from_h(h, constants)
    R = constants["R"]
    rho = P / (R * T)
    s = entropy_from_PT(P, T, constants)
    return assemble_properties(T, P, rho, h, s, constants)

def calculate_properties_Ps(P, s, constants):
    T = temperature_from_Ps(P, s, constants)
    R = constants["R"]
    rho = P / (R * T)
    cp, _ = specific_heat(constants)
    h = cp * T
    return assemble_properties(T, P, rho, h, s, constants)

def calculate_properties_rhoh(rho, h, constants):
    T = temperature_from_h(h, constants)
    R = constants["R"]
    P = rho * R * T
    s = entropy_from_rhoP(rho, P, constants)
    return assemble_properties(T, P, rho, h, s, constants)

def calculate_properties_rhop(rho, P, constants):
    cp, _ = specific_heat(constants)
    T = temperature_from_rhoP(rho, P, constants)
    h = cp * T
    s = entropy_from_rhoP(rho, P, constants)
    return assemble_properties(T, P, rho, h, s, constants)

def assemble_properties(T, P, rho, h, s, constants):
    R = constants["R"]
    gamma = constants["gamma"]
    props = {
        "T": T,
        "p": P,
        "d": rho,
        "rho": rho,
        "h": h,
        "s": s,
        "mu": viscosity_from_T(T, constants),
        "k": conductivity_from_T(T, constants),
        "a": speed_of_sound_from_T(T, constants),
        "gamma": gamma,
        "cp": gamma * R / (gamma - 1),
        "cv": R / (gamma - 1),
        "Z": 1.0,
        "gruneisen": gamma - 1
    }
    for original, alias in PROPERTY_ALIAS.items():
        if original not in props:
            continue
        else:
            props[alias] = props[original]
            
    return props



# ----------------------------------------------------------------------------- #
# State evaluators (public API)
# ----------------------------------------------------------------------------- #

jit_PT   = jax.jit(lambda a,b,c: calculate_properties_PT(a,b,c))
jit_hs   = jax.jit(lambda a,b,c: calculate_properties_hs(a,b,c))
jit_hP   = jax.jit(lambda a,b,c: calculate_properties_hP(a,b,c))
jit_Ps   = jax.jit(lambda a,b,c: calculate_properties_Ps(a,b,c))
jit_rhoh = jax.jit(lambda a,b,c: calculate_properties_rhoh(a,b,c))
jit_rhop = jax.jit(lambda a,b,c: calculate_properties_rhop(a,b,c))

PROPERTY_CALCULATORS = {
    fp.PT_INPUTS: jit_PT,
    fp.HmassSmass_INPUTS: jit_hs,
    fp.HmassP_INPUTS: jit_hP,
    fp.PSmass_INPUTS: jit_Ps,
    fp.DmassHmass_INPUTS: jit_rhoh,
    fp.DmassP_INPUTS: jit_rhop,
}

def get_props(input_pair, prop1, prop2, constants):
    return PROPERTY_CALCULATORS[input_pair](prop1, prop2, constants)


# ----------------------------------------------------------------------------- #
# Gradient calculations (only used when JAX is installed)
# ----------------------------------------------------------------------------- #

def get_props_gradient(input_pair, constants, x, y, method="auto", eps_rel=1e-6):
    """
    Return (d/dx, d/dy) dictionaries of partials for perfect_gas_props at (x, y).

    method: "auto" (use JAX if available), "jax" (force JAX), or "fd" (finite diff).
    eps_rel: relative step size for finite differences.
    """
    # local wrapper: f(x, y) -> dict
    def f(a, b):
        return get_props(input_pair, a, b, constants)

    # choose method
    use_jax = (method == "jax") or (method == "auto" and JAX_AVAILABLE)
    if use_jax:
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX not available but method='jax' was requested.")
        dfdx = jax.jacfwd(lambda a, b: f(a, b), argnums=0)(x, y)
        dfdy = jax.jacfwd(lambda a, b: f(a, b), argnums=1)(x, y)
        return dfdx, dfdy

    # finite differences path
    x = jnp.asarray(x); y = jnp.asarray(y)
    sx = jnp.maximum(jnp.abs(x), 1.0)
    sy = jnp.maximum(jnp.abs(y), 1.0)
    dx = eps_rel * sx
    dy = eps_rel * sy

    fxp = f(x + dx, y)
    fxm = f(x - dx, y)
    fyp = f(x, y + dy)
    fym = f(x, y - dy)

    keys = fxp.keys()
    dfdx = {k: (fxp[k] - fxm[k]) / (2.0 * dx) for k in keys}
    dfdy = {k: (fyp[k] - fym[k]) / (2.0 * dy) for k in keys}
    return dfdx, dfdy

