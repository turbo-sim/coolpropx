import os
import pytest
import numpy as np
import coolpropx as cpx
import coolpropx.perfect_gas as pg

import jax
from scipy.optimize._numdiff import approx_derivative

# tolerances
RTOL = 1e-8
ATOL = 1e-12

# properties to compare
PROP_KEYS = ("T", "p", "d", "h", "s", "mu", "k", "a", "gamma")

# input types to exercise
INPUT_TYPES = [
    cpx.HmassSmass_INPUTS,
    cpx.HmassP_INPUTS,
    cpx.PSmass_INPUTS,
    cpx.DmassHmass_INPUTS,
    cpx.DmassP_INPUTS,
]

# reference cases (pressures in Pa, temperatures in K)
CASES = [
    {"id": "air-300K-100kPa",       "fluid": "air",      "T": 300.0, "p": 1e5},
    {"id": "nitrogen-300K-100kPa",  "fluid": "nitrogen", "T": 300.0, "p": 1e5},
    {"id": "water-300K-1kPa",       "fluid": "water",    "T": 300.0, "p": 1e3},
    {"id": "CO2-800K-10000kPa",     "fluid": "co2",     "T": 800.0, "p": 100e5},
]

CASE_IDS = [c["id"] for c in CASES]

@pytest.fixture(scope="module", params=CASES, ids=CASE_IDS)
def case(request):
    """Provides one fluid reference case at a time and its perfect-gas constants."""
    c = request.param
    # compute constants at the requested reference state
    constants = pg.get_perfect_gas_constants(c["fluid"], c["T"], c["p"], display=False)

    # sanity: force the baseline to exactly this PT
    ref = pg.get_props(cpx.PT_INPUTS, c["p"], c["T"], constants)

    return {
        "metadata": c,           # id, fluid, T, p
        "constants": constants,  # perfect gas constants
        "ref": ref,              # baseline properties at (p,T)
        "h": ref["h"],
        "s": ref["s"],
        "rho": ref["d"],
        "p": ref["p"],
        "T": ref["T"],
    }

@pytest.mark.parametrize("input_type", INPUT_TYPES, ids=INPUT_TYPES)
def test_perfect_gas_multi_reference(input_type, case):
    constants = case["constants"]
    ref_vals = case["ref"]
    cid = case["metadata"]["id"]

    # choose inputs for each solver
    if input_type == cpx.HmassSmass_INPUTS:
        v1, v2 = case["h"], case["s"]
    elif input_type == cpx.HmassP_INPUTS:
        v1, v2 = case["h"], case["p"]
    elif input_type == cpx.PSmass_INPUTS:
        v1, v2 = case["p"], case["s"]
    elif input_type == cpx.DmassHmass_INPUTS:
        v1, v2 = case["rho"], case["h"]
    elif input_type == cpx.DmassP_INPUTS:
        v1, v2 = case["rho"], case["p"]
    else:
        pytest.skip(f"unhandled input_type: {input_type}")

    test_vals = pg.get_props(input_type, v1, v2, constants)

    for k in PROP_KEYS:
        value_calc = np.asarray(test_vals[k])
        value_ref = np.asarray(ref_vals[k])
        if not np.allclose(value_calc, value_ref, rtol=RTOL, atol=ATOL):
            diff = value_calc - value_ref
            pytest.fail(
                f"[{cid}] {input_type}: property '{k}' mismatch\n"
                f"  ref  = {value_ref:.12e}\n"
                f"  calc = {value_calc:.12e}\n"
                f"  err  = {diff:.3e}\n"
                f"  rtol={RTOL}, atol={ATOL}"
            )




@pytest.mark.parametrize("input_type", INPUT_TYPES, ids=INPUT_TYPES)
def test_perfect_gas_derivatives(input_type, case):
    """Checks that JAX-computed derivatives match finite differences."""
    constants = case["constants"]
    cid = case["metadata"]["id"]

    # choose inputs for each solver
    if input_type == cpx.HmassSmass_INPUTS:
        v1, v2 = case["h"], case["s"]
    elif input_type == cpx.HmassP_INPUTS:
        v1, v2 = case["h"], case["p"]
    elif input_type == cpx.PSmass_INPUTS:
        v1, v2 = case["p"], case["s"]
    elif input_type == cpx.DmassHmass_INPUTS:
        v1, v2 = case["rho"], case["h"]
    elif input_type == cpx.DmassP_INPUTS:
        v1, v2 = case["rho"], case["p"]
    else:
        pytest.skip(f"unhandled input_type: {input_type}")

    # initial vector for derivative check
    x0 = np.array([float(v1), float(v2)], dtype=float)

    for k in PROP_KEYS:
        # differentiable wrapper to get property k
        def prop_fun(x):
            out = pg.get_props(input_type, x[0], x[1], constants)
            return out[k]

        # JAX forward-mode gradient
        grad_jax = np.array(jax.jacfwd(prop_fun)(x0), dtype=float)

        # finite-difference gradient
        grad_fd = approx_derivative(
            prop_fun, x0=x0, method="2-point", rel_step=1e-6
        )

        if not np.allclose(grad_jax, grad_fd, rtol=1e-5, atol=1e-5):
            diff = grad_jax - grad_fd
            pytest.fail(
                f"[{cid}] {input_type}: derivative mismatch for {k}\n"
                f"  grad_jax = {grad_jax}\n"
                f"  grad_fd  = {grad_fd}\n"
                f"  diff     = {diff}"
            )



if __name__ == "__main__":

    # Running pytest from this script
    # pytest.main([__file__])
    pytest.main([__file__, "-v"])
    # # pytest.main([__file__, "-vv"])



 