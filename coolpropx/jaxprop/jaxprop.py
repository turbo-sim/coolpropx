# coolpropx/jaxprop.py
from __future__ import annotations
from functools import partial
import math
import jax
import jax.numpy as jnp

# properties you need; keep this fixed
_WANTED = (
    "p",
    "T",
    "rho",
    "d",
    "u",
    "h",
    "s",
    "a",
    "gruneisen",
    "mu",
    "k",
    "cp",
    "cv",
    "gamma",
)

# output template for pure_callback (shapes/dtypes must be static)
_TEMPLATE = {k: jax.ShapeDtypeStruct((), jnp.float64) for k in _WANTED}


def _to_scalar64(x) -> float:
    """robust cast to finite float; otherwise NaN"""
    try:
        xf = float(x)
        return xf if math.isfinite(xf) else float("nan")
    except Exception:
        return float("nan")


def _props_py(input_state, p1, p2, fluid) -> dict[str, float]:
    """Runs on host (Python). Returns plain floats."""
    p1f = float(jnp.squeeze(p1))
    p2f = float(jnp.squeeze(p2))
    st = fluid.get_state(input_state, p1f, p2f).to_dict()
    return {k: _to_scalar64(st.get(k)) for k in _WANTED}


@partial(jax.custom_jvp, nondiff_argnums=(0, 3))
def get_props(input_state, prop1, prop2, fluid):
    """
    CoolProp-backed properties usable under jit/grad.

    Differentiable w.r.t. (prop1, prop2) via a custom JVP (finite differences).
    input_state and fluid are non-differentiable static args.
    """

    def _cb(p1, p2):
        # returns dict[str, float]; pure_callback will map to jnp scalars using _TEMPLATE
        return _props_py(input_state, p1, p2, fluid)

    return jax.pure_callback(_cb, _TEMPLATE, prop1, prop2)


@get_props.defjvp
def _get_props_jvp(input_state, fluid, primals, tangents):
    p1, p2 = primals
    p1_dot, p2_dot = tangents

    # relative step with floor for stability
    eps1 = 1e-6 * (jnp.abs(p1) + 1.0)
    eps2 = 1e-6 * (jnp.abs(p2) + 1.0)

    # single generic python callback; no tracers captured
    def _cb(p1v, p2v):
        return _props_py(input_state, p1v, p2v, fluid)

    # primal at (p1, p2)
    base = jax.pure_callback(_cb, _TEMPLATE, p1, p2)

    # forward-diff samples: pass the already-perturbed args to pure_callback
    p1p = jax.pure_callback(_cb, _TEMPLATE, p1 + eps1, p2)
    p2p = jax.pure_callback(_cb, _TEMPLATE, p1, p2 + eps2)

    # directional derivative
    df_dp1 = {k: (p1p[k] - base[k]) / eps1 for k in _WANTED}
    df_dp2 = {k: (p2p[k] - base[k]) / eps2 for k in _WANTED}
    jvp = {k: df_dp1[k] * p1_dot + df_dp2[k] * p2_dot for k in _WANTED}

    return base, jvp



    # It seems that the nice custom derivative with alpha will not work because of how JAX tracing works
    # p1_dot and p2_dot are tracers, not numerical values, and therefore cannot be passed to coolprop

    # @get_props.defjvp
    # def _get_props_jvp(input_state, fluid, primals, tangents):
    #     """
    #     Custom JVP (forward-mode) using a directional, scale-aware finite difference:

    #         alpha = sqrt( (p1_dot/d1)^2 + (p2_dot/d2)^2 )   with d1,d2 ~ reference scales
    #         step  = (p1_dot/alpha, p2_dot/alpha)
    #         jvp   = alpha * (f(p + step) - f(p))

    #     This keeps the step aligned with the tangent, with size set by the
    #     reference magnitudes to avoid poor conditioning.
    #     """
    #     p1, p2 = primals
    #     p1_dot, p2_dot = tangents

    #     # Try to pull per-variable scales from reference_state, fallback to |p| or 1.0
    #     name_1, name_2 = INPUT_PAIR_MAP[input_state]
    #     delta_p1 = 1e-6 * fluid.reference_state[name_1]
    #     delta_p2 = 1e-6 * fluid.reference_state[name_2]
    #     alpha = jnp.sqrt((p1_dot/delta_p1)**2 + (p2_dot/delta_p2)**2)

    #     # Base and "plus" states (sanitize once so diffs are safe)
    #     p1f = p1 + (p1_dot / alpha)
    #     p2f = p2 + (p2_dot / alpha)
    #     base_state = _sanitize(fluid.get_state(input_state, p1, p2).to_dict())
    #     plus_state = _sanitize(fluid.get_state(input_state, p1f, p2f).to_dict())

    #     # Directional 
    #     jvp = {k: alpha * (plus_state[k] - base_state[k]) for k in base_state}

    #     return base_state, jvp