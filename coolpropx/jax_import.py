# Import JAX in a flexible way
try:
    import os
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    JAX_AVAILABLE = True
except ImportError:
    import numpy as np
    jnp = np
    lax = None
    JAX_AVAILABLE = False

import logging
logger = logging.getLogger("jax")
logger.setLevel(logging.WARNING)

__all__ = ["jax", "jnp", "JAX_AVAILABLE"]


if JAX_AVAILABLE:

    import inspect
    import difflib
    import textwrap
    import diffrax as dfx


    # Create list of Diffrax solvers
    SOLVERS = {
        name: cls
        for name, cls in inspect.getmembers(dfx)
        if inspect.isclass(cls)
        and issubclass(cls, dfx.AbstractSolver)
        and not inspect.isabstract(cls)                    # drop Abstract*
        and not issubclass(cls, dfx.AbstractItoSolver)      # drop SDE (Itô)
        and not issubclass(cls, dfx.AbstractStratonovichSolver)  # drop SDE (Stratonovich)
    }

    def format_list_80(items: list[str], width: int = 80) -> str:
        body = ", ".join(sorted(items))
        wrapped = textwrap.fill(
            body, width=width - 2, subsequent_indent="",
            break_long_words=False, break_on_hyphens=False,
        )
        # ensure bracketed style even when wrapped
        lines = wrapped.splitlines()
        if len(lines) == 1:
            return f"{lines[0]}"
        return "\n".join(lines)

    def make_diffrax_solver(name: str, **kwargs):
        key = name.strip()
        if key in SOLVERS:
            return SOLVERS[key](**kwargs)
        # hint first, then full list on one (wrapped) line
        opts = list(SOLVERS.keys())
        suggestions = difflib.get_close_matches(key, opts, n=2)
        parts = []
        if suggestions:
            parts.append(f"Invalid solver '{name}'. Did you mean {', '.join(repr(s) for s in suggestions)}?")
        parts.append("\nList of valid solvers:")
        parts.append(format_list_80(opts))
        parts.append("\n")
        raise ValueError("\n".join(parts))

    # adjoints: discover programmatically; exclude abstract classes
    ADJOINTS = {
        name: cls
        for name, cls in inspect.getmembers(dfx)
        if inspect.isclass(cls)
        and issubclass(cls, dfx.AbstractAdjoint)
        and not inspect.isabstract(cls)
    }

    def make_diffrax_adjoint(name: str, **kwargs):
        key = name.strip()
        if key in ADJOINTS:
            return ADJOINTS[key](**kwargs)
        # hint first, then a wrapped single-line-style list
        opts = list(ADJOINTS.keys())
        suggestions = difflib.get_close_matches(key, opts, n=2)
        parts = []
        if suggestions:
            parts.append(
                f"Invalid adjoint '{name}'. Did you mean {', '.join(repr(s) for s in suggestions)}?"
            )
        parts.append("\nList of valid adjoints:")
        parts.append(format_list_80(opts))
        parts.append("\n")
        raise ValueError("\n".join(parts))
