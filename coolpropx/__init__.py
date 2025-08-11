# Highlight exception messages
# https://stackoverflow.com/questions/25109105/how-to-colorize-the-output-of-python-errors-in-the-gnome-terminal/52797444#52797444
try:
    import IPython.core.ultratb
except ImportError:
    # No IPython. Use default exception printing.
    pass
else:
    import sys
    sys.excepthook = IPython.core.ultratb.FormattedTB(color_scheme='linux', call_pdb=False)


from .core_calculations import *
from .fluid_properties import *
from .graphics import *
from .utils import *


from . import perfect_gas
from . import jax_import
# from .perfect_gas import *


# Package info
__version__ = "0.2.14"
PACKAGE_NAME = "coolpropx"
URL_GITHUB = "https://github.com/turbo-sim/coolpropx"
URL_DOCS = "https://turbo-sim.github.io/coolpropx/"
URL_PYPI = "https://pypi.org/project/coolpropx/"
URL_DTU = "https://thermalpower.dtu.dk/"
BREAKLINE = 80 * "-"

