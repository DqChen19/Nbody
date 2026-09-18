"""
NbodyGradient

__init___.py

"""
import os
# This package integrates small-N systems (a handful of bodies) through
# control-flow-heavy code (lax.while_loop Kepler solves nested inside a
# lax.fori_loop over thousands of steps). GPU is dramatically slower here than CPU. 
# Default to CPU unless the caller has already requested a platform explicitly.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True) #enable float64

# Global constant
NDIM    = 3
YEAR    = 365.242
GNEWT   = 39.4845 / (YEAR * YEAR)
THIRD   = 1.0 / 3.0
ALPHA0  = 0.0

# module

from .pre_alloc_arrays import (dTime,  Jacobian, Derivatives,)
from .ics.initial_conditions import (Elements, ElementsIC, CartesianIC, InitialConditions,)
from .integrator.Integrator_with_grad import (Integrator, ahl21)
from .integrator.Integrator_no_grad import (Integrator_no_grad, ahl21_no_grad)
from .outputs.Outputs import (CartesianOutput,)
from .integrator.Transits import (create_transit_parameters, create_transit_timing, create_transit_snapshot)
from .integrator.State import State, dState
from .ics.defaults import (available_systems, get_default_ICs,)
from .ics.init_nbody import init_nbody
from .ics.kepler_init import kepler_init
from .ics.setup_hierarchy import hierarchy
# APIs

__all__ = [
    "NDIM", "YEAR", "GNEWT", "THIRD", "ALPHA0",
    "Elements", "ElementsIC", "CartesianIC", "InitialConditions",
    "State", "dState", "init_nbody", 'kepler_init',
    "Integrator", "dTime", "Integrator_no_grad", "Jacobian", "Derivatives",
    "CartesianOutput",
    "create_transit_timing", "create_transit_parameters", "create_transit_snapshot",
    "ahl21_no_grad", "ahl21", 
    "hierarchy",
    "available_systems", "get_default_ICs"
]
