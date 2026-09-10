from dataclasses import dataclass
from copy import deepcopy
import pickle
from ..pre_alloc_arrays import Derivatives
from ..integrator.Integrator_with_grad import check_step


class AbstractOutput:
    pass

@dataclass
class CartesianOutput(AbstractOutput):
    nbody: int
    nstep: int
    filename: str = "data.pkl"
    file: bool = False

    def __post_init__(self):
        self.states = [None] * self.nstep


def run_cartesian_output(integrator, state, output):
    t0 = state.t[0]
    nsteps = output.nstep

    direction = check_step(t0, integrator.tmax)
    h = integrator.h * direction

    derivatives = Derivatives.create(state.x.dtype, state.n)

    for i in range(nsteps):
        output.states[i] = deepcopy(state)
        state, derivatives = integrator.scheme_grad(state, derivatives, h,)
        state.t = state.t.at[0].set( t0 + h * (i + 1) )

    if output.file:
        with open(output.filename, "wb") as file:
            pickle.dump({"states": output.states},file,)

    return state
