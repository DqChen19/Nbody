from __future__ import annotations
from typing import Callable

from .State import State
from .kernels import integrate_transit_output_reverse
from .Transits import TransitTiming
from .ahl21.ahl21_no_grad import ahl21_no_grad

# =========================================================
# Integrator class: reverse-mode (adjoint) differentiable transit timing.
# =========================================================
class Integrator_reverse:
    """
    Transit-timing integrator that jax.grad / jax.vjp can differentiate
    It runs ahl21_no_grad by default; 
    derivatives come from JAX reverse-mode autodiff through the integration

    Only TransitTiming output is supported currently.

    Parameters
    ----------
    h: Positive base step size.
    tmax: Elapsed integration duration (negative integrates backward in time).
    t0: Reference initial time. Retained for compatibility with Julia.
    """

    def __init__(self, scheme_no_grad: Callable, h: float, tmax: float, t0: float = 0.0,):
        self.scheme_no_grad = scheme_no_grad
        self.h = abs(float(h))
        self.t0 = float(t0)
        self.tmax = float(tmax)

    @classmethod
    def default(cls, h: float, tmax: float, t0: float = 0.0) -> "Integrator_reverse": #use ahl21-no-grad as default
        return cls(scheme_no_grad=ahl21_no_grad, h=h, tmax=tmax, t0=t0,)

    @classmethod
    def with_scheme(cls, scheme_no_grad: Callable, h: float, tmax: float, t0: float = 0.0,) -> "Integrator_reverse":
        return cls(scheme_no_grad=scheme_no_grad, h=h, tmax=tmax, t0=t0,)

    def __call__(self, s: State, output: TransitTiming, slots=None,):
        """slots: optional (rows, epochs) of output.tt entries that need gradients; see integrate_transit_output_reverse."""
        if not isinstance(output, TransitTiming):
            raise TypeError(f"Integrator_reverse supports TransitTiming output only; got {type(output).__name__}.")
        nsteps = abs(round(self.tmax / self.h))
        direction = 1.0 if self.tmax > 0.0 else -1.0
        h = direction * self.h
        return integrate_transit_output_reverse(s, output, scheme_no_grad=self.scheme_no_grad, h=h, nsteps=nsteps, slots=slots,)
