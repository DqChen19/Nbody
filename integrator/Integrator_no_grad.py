from __future__ import annotations
import numpy as np
import jax.numpy as jnp
import jax
import jax.lax as lax
from dataclasses import replace
from functools import partial
from typing import Any, Callable, Optional

from ..utils import *
from .State import State
from .snapshot import calc_bv
from .kernels import  integrate_transit_output_no_grad
from .Transits import TransitParameters, TransitSnapshot, TransitTiming
from .ahl21.ahl21_no_grad import ahl21_no_grad

NDIM    = 3
Array = jax.Array
# Helpful functions:
def check_step(t0, tmax) -> float:
    '''
    Check the direction of the step
    '''
    sign_tmax = jnp.sign(tmax)
    return jnp.where(jnp.abs(tmax) > jnp.abs(t0), sign_tmax, jnp.where(sign_tmax != jnp.sign(t0), sign_tmax, -sign_tmax,),)


@partial(jax.jit, static_argnames=("scheme_no_grad",),)
def _run_n_steps_no_grad(s: State, h: Array, *, scheme_no_grad: Callable, nsteps: int,):
    """
    s_new = scheme_no_grad(s, h)
    """
    def body_fun(_, state):
        return scheme_no_grad(state,h,)

    return lax.fori_loop(0, nsteps, body_fun, s,)


# =========================================================
# Integrator class: JAX-compatible N-body integration dispatcher.
# =========================================================
class Integrator_no_grad:
    """
    Parameters
    ----------
    scheme_no_grad: No-gradient one-step scheme: s_new = scheme_no_grad(s, h)

    h: Positive base step size.
    tmax:  Default elapsed integration duration.
    t0: Reference initial time. Retained for compatibility with Julia.
    """

    def __init__(self, scheme_no_grad: Callable, h: float, tmax: float, t0: float = 0.0,):
        self.scheme_no_grad = scheme_no_grad
        # Keep a positive base step. Direction is chosen per call.
        self.h = abs(float(h))
        self.t0 = float(t0)
        self.tmax = float(tmax)

    @classmethod
    def default(cls, h: float, tmax: float, t0: float = 0.0) -> "Integrator_no_grad":
        return cls(scheme_no_grad=ahl21_no_grad, h=h,  tmax=tmax, t0=t0,)

    @classmethod
    def with_scheme(cls, scheme_no_grad: Callable, h: float, tmax: float,t0: float = 0.0,) -> "Integrator_no_grad":
        return cls(scheme_no_grad=scheme_no_grad, h=h, tmax=tmax, t0=t0,)
    
    def __call__(self, s: State, arg: Optional[Any] = None,):
        """
        Supported calls for dispatch
        """
        # Dispatch This method itself is intentionally
        # not decorated with jax.jit.
        if isinstance(arg, TransitSnapshot): # Evaluate sky-plane quantities at specified times.
            return self._call_snapshot(s, arg,)

        if isinstance(arg,TransitTiming): #Detect transit times (and sky-plane parameters).
            return self._call_transits(s, arg,)

        if isinstance(arg, TransitParameters):
            return self._call_parameters(s, arg,)
        # Julia: intr(s) -> intr(s, s.t[1] + intr.tmax)
        if arg is None: #Integrate by self.tmax.
            current_time = float(np.asarray(s.t[0]))
            target_time = current_time + self.tmax
            return self._to_time(s, target_time,)

        # bool is a subclass of int, so reject it explicitly.
        if isinstance(arg, bool):
            raise TypeError( "Boolean values cannot be used as step counts.")

        # Julia: intr(s, N::Int64)
        if isinstance(arg, (int, np.integer)): #Integrate to nsteps
            return self._n_steps(s, int(arg),)

        # Julia: intr(s, time::T)
        if isinstance(arg, (float, np.floating)): # Integrate to absolute time
            return self._to_time(s, float(arg),)

        raise TypeError(f"arg must be None, a target time, a step count, TransitTiming, TransitParameters, or TransitSnapshot; Here got {type(arg).__name__}.")
    
    def _n_steps(self, s: State, nsteps: int, ):
        """
        Integrate a fixed number of full steps.
        A negative nsteps integrates backward without mutating self.h.
        """
        if nsteps == 0:
            return s

        direction = 1.0 if nsteps > 0 else -1.0
        count = abs(nsteps)
        h = jnp.asarray(direction * self.h, dtype=s.x.dtype,)
        s = _run_n_steps_no_grad(s, h, scheme_no_grad=self.scheme_no_grad, nsteps=count,)

        # The integration scheme updates x/v, while the Integrator owns
        # the absolute simulation clock.
        new_time = s.t[0] + jnp.asarray(count,dtype=s.t.dtype,) * h
        s = replace(s, t=s.t.at[0].set(new_time),)

        return s
    
    def _to_time(self, s: State, time: float,):
        """
        Integrate from the current s.t[0] to an absolute target time.

        This method performs:
        1. zero or more full steps of size abs(self.h);
        2. one shorter remainder step when required.
        """

        current_time = float(np.asarray(s.t[0]))
        target_time = float(time)
        delta_t = target_time - current_time

        if delta_t == 0.0:
            return s

        direction = 1.0 if delta_t > 0.0 else -1.0
        full_h = direction * self.h

        nsteps = int(np.floor(abs(delta_t) / self.h))
        elapsed = nsteps * full_h
        final_h = delta_t - elapsed
        h_array = jnp.asarray(full_h,dtype=s.x.dtype,)

        if nsteps > 0:
            s = _run_n_steps_no_grad(s, h_array, scheme_no_grad=self.scheme_no_grad, nsteps=nsteps,)

        if not np.isclose(final_h, 0.0, rtol=0.0, atol=np.finfo(float).eps * 16,):
            final_h_array = jnp.asarray(final_h, dtype=s.x.dtype,)
            s = self.scheme_no_grad(s, final_h_array,)

        # Set the exact requested endpoint once.
        s = replace(s, t=s.t.at[0].set(jnp.asarray(target_time,dtype=s.t.dtype)),)
        return s
    
    def _call_transits(self, s, output,):
        nsteps = abs(round(self.tmax / self.h))
        direction = 1.0 if self.tmax > 0.0 else -1.0
        h = direction * self.h
        s, output = integrate_transit_output_no_grad(s, output, scheme_no_grad=self.scheme_no_grad, h=h, nsteps=nsteps,)
        return s, output

    def _call_parameters(self, s, output,):
        nsteps = abs(round(self.tmax / self.h))
        direction = 1.0 if self.tmax > 0.0 else -1.0
        h = direction * self.h
        s, output = integrate_transit_output_no_grad(s, output, scheme_no_grad=self.scheme_no_grad, h=h, nsteps=nsteps,)
        return s, output
    
    def _call_snapshot(self, s, output,):
        vsky = output.vsky
        bsky2 = output.bsky2

        # Snapshot is currently a Python-level orchestration loop.
        # Each long integration segment calls the jitted step kernel.
        for time_index in range(output.nt):
            target_time = float(np.asarray(output.times[time_index]))
            s = self._to_time(s, target_time,)

            central_body = 0
            for body_index in range(1, s.n):
                (current_vsky, current_bsky2,) = calc_bv(s, central_body, body_index,)

                vsky = vsky.at[body_index, time_index,].set(current_vsky)
                bsky2 = bsky2.at[body_index,time_index,].set(current_bsky2)

        output = replace(output, vsky=vsky, bsky2=bsky2,)

        return s, output