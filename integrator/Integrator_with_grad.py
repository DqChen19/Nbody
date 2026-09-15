from __future__ import annotations
import numpy as np
import jax.numpy as jnp
import jax
import jax.lax as lax
from dataclasses import replace
from functools import partial
from typing import Any, Callable, Optional

from ..pre_alloc_arrays import Derivatives
from ..utils import *
from .State import State
from .snapshot import calc_dbvdq,calc_dbvdelements
from .kernels import integrate_transit_output_grad
from .Transits import TransitParameters, TransitSnapshot, TransitTiming
from .ahl21.ahl21 import ahl21

NDIM    = 3
Array = jax.Array
# Helpful functions:
def check_step(t0, tmax) -> float:
    '''
    Check the direction of the step
    '''
    sign_tmax = jnp.sign(tmax)
    return jnp.where(jnp.abs(tmax) > jnp.abs(t0), sign_tmax, jnp.where(sign_tmax != jnp.sign(t0), sign_tmax, -sign_tmax,),)


@partial(jax.jit, static_argnames=("scheme_grad",),)
def _run_n_steps_grad(s: State, d: Derivatives, h: Array, *, scheme_grad: Callable, nsteps: int,):
    """
    Perform nsteps using the gradient propagation scheme.
    Required scheme interface
    -------------------------
    s_new, d_new = scheme_grad(s, d, h)
    """

    def body_fun(_, carry):
        state, derivatives = carry
        state, derivatives = scheme_grad(state, derivatives,h,)
        return state, derivatives

    return lax.fori_loop(0, nsteps, body_fun, (s, d),)


# =========================================================
# Integrator class: JAX-compatible N-body integration dispatcher.
# =========================================================
class Integrator:
    """
    Parameters
    ----------
    scheme_grad: Gradient-enabled one-step scheme: s_new, d_new = scheme_grad(s, d, h)

    h: Positive base step size.
    tmax:  Default elapsed integration duration.
    t0: Reference initial time. Retained for compatibility with Julia.
    """

    def __init__(self, scheme_grad: Callable, h: float, tmax: float, t0: float = 0.0,):
        self.scheme_grad = scheme_grad
        # Keep a positive base step. Direction is chosen per call.
        self.h = abs(float(h))
        self.t0 = float(t0)
        self.tmax = float(tmax)

    @classmethod
    def default(cls, h: float, tmax: float, t0: float = 0.0) -> "Integrator":
        return cls(scheme_grad=ahl21,  h=h,  tmax=tmax, t0=t0,)

    @classmethod
    def with_scheme(cls, scheme_grad: Callable, h: float, tmax: float,t0: float = 0.0,) -> "Integrator":
        return cls(scheme_grad=scheme_grad, h=h, tmax=tmax, t0=t0,)
    
    def __call__(self, s: State, arg: Optional[Any] = None, derivatives: Optional[Derivatives] = None, 
                 *, return_arrays: bool = False,):
        """
        Supported calls for dispatch
        """
        # Dispatch This method itself is intentionally
        # not decorated with jax.jit.
        if isinstance(arg, TransitSnapshot): # Evaluate sky-plane quantities at specified times.
            return self._call_snapshot(s, arg, derivatives, return_arrays=return_arrays,)

        if isinstance(arg, TransitTiming): #Detect transit times (and sky-plane parameters).
            return self._call_transits(s, arg, derivatives, return_arrays=return_arrays,)

        if isinstance(arg, TransitParameters): #Detect transit times (and sky-plane parameters).
            return self._call_parameters(s, arg, derivatives, return_arrays=return_arrays,)

        # Julia: intr(s) -> intr(s, s.t[1] + intr.tmax)
        if arg is None: #Integrate by self.tmax.
            current_time = float(np.asarray(s.t[0]))
            target_time = current_time + self.tmax
            return self._to_time(s, target_time, derivatives, return_arrays=return_arrays,)

        # bool is a subclass of int, so reject it explicitly.
        if isinstance(arg, bool):
            raise TypeError( "Boolean values cannot be used as step counts.")

        # Julia: intr(s, N::Int64)
        if isinstance(arg, (int, np.integer)): #Integrate to nsteps
            return self._n_steps(s, int(arg), derivatives, return_arrays=return_arrays,)

        # Julia: intr(s, time::T)
        if isinstance(arg, (float, np.floating)): # Integrate to absolute time
            return self._to_time(s, float(arg), derivatives, return_arrays=return_arrays,)

        raise TypeError(f"arg must be None, a target time, a step count, TransitTiming, TransitParameters, or TransitSnapshot; Here got {type(arg).__name__}.")
    
    def _n_steps(self, s: State, nsteps: int, derivatives: Optional[Derivatives] = None, *, return_arrays: bool):
        """
        Integrate a fixed number of full steps.
        A negative nsteps integrates backward without mutating self.h.
        """
        if nsteps == 0:
           
            if derivatives is None:
                derivatives = Derivatives.create(s.n)

            if return_arrays:
                return s, derivatives
            return s

        direction = 1.0 if nsteps > 0 else -1.0
        count = abs(nsteps)
        h = jnp.asarray(direction * self.h, dtype=s.x.dtype,)

        if derivatives is None:
            derivatives = Derivatives.create(s.n)
        s, derivatives = _run_n_steps_grad(s, derivatives, h, scheme_grad=self.scheme_grad, nsteps=count,)


        # The integration scheme updates x/v, while the Integrator owns
        # the absolute simulation clock.
        new_time = s.t[0] + jnp.asarray(count,dtype=s.t.dtype,) * h
        s = replace(s, t=s.t.at[0].set(new_time),)

        if return_arrays:
            return s, derivatives

        return s
    
    def _to_time(self, s: State, time: float, derivatives: Optional[Derivatives] = None, *, return_arrays: bool,):
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
            if derivatives is None:
                derivatives = Derivatives.create(s.n)

            if return_arrays:
                return s, derivatives

            return s

        direction = 1.0 if delta_t > 0.0 else -1.0
        full_h = direction * self.h

        nsteps = int(np.floor(abs(delta_t) / self.h))
        elapsed = nsteps * full_h
        final_h = delta_t - elapsed
        h_array = jnp.asarray(full_h,dtype=s.x.dtype,)

        if derivatives is None:
            derivatives = Derivatives.create(s.n)

        if nsteps > 0:
            s, derivatives = _run_n_steps_grad(s, derivatives, h_array, scheme_grad=self.scheme_grad, nsteps=nsteps,)

        if not np.isclose(final_h, 0.0, rtol=0.0, atol=np.finfo(float).eps * 16,):
            final_h_array = jnp.asarray(final_h, dtype=s.x.dtype,)
            s, derivatives = self.scheme_grad(s, derivatives,final_h_array,)


        # Set the exact requested endpoint once.
        s = replace(s, t=s.t.at[0].set(jnp.asarray(target_time,dtype=s.t.dtype)),)

        if return_arrays:
            return s, derivatives

        return s
    
    def _call_transits(self, s, output, derivatives: Optional[Derivatives] = None, *, return_arrays: bool,):
        nsteps = abs(round(self.tmax / self.h))
        direction = 1.0 if self.tmax > 0.0 else -1.0
        h = direction * self.h

        if derivatives is None:
            derivatives = Derivatives.create(s.n)
        s, derivatives, output = integrate_transit_output_grad(s, derivatives, output,
                                                                scheme_grad=self.scheme_grad, h=h, nsteps=nsteps,)
        if return_arrays:
            return s, output, derivatives

        return s, output

    def _call_parameters(self, s, output, derivatives: Optional[Derivatives] = None, *, return_arrays: bool,):
            nsteps = abs(round(self.tmax / self.h))
            direction = 1.0 if self.tmax > 0.0 else -1.0
            h = direction * self.h
    
            if derivatives is None:
                derivatives = Derivatives.create(s.n)
            s, derivatives, output = integrate_transit_output_grad(s, derivatives, output,
                                                                    scheme_grad=self.scheme_grad, h=h, nsteps=nsteps,)
            if return_arrays:
                return s, output, derivatives
    
            return s, output


    def _call_snapshot(self, s, output, derivatives: Optional[Derivatives] = None, *, return_arrays: bool,):
        if derivatives is None:
            derivatives = Derivatives.create(s.n)
        vsky = output.vsky
        bsky2 = output.bsky2
        dbvdq0 = output.dbvdq0

        # Snapshot is currently a Python-level orchestration loop.
        # Each long integration segment calls the jitted step kernel.
        for time_index in range(output.nt):
            target_time = float(np.asarray(output.times[time_index]))
            s, derivatives = self._to_time(s, target_time, derivatives,return_arrays=True,)

            central_body = 0
            for body_index in range(1, s.n):
                (current_vsky,current_bsky2,current_dbvdq,) = calc_dbvdq(s, central_body, body_index,)
                dbvdq0 = dbvdq0.at[ :, body_index,  time_index,  :,   :,].set(current_dbvdq)
                vsky = vsky.at[body_index, time_index,].set(current_vsky)
                bsky2 = bsky2.at[body_index,time_index,].set(current_bsky2)

        dbvdelements = calc_dbvdelements(dbvdq0, s.jac_init,)

        output = replace(output, vsky=vsky,bsky2=bsky2, dbvdq0=dbvdq0, dbvdelements=dbvdelements,)

        if return_arrays:
            return s, output, derivatives

        return s, output
