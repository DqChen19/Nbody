from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from dataclasses import replace
from typing import Callable

from .snapshot import g_func, gd_func

# =================================
# calc_vsky, calc_bsky2, and dtbvdq
# =================================
def calc_vsky(v, i: int, j: int) -> float:
    '''
    Relative sky-plane velocity magnitude
    i: body being transited, normally the star
    j: occulting body
    '''
    dvx = v[0, j] - v[0, i]
    dvy = v[1, j] - v[1, i]
    return jnp.hypot(dvx, dvy)

def calc_bsky2(x, i: int, j: int) -> float:
    '''
    Squared projected sky-plane separation.
    '''
    dx = x[0, j] - x[0, i]
    dy = x[1, j] - x[1, i]
    return dx * dx + dy * dy


def dtbvdq(s, i: int, j: int, *, include_bv: bool,):
    """
    Calculate derivatives of transit observables with respect to the
    initial Cartesian state.

    Parameters
    ----------
    s: State evaluated at the refined transit time.
    i: Body being transited, normally the star.
    j: Occulting body.

    include_bv
        False: Return only transit-time derivatives, shape (1, 7, n).
        True:Return derivatives of transit time, vsky, and bsky2, shape (3, 7, n).

    Returns
    -------
    observables
        If include_bv is False: None
        If include_bv is True: Array [vsky, bsky2].

    derivatives
        Shape (1, 7, n) or (3, 7, n).
    """
    n = s.n

    dx = s.x[0, j] - s.x[0, i]
    dy = s.x[1, j] - s.x[1, i]

    dvx = s.v[0, j] - s.v[0, i]
    dvy = s.v[1, j] - s.v[1, i]

    indi, indj = 7 * i, 7 * j

    gdot = gd_func(i, j, s.x, s.v, s.dqdt,)
    eps = jnp.finfo(s.x.dtype).eps
    safe_gdot = jnp.where(jnp.abs(gdot) > eps,gdot, jnp.where(gdot >= 0, eps, -eps),)

    # Each row has shape (7*n,).
    djx = (s.jac_step[indj + 0, :]- s.jac_step[indi + 0, :])
    djy = (s.jac_step[indj + 1, :] - s.jac_step[indi + 1, :])
    djvx = (s.jac_step[indj + 3, :] - s.jac_step[indi + 3, :])
    djvy = (s.jac_step[indj + 4, :] - s.jac_step[indi + 4, :])

    # ∂g/∂q0 at fixed time.
    dg_dq = (djx * dvx + djy * dvy + djvx * dx + djvy * dy)

    # Implicit differentiation:
    # g(t(q0), q0) = 0
    # dt/dq0 = -(∂g/∂q0)/(dg/dt)
    dtdq_flat = -dg_dq / safe_gdot

    # Convert flat body-major ordering:
    # [body0 x,y,z,vx,vy,vz,m, body1 ...]
    # to shape (7, n).
    dtdq = dtdq_flat.reshape(n, 7).T

    if not include_bv:
        return None, dtdq[jnp.newaxis, :, :]

    vsky = calc_vsky(s.v, i, j)
    bsky2 = calc_bsky2(s.x, i, j)

    safe_vsky = jnp.maximum(vsky, eps)

    adx = (s.dqdt[indj + 3] - s.dqdt[indi + 3])
    ady = (s.dqdt[indj + 4] - s.dqdt[indi + 4])

    # Explicit time derivative of vsky.
    dvsky_dt = ( dvx * adx + dvy * ady) / safe_vsky

    # Total derivative evaluated at the implicitly defined transit time.
    dvsky_dq_flat = ((djvx * dvx + djvy * dvy) / safe_vsky+ dvsky_dt * dtdq_flat)
    # At exact mid-transit g = 0, so dbsky2/dt = 2g = 0.
    #
    # To reproduce the Julia implementation, only the explicit term is included here.
    dbsky2_dq_flat = 2.0 * (djx * dx + djy * dy)

    dvsky_dq = dvsky_dq_flat.reshape(n, 7).T
    dbsky2_dq = dbsky2_dq_flat.reshape(n, 7).T

    derivatives = jnp.stack((dtdq, dvsky_dq, dbsky2_dq,),axis=0,)
    observables = jnp.stack((vsky, bsky2,))

    return observables, derivatives

# ================================
# Transit Timing
# ================================
def calc_dtdelements(s, tt):
    """
    Convert d(transit time)/dq0 to derivatives with respect to initial orbital elements.

    Required shapes
    ---------------
    tt.dtdq0: (n, ntt, 7, n)
    s.jac_init: (7*n, 7*n)

    Returns
    -------
    New TransitTiming object with updated dtdelements.
    """
    n = s.n
    dtdq0 = tt.dtdq0 # (n, ntt, 7, n)
    dtdq_flat = (jnp.swapaxes(dtdq0, -1, -2).reshape(n, tt.ntt, 7 * n)) #(n, ntt, 7*n)
    result_flat = jnp.einsum("...p,pq->...q", dtdq_flat,s.jac_init,) # (n, ntt, 7*n) @ (7*n, 7*n)
    # Return to (n, ntt, 7, n).
    dtdelements = jnp.swapaxes(result_flat.reshape(n, tt.ntt, n, 7), -1,-2,)

    return replace(tt,dtdelements=dtdelements,)

# ================================
# Transit Parameters
# ================================
def calc_dtbvdelements(s, tt):
    """
    Convert derivatives of transit time, vsky, and bsky2 from
    Cartesian initial conditions to orbital elements.

    Required shape
    --------------
    tt.dtbvdq0 (3, n, ntt, 7, n)
    """
    n = s.n

    dtbvdq_flat = (jnp.swapaxes(tt.dtbvdq0, -1, -2).reshape(3, n, tt.ntt, 7 * n))
    result_flat = jnp.einsum("...p,pq->...q", dtbvdq_flat, s.jac_init,)

    dtbvdelements = jnp.swapaxes(result_flat.reshape(3, n,tt.ntt, n, 7,), -1, -2,)
    return replace(tt, dtbvdelements=dtbvdelements,)

def zero_derivatives(d):
    return jax.tree_util.tree_map(jnp.zeros_like, d,)

def find_transit_grad(s_anchor, d_template,*, transited_body: int, 
                      occultor: int, dt_initial, scheme_grad: Callable,):
    """
    Refine the transit-time offset using Newton iteration.

    Parameters
    ----------
    s_anchor: State at the beginning of the enclosing integration step.
    d_template: Derivatives object with the correct PyTree structure.
    transited_body: Usually the central star.
    occultor: Candidate transiting body.
    dt_initial: Linear-interpolation initial guess measured relative to s_prior.
    scheme_grad
        One-step gradient scheme:
            s_new, d_new = scheme_grad(s, d, h)

    Returns
    -------
    (refined transit offset: relative to the rough guess of the transit time)
    s_transit: State evaluated at the refined transit offset.
    d_transit: Derivatives evaluated at the refined transit offset.
    dt_transit: Refined time offset relative to s_prior.t[0].
    """
    dtype = s_anchor.x.dtype

    dt0 = jnp.asarray(dt_initial, dtype=dtype)
    stmp = jnp.asarray(0.0, dtype=dtype)

    previous_1 = dt0 + jnp.asarray(1.0, dtype=dtype)
    previous_2 = dt0 + jnp.asarray(2.0, dtype=dtype)

    iteration = jnp.asarray(0, dtype=jnp.int32)
    d_zero = zero_derivatives(d_template)

    # Initial placeholder. It will be overwritten in the first iteration.
    s_trial = s_anchor
    d_trial = d_zero

    carry = (dt0, stmp, previous_1, previous_2, iteration, s_trial, d_trial,)

    def cond_fun(carry):
        (dt_estimate, _, old_1, old_2, iteration, _, _,) = carry
        return ((iteration < 20) & (dt_estimate != old_1) & (dt_estimate != old_2))

    def body_fun(carry):
        (dt_estimate, compensation, old_1, _old_2, iteration, _s_trial, _d_trial,) = carry

        new_old_2 = old_1
        new_old_1 = dt_estimate

        d_start = zero_derivatives(d_template)
        s_new, d_new = scheme_grad(s_anchor, d_start, dt_estimate,)

        gsky = g_func(transited_body, occultor, s_new.x, s_new.v,)
        gdot = gd_func(transited_body, occultor, s_new.x, s_new.v, s_new.dqdt,)

        eps = jnp.finfo(dtype).eps
        safe_gdot = jnp.where(jnp.abs(gdot) > eps,gdot,jnp.where(gdot >= 0, eps, -eps),)

        correction = -gsky / safe_gdot

        # JAX form of compensated summation.
        y = correction - compensation
        t = dt_estimate + y
        updated_compensation = (t - dt_estimate) - y
        updated_dt = t
        
        return (updated_dt, updated_compensation, new_old_1, new_old_2, iteration + 1, s_new, d_new,)

    (dt_final, _, _, _, _, _, _,) = lax.while_loop(cond_fun, body_fun, carry,)

    # Julia recomputes the state and derivatives once at the final dt.
    d_final_start = zero_derivatives(d_template)
    s_final, d_final = scheme_grad(s_anchor, d_final_start, dt_final,)

    return s_final, d_final, dt_final

def find_transit_no_grad(s_anchor, *, transited_body: int, occultor: int, dt_initial, scheme_no_grad: Callable,):
    """No-gradient counterpart of find_transit_grad."""
    dtype = s_anchor.x.dtype

    dt0 = jnp.asarray(dt_initial, dtype=dtype)
    previous_1 = dt0 + jnp.asarray(1.0, dtype=dtype)
    previous_2 = dt0 + jnp.asarray(2.0, dtype=dtype)
    iteration = jnp.asarray(0, dtype=jnp.int32)

    carry = (dt0, previous_1, previous_2, iteration, s_anchor,)

    def cond_fun(carry):
        dt_estimate, old_1, old_2, iteration, _ = carry

        return ((iteration < 20) & (dt_estimate != old_1) & (dt_estimate != old_2))

    def body_fun(carry):
        (dt_estimate, old_1, _, iteration, _,) = carry

        new_old_2 = old_1
        new_old_1 = dt_estimate
        ###############
        s_trial = scheme_no_grad(s_anchor, dt_estimate,) #here used the ahl21_no_grad!!
        ###############
        gsky = g_func(transited_body, occultor, s_trial.x, s_trial.v,)
        gdot = gd_func(transited_body,occultor, s_trial.x, s_trial.v, s_trial.dqdt,)

       
        correction = - gsky / gdot
        updated_dt = dt_estimate + correction

        return (updated_dt, new_old_1, new_old_2, iteration + 1, s_trial,)

    (dt_final, _, _, _, _,)  = lax.while_loop(cond_fun, body_fun, carry,) ###
    s_final = scheme_no_grad(s_anchor,dt_final,)

    return s_final, dt_final