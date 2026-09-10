from __future__ import annotations
from dataclasses import replace
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from .timing import find_transit_no_grad, find_transit_grad, calc_vsky, calc_bsky2, dtbvdq
from .snapshot import g_func
from .Transits import TransitParameters, TransitTiming
# ================================
# gsave intialize
# ================================
def initialize_gsave_vmap(s, output):
    occs = jnp.asarray(output.occs, dtype=jnp.int32,)
    values = jax.vmap(
        lambda i: g_func(i, output.ti, s.x, s.v,))(occs)

    return replace(output, gsave=output.gsave.at[occs].set(values),)

# ================================
# transit loop
# ================================
@partial(jax.jit,static_argnames=("scheme_grad",),)
def integrate_transit_output_grad(s, d, output, *, scheme_grad, h, nsteps,):
    """
    Integrate for a fixed number of steps and detect transits.
    scheme_grad must have the interface:

        s_new, d_new = scheme_grad(s, d, h)
    """
    t0 = state_time(s)
    output = initialize_gsave_vmap(s,output,)

    def body_fun(istep, carry):
        state, derivatives, transit_output = carry

        # State before this integration step.
        state_prior = state
        derivatives_prior = derivatives

        # Advance one complete step.
        state, derivatives = scheme_grad(state, derivatives, h,)
        new_time = (t0 + (istep.astype(state.x.dtype) + jnp.asarray(1.0, dtype=state.x.dtype)) * h)
        state = replace_state_time(state, new_time,)

        # Check whether g crossed zero during this step.
        state, derivatives, transit_output = (
            detect_transit_grad(
                state_current=state,
                derivatives_current=derivatives,
                state_prior=state_prior,
                derivatives_prior=derivatives_prior,
                output=transit_output,
                h=h,
                scheme_grad=scheme_grad,
            ))

        return (state, derivatives, transit_output,)

    return lax.fori_loop(0, nsteps, body_fun, (s, d, output),)

@partial(jax.jit, static_argnames=("scheme_no_grad","nsteps"),)
def integrate_transit_output_no_grad(s, output, *, scheme_no_grad, h, nsteps,):
    """
    Integrate without propagating Jacobians.
    scheme_no_grad must have the interface:
        s_new = scheme_no_grad(s, h)
    """

    t0 = state_time(s)
    output = initialize_gsave_vmap(s, output,)

    def body_fun(istep, carry):
        state, transit_output = carry
        state_prior = state
        state = scheme_no_grad(state,h,)

        new_time = (t0+ (istep.astype(state.x.dtype) + jnp.asarray(1.0, dtype=state.x.dtype))* h)
        state = replace_state_time(state,new_time,)
        state, transit_output = detect_transit_no_grad(
            state_current=state,
            state_prior=state_prior,
            output=transit_output,
            h=h,
            scheme_no_grad=scheme_no_grad,)

        return state, transit_output

    return lax.fori_loop(0, nsteps, body_fun, (s, output),)

def state_time(s):
    """Return the scalar time stored in State."""
    if s.t.ndim == 0:
        return s.t
    return s.t[0]

def replace_state_time(s, new_time):
    """Return a new State with an updated time."""

    if s.t.ndim == 0:
        new_t = jnp.asarray(new_time,dtype=s.t.dtype,)
    else:
        new_t = s.t.at[0].set(new_time)

    return replace(s, t=new_t,)

def detect_transit_no_grad(*, state_current, state_prior, output, h, scheme_no_grad,):
    """
    Detect transit candidates in the full step from state_prior to
    state_current and record any refined transit events.

    This function does not alter the main integration state.
    """
    rstar = jnp.asarray(1.0e12, dtype=state_current.x.dtype,)

    updated_output = output

    for occultor in output.occs:
        gi = g_func(output.ti, occultor,state_current.x, state_current.v,)

        g_previous = updated_output.gsave[occultor]
        ri = jnp.linalg.norm(state_current.x[:, occultor])
        candidate = ((gi > 0.0) & (g_previous < 0.0)& (-state_current.x[2, occultor] > 0.25 * ri) & (ri < rstar))

        old_count = updated_output.count[occultor]
        new_count = (old_count + candidate.astype(updated_output.count.dtype))

        updated_output = replace(
            updated_output,
            count=updated_output.count.at[occultor].set(new_count),
            gsave=updated_output.gsave.at[occultor].set(gi),)

        should_store = (candidate & (new_count <= updated_output.ntt))
        storage_index = jnp.maximum(new_count - 1,0,)

        denominator = gi - g_previous
        eps = jnp.finfo(gi.dtype).eps

        safe_denominator = jnp.where(jnp.abs(denominator) > eps,
            denominator,
            jnp.where(denominator >= 0.0, eps, -eps,),)

        # Julia:
        # dt0 = -gi * intr.h / (gi - gsave)
        dt_initial = (-g_previous * jnp.asarray(h, dtype=gi.dtype) / safe_denominator)

        updated_output = lax.cond(
            should_store,
            lambda current_output: record_transit_no_grad(
                state_prior=state_prior,
                output=current_output,
                occultor=occultor,
                storage_index=storage_index,
                dt_initial=dt_initial,
                scheme_no_grad=scheme_no_grad,
            ),
            lambda current_output: current_output,
            updated_output,
        )

    return state_current, updated_output

def record_transit_no_grad(*, state_prior, output, occultor, storage_index, dt_initial, scheme_no_grad,):
    state_transit, dt_transit = find_transit_no_grad(
        state_prior,
        transited_body=output.ti,
        occultor=occultor,
        dt_initial=dt_initial,
        scheme_no_grad=scheme_no_grad,
    )

    transit_time = (state_prior.t[0] + dt_transit)

    if isinstance(output, TransitTiming):
        new_tt = output.tt.at[occultor, storage_index,].set(transit_time)
        return replace(output, tt=new_tt,)

    if isinstance(output, TransitParameters):
        vsky = calc_vsky(state_transit.v, output.ti, occultor,)
        bsky2 = calc_bsky2(state_transit.x, output.ti, occultor,)

        new_ttbv = (output.ttbv.at[0, occultor, storage_index].set(transit_time)
            .at[1, occultor, storage_index].set(vsky)
            .at[2, occultor, storage_index].set(bsky2))

        return replace(output, ttbv=new_ttbv,)

    raise TypeError("output must be TransitTiming or TransitParameters.")

def detect_transit_grad( *, state_current, state_prior, derivatives_current, output, h, scheme_grad,):
    """
    Detect and record gradient-enabled transit events occurring between
    state_prior and state_current.

    This function updates only the transit output. It does not replace
    the main integration state with the temporary Newton-search state.

    Parameters
    ----------
    state_current
        State after one complete integration step.

    state_prior
        State at the beginning of that step.

    derivatives_current
        Current Derivatives PyTree. It is used as a structural template
        by the Newton transit refinement.

    output
        TransitTiming or TransitParameters.

    h
        Full integration step used between state_prior and state_current.

    scheme_grad
        Gradient-enabled integration scheme.

    Returns
    -------
    Updated transit output.
    """
    dtype = state_current.x.dtype

    rstar = jnp.asarray(1.0e12, dtype=dtype,)

    updated_output = output

    # output.occs should be static, preferably a Python tuple.
    for occultor in output.occs:
        gi = g_func(output.ti, occultor, state_current.x, state_current.v,)
        g_previous = updated_output.gsave[occultor]

        # This reproduces the Julia implementation:
        # ri = sqrt(x^2 + y^2 + z^2)
        ri = jnp.sqrt(jnp.sum(state_current.x[:, occultor] ** 2))

        in_front = (- state_current.x[2, occultor] > jnp.asarray(0.25, dtype=dtype) * ri)
        candidate = ((gi > 0.0) & (g_previous < 0.0) & in_front & (ri < rstar))

        old_count = updated_output.count[occultor]
        new_count = ( old_count+ candidate.astype(updated_output.count.dtype) )

        new_count_array = updated_output.count.at[occultor].set(new_count)
        new_gsave = updated_output.gsave.at[occultor].set(gi)

        updated_output = replace(updated_output,count=new_count_array, gsave=new_gsave,)

        # Julia:
        # if tt.count[i] <= tt.ntt
        # new_count is one-based conceptually, while storage_index below is zero-based.
        should_store = (candidate& (new_count <= updated_output.ntt))
        storage_index = jnp.maximum(new_count - 1,jnp.asarray(0,dtype=new_count.dtype,),)

        denominator = gi - g_previous
        eps = jnp.finfo(dtype).eps

        safe_denominator = jnp.where(jnp.abs(denominator) > eps,
            denominator,
            jnp.where(
                denominator >= 0.0,
                eps,
                -eps,
            ),)

        # Linear-interpolation initial estimate.
        #
        # Julia:
        # dt0 = -gi * intr.h / (gi - tt.gsave[i])
        dt_initial = (-g_previous * jnp.asarray(h, dtype=dtype) / safe_denominator)

        updated_output = lax.cond(
            should_store,
            lambda current_output: record_transit_grad(
                state_prior=state_prior,
                derivatives_template=derivatives_current,
                output=current_output,
                occultor=occultor,
                storage_index=storage_index,
                dt_initial=dt_initial,
                scheme_grad=scheme_grad,
            ),
            lambda current_output: current_output,
            updated_output,
        )

    return state_current, derivatives_current,updated_output

def record_transit_grad(*, state_prior, derivatives_template, output, occultor, storage_index, dt_initial, scheme_grad,):
    """
    Refine and record one detected transit with derivatives.

    Parameters
    ----------
    state_prior
        State at the beginning of the enclosing full integration step.

    derivatives_template
        Derivatives PyTree with the correct structure. It is zeroed
        before integrating from state_prior to the transit offset.

    output
        TransitTiming or TransitParameters.

    occultor
        Index of the transiting body.

    storage_index
        Zero-based index at which this event is stored.

    dt_initial
        Initial transit-time offset relative to state_prior.

    scheme_grad
        Gradient-enabled one-step integration scheme:

            state_new, derivatives_new = scheme_grad(
                state,
                derivatives,
                h,
            )

    Returns
    -------
    Updated TransitTiming or TransitParameters object.
    """
    state_transit, derivatives_transit, dt_transit = (
        find_transit_grad(
            state_prior,
            derivatives_template,
            transited_body=output.ti,
            occultor=occultor,
            dt_initial=dt_initial,
            scheme_grad=scheme_grad,
        )
    )

    transit_time = state_prior.t[0] + dt_transit

    if isinstance(output, TransitTiming):
        _, derivatives = dtbvdq(
            state_transit,
            output.ti,
            occultor,
            include_bv=False,
        )

        # derivatives has shape (1, 7, n).
        dtdq = derivatives[0]
        new_tt = output.tt.at[occultor,storage_index,].set(transit_time)
        new_dtdq0 = output.dtdq0.at[occultor,storage_index,:,:,].set(dtdq)

        return replace(output,tt=new_tt,dtdq0=new_dtdq0,)

    if isinstance(output, TransitParameters):
        observables, derivatives = dtbvdq(
            state_transit,
            output.ti,
            occultor,
            include_bv=True,
        )

        vsky = observables[0]
        bsky2 = observables[1]

        # derivatives shape: (3, 7, n)
        new_ttbv = (output.ttbv
                        .at[0, occultor, storage_index].set(transit_time)
                        .at[1, occultor, storage_index].set(vsky)
                        .at[2, occultor, storage_index].set(bsky2))

        new_dtbvdq0 = output.dtbvdq0.at[ :, occultor, storage_index, :, :,].set(derivatives)

        return replace(output, ttbv=new_ttbv, dtbvdq0=new_dtbvdq0,)

    raise TypeError("output must be TransitTiming or TransitParameters, not {type(output).__name__}.")