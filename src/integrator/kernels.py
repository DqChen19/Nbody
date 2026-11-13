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
    values = jax.vmap(lambda i: g_func(i, output.ti, s.x, s.v,))(occs)
    return replace(output, gsave=output.gsave.at[occs].set(values),)

# ================================
# transit loop
# ================================
@partial(jax.jit,static_argnames=("scheme_grad", "scheme_no_grad",),)
def integrate_transit_output_grad(s, d, output, *, scheme_grad, h, nsteps, scheme_no_grad=None,):
    """
    Input:
        s: Current state, [xi, vi, mi] (7n,)
        d: derivatives
        output: transit information/structure

    Integrate for a fixed number of steps and detect transits.
    scheme_grad must have the interface:

        s_new, d_new, transit_new = scheme_grad(s, d, h)
    """
    t0 = state_time(s)
    output = initialize_gsave_vmap(s,output,)

    def body_fun(istep, carry):
        state, derivatives, transit_output = carry

        # State before this integration step.
        state_prior = state
        #derivatives_prior = derivatives

        # Advance one complete step. update s an d with J
        state, derivatives = scheme_grad(state, derivatives, h,) 
        
        new_time = (t0 + (istep.astype(state.x.dtype) + jnp.asarray(1.0, dtype=state.x.dtype)) * h)
        state = replace_state_time(state, new_time,)

        # Check whether g crossed zero during this step.
        # Detect if there is a potential transit
        state, derivatives, transit_output = (detect_transit_grad(
                state_current=state,
                derivatives_current=derivatives,
                state_prior=state_prior,
                output=transit_output,
                h=h,
                scheme_grad=scheme_grad,
                scheme_no_grad=scheme_no_grad,
            ))

        return (state, derivatives, transit_output,)

    return lax.fori_loop(0, nsteps, body_fun, (s, d, output),)

@partial(jax.jit, static_argnames=("scheme_no_grad",),)
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

def _transit_candidates(state_current, output, h):
    """
    Evaluate the transit test for every occultor at once.

    Returns per-occultor arrays (ordered as output.occs) so that the
    expensive Newton refinement can be traced once and driven by a traced
    occultor index, instead of being unrolled once per occultor.
    """
    x = state_current.x
    v = state_current.v
    rstar = jnp.asarray(1.0e12, dtype=x.dtype,)

    occs = jnp.asarray(output.occs, dtype=jnp.int32,)
    gi = jax.vmap(lambda j: g_func(output.ti, j, x, v,))(occs)
    g_previous = output.gsave[occs]

    # Julia: ri = sqrt(x^2 + y^2 + z^2)
    ri = jnp.sqrt(jnp.sum(x[:, occs] ** 2, axis=0))
    in_front = (-x[2, occs] > jnp.asarray(0.25, dtype=x.dtype) * ri)
    candidate = ((gi > 0.0) & (g_previous < 0.0) & in_front & (ri < rstar))

    # new_count is one-based conceptually, storage_index is zero-based.
    new_count = (output.count[occs] + candidate.astype(output.count.dtype))
    should_store = (candidate & (new_count <= output.ntt))
    storage_index = jnp.maximum(new_count - 1, 0,)

    # Julia: dt0 = -gi * intr.h / (gi - gsave[i])
    dt_initial = (-gi * h / (gi - g_previous))

    return occs, gi, new_count, should_store, storage_index, dt_initial

def _record_pending(output, occs, should_store, storage_index, dt_initial, record_fn):
    """
    Refine and record every flagged occultor, one at a time in ascending
    order, inside a single lax.while_loop. record_fn(output, occultor,
    storage_index, dt_initial) is traced once regardless of len(occs).
    """
    def cond_fun(carry):
        pending, _ = carry
        return jnp.any(pending)

    def body_fun(carry):
        pending, current_output = carry
        k = jnp.argmax(pending)
        current_output = record_fn(current_output, occs[k], storage_index[k], dt_initial[k],)
        return pending.at[k].set(False), current_output

    _, output = lax.while_loop(cond_fun, body_fun, (should_store, output),)
    return output

def detect_transit_no_grad(*, state_current, state_prior, output, h, scheme_no_grad,):
    """
    Detect transit candidates in the full step from state_prior to
    state_current and record any refined transit events.

    This function does not alter the main integration state.
    """
    occs, gi, new_count, should_store, storage_index, dt_initial = _transit_candidates(state_current, output, h,)

    updated_output = _record_pending(
        output, occs, should_store, storage_index, dt_initial,
        lambda current_output, occultor, index, dt0: record_transit_no_grad(
            state_anchor=state_current,
            output=current_output,
            occultor=occultor,
            storage_index=index,
            dt_initial=dt0,
            scheme_no_grad=scheme_no_grad,
        ),
    )
    updated_output = replace(updated_output,
                count=updated_output.count.at[occs].set(new_count),
                gsave=updated_output.gsave.at[occs].set(gi),)

    return state_current, updated_output

def record_transit_no_grad(*, state_anchor, output, occultor, storage_index, dt_initial, scheme_no_grad,):
    state_transit, dt_transit = find_transit_no_grad(
        state_anchor,
        transited_body=output.ti,
        occultor=occultor,
        dt_initial=dt_initial,
        scheme_no_grad=scheme_no_grad,
    )

    transit_time = (state_anchor.t[0] + dt_transit)

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

def detect_transit_grad( *, state_current, state_prior, derivatives_current, output, h, scheme_grad, scheme_no_grad=None,):
    """
    Detect and record gradient-enabled transit events occurring between
    state_prior and state_current.

    This function updates only the transit output. It does not replace
    the main integration state with the temporary Newton-search state.

    Parameters
    ----------
    state_current: State after one complete integration step.
    state_prior: State at the beginning of that step.
    derivatives_current: Current Derivatives PyTree. It is used as a structural template
        by the Newton transit refinement.
    output: TransitTiming or TransitParameters.
    h: Full integration step used between state_prior and state_current.
    scheme_grad: Gradient-enabled integration scheme.

    Returns
    -------
    Updated transit output: state_current, derivatives_current,updated_output
    """
    occs, gi, new_count, should_store, storage_index, dt_initial = _transit_candidates(state_current, output, h,)

    updated_output = replace(output, count=output.count.at[occs].set(new_count), gsave=output.gsave.at[occs].set(gi),)

    updated_output = _record_pending(updated_output, occs, should_store, storage_index, dt_initial,
        lambda current_output, occultor, index, dt0: record_transit_grad(
            state_anchor=state_current,
            derivatives_template=derivatives_current,
            output=current_output,
            occultor=occultor,
            storage_index=index,
            dt_initial=dt0,
            scheme_grad=scheme_grad,
            scheme_no_grad=scheme_no_grad,
        ),)

    return state_current, derivatives_current,updated_output

def record_transit_grad(*, state_anchor, derivatives_template, output, occultor, storage_index, dt_initial, scheme_grad, scheme_no_grad=None,):
    """
    Refine and record one detected transit with derivatives.

    Parameters
    ----------
    state_prior
        State at the beginning of the enclosing full integration step.

    derivatives_template
        Derivatives PyTree with the correct structure. It is zeroed
        before integrating from state_prior to the transit offset.

    output: TransitTiming or TransitParameters.
    occultor: Index of the transiting body.
    storage_index: Zero-based index at which this event is stored.

    dt_initial: Initial transit-time offset relative to state_prior.

    scheme_grad: Gradient-enabled one-step integration scheme:
            state_new, derivatives_new = scheme_grad(state, derivatives,h,)

    Returns
    -------
    Updated TransitTiming or TransitParameters object.
    """
    state_transit, _, dt_transit = (
        find_transit_grad(
            state_anchor,
            derivatives_template,
            transited_body=output.ti,
            occultor=occultor,
            dt_initial=dt_initial,
            scheme_grad=scheme_grad,
            scheme_no_grad=scheme_no_grad,))

    transit_time = state_anchor.t[0] + dt_transit

    if isinstance(output, TransitTiming):
        _, derivatives = dtbvdq(state_transit, output.ti,occultor, include_bv=False,)

        # derivatives has shape (1, 7, n).
        dtdq = derivatives[0]
        new_tt = output.tt.at[occultor,storage_index,].set(transit_time)
        new_dtdq0 = output.dtdq0.at[occultor,storage_index,:,:,].set(dtdq)

        return replace(output, tt=new_tt,dtdq0=new_dtdq0, dtdq = dtdq)

    if isinstance(output, TransitParameters):
        observables, derivatives = dtbvdq(state_transit, output.ti, occultor, include_bv=True,)

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

# ====================================
# Reverse-mode (adjoint) transit loop
# ====================================
SLOT_BATCH = 256   # transit slots per batch in the differentiable refinement (phase 2)

@partial(jax.jit, static_argnames=("scheme_no_grad", "nsteps"),)
def integrate_transit_output_reverse(s, output, *, scheme_no_grad, h, nsteps, slots=None):
    """
    Reverse-mode differentiable counterpart of integrate_transit_output_no_grad.

    jax.grad / jax.vjp of output.tt with respect to s.x, s.v and s.m work
    through this kernel. It records only TransitTiming.tt and count (no dtdq0).

    Phase 1 (lax.scan over a static nsteps, jax.checkpoint-ed step body):
        integrate, detect transits and Newton-refine them exactly as the no-grad
        kernel. Each step returns its state (x, v, compensation errors) as a scan output. 
        The backward pass stores only the small carry per step and recomputes the step.
        
    Phase 2 (after the loop, vmapped over all (occultor, transit) slots):
        tt = t_step + dt_root - g(scheme(anchor, dt_root)) / gdot,
        one differentiable Newton step from the stored anchor state. Its value
        equals the refined root (g = 0 there); its derivative is the
        implicit-function result -(dg/dq) / gdot, as in dtbvdq.

    slots: optional (rows, epochs) int arrays, the output.tt entries that need
        gradients (e.g. the observed transits). Phase 2 then refines only those;
        every other transit keeps its Newton-root time with no gradient. rows are
        output.tt row (body) indices and must be occultors. None refines every slot.

    Keeping every differentiable transit computation out of the scan matters:
    a lax.cond inside a differentiated step keeps the residuals of both branches,
    which would cost one refinement per occultor at every step.
    """
    t0 = state_time(s)
    output = initialize_gsave_vmap(s, output,)
    occs = jnp.asarray(output.occs, dtype=jnp.int32,)
    nocc = len(output.occs)
    dtype = s.x.dtype

    def body_fun(carry, istep):
        x, v, xerror, verror, gsave, count = carry
        state = scheme_no_grad(replace(s, x=x, v=v, xerror=xerror, verror=verror), h,)

        new_time = (t0 + (istep.astype(dtype) + jnp.asarray(1.0, dtype=dtype)) * h)
        state = replace_state_time(state, new_time,)
        fixed_state = lax.stop_gradient(state)

        _, gi, new_count, should_store, storage_index, dt_initial = _transit_candidates(
            fixed_state, replace(output, gsave=gsave, count=count), h,)

        def refine(k):
            return lax.cond(should_store[k],
                lambda _: transit_root_and_slope(fixed_state, transited_body=output.ti, occultor=occs[k],
                                                 dt_initial=dt_initial[k], scheme_no_grad=scheme_no_grad,),
                lambda _: (jnp.zeros((), dtype=dtype), jnp.ones((), dtype=dtype)), operand=None,)

        dt_root, gdot = lax.map(refine, jnp.arange(nocc))

        carry = (state.x, state.v, state.xerror, state.verror, gsave.at[occs].set(gi), count.at[occs].set(new_count),)
        return carry, ((state.x, state.v, state.xerror, state.verror), dt_root, gdot, should_store, storage_index)

    init = (s.x, s.v, s.xerror, s.verror, output.gsave, output.count,)
    (x, v, xerror, verror, gsave, count), (anchors, dt_root, gdot, stored, index) = lax.scan(jax.checkpoint(body_fun), init, jnp.arange(nsteps),)

    # (nsteps, nocc) step outputs -> (nocc, ntt) transit slots; steps without a transit are dropped.
    occ_k = jnp.broadcast_to(jnp.arange(nocc), stored.shape)
    cols = jnp.where(stored, index, output.ntt)

    def to_slots(values, fill):
        # Create a (nocc, output.ntt) and write the values into position [occ_k, cols]
        return jnp.full((nocc, output.ntt), fill, dtype=values.dtype).at[occ_k, cols].set(values, mode="drop")

    step_slot = to_slots(jnp.broadcast_to(jnp.arange(nsteps)[:, None], stored.shape), 0)
    filled = to_slots(stored, False)
    dt_slot = to_slots(dt_root, 0.0)
    gdot_slot = to_slots(gdot, 1.0)

    def slot_time(step, dt, slope, occultor):
        anchor = replace(s, x=anchors[0][step], v=anchors[1][step], xerror=anchors[2][step], verror=anchors[3][step],)
        s_root = scheme_no_grad(anchor, dt,)
        gsky = g_func(output.ti, occultor, s_root.x, s_root.v,)
        anchor_time = t0 + (step.astype(dtype) + jnp.asarray(1.0, dtype=dtype)) * h
        return anchor_time + dt - gsky / slope

    # Batched lax.map over the flattened slots, with each slot's step checkpointed: storing
    # its residuals instead would cost ~55 KB per slot (~0.4 GB for TRAPPIST-1).
    occ_slot = jnp.broadcast_to(occs[:, None], (nocc, output.ntt))
    if slots is None:
        times = lax.map(jax.checkpoint(lambda args: slot_time(*args)),
                        (step_slot.reshape(-1), dt_slot.reshape(-1), gdot_slot.reshape(-1), occ_slot.reshape(-1)),
                        batch_size=SLOT_BATCH,).reshape(nocc, output.ntt)
    else:
        # Newton-root times (no gradient) everywhere, then the differentiable step for the requested slots only
        rows, epochs = (jnp.asarray(a, dtype=jnp.int32) for a in slots)
        k_sel = jnp.argmax(occs[None, :] == rows[:, None], axis=1)
        e_sel = jnp.clip(epochs, 0, output.ntt - 1)
        root_times = t0 + (step_slot.astype(dtype) + jnp.asarray(1.0, dtype=dtype)) * h + dt_slot
        times_sel = lax.map(jax.checkpoint(lambda args: slot_time(*args)),
                            (step_slot[k_sel, e_sel], dt_slot[k_sel, e_sel], gdot_slot[k_sel, e_sel], occ_slot[k_sel, e_sel]),
                            batch_size=SLOT_BATCH,)
        times = lax.stop_gradient(root_times).at[k_sel, e_sel].set(times_sel)
    tt = output.tt.at[occs].set(jnp.where(filled, times, output.tt[occs]))

    s = replace_state_time(replace(s, x=x, v=v, xerror=xerror, verror=verror), t0 + nsteps * h,)
    return s, replace(output, tt=tt, count=count, gsave=gsave,)

def transit_root_and_slope(state_anchor, *, transited_body, occultor, dt_initial, scheme_no_grad,):
    """
    Newton root dt of g(scheme(state_anchor, dt)) and its slope gdot there (no gradients).

    gdot must be the exact dt-derivative of g along the integrator map, since it sets
    the transit-time derivative -(dg/dq) / gdot. gd_func with the no-grad dqdt (plain
    Newtonian accelerations) is only accurate to the integrator error (~1e-6 relative),
    enough for Newton to converge but not for the derivative, so gdot is a jvp in dt.
    """
    _, dt_root = find_transit_no_grad(state_anchor, transited_body=transited_body,
                    occultor=occultor, dt_initial=dt_initial, scheme_no_grad=scheme_no_grad,)

    def g_of_dt(dt):
        s_trial = scheme_no_grad(state_anchor, dt,)
        return g_func(transited_body, occultor, s_trial.x, s_trial.v,)

    _, gdot = jax.jvp(g_of_dt, (dt_root,), (jnp.ones_like(dt_root),))
    return lax.stop_gradient(dt_root), lax.stop_gradient(gdot)
