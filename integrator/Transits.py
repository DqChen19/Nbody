from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Any
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from .State import State
# ==========================================
# Transit Timing class, jitted
# ==========================================
@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class TransitTiming:
    '''
    Transit times and derivatives.
    '''
    # the transit time of each body; ntt: number of tranist times
    tt: jax.Array               # (n, ntt)
    # derivatives of the transit times with respect to the initial Cartesian coordinates and masses
    dtdq0: jax.Array            # (n, ntt, 7, n)
    # derivatives of the transit times with respect to the initial orbitial elements and masses
    dtdelements: jax.Array      # (n, ntt, 7, n)

    # Runtime arrays
    count: jax.Array            # count tranist times of each body
    dtdq: jax.Array             # \partial t_transit / \partial q ->
    gsave: jax.Array            # save the g_func()
    s_prior: Any                # last state before transit happen, position, velocity, time, Jacobian, mass, other Integration state
    s_transit: Any              # transit state

    # Static metadata
    ntt: int                    # number of transit time
    ti: int                     # Index of the body with respect to which transits are measured. (Default is the central body)
    occs: tuple[int, ...]       # the idx of all bodies that potentially show transit

    def tree_flatten(self):
        children = (self.tt, self.dtdq0, self.dtdelements, self.count,
                    self.dtdq, self.gsave, self.s_prior, self.s_transit,)
        aux_data = (self.ntt, self.ti, self.occs,)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        ntt, ti, occs = aux_data

        (tt, dtdq0, dtdelements, count, dtdq, gsave, s_prior, s_transit,) = children

        return cls(tt=tt, dtdq0=dtdq0, dtdelements=dtdelements, count=count, dtdq=dtdq, gsave=gsave,
                s_prior=s_prior, s_transit=s_transit, ntt=ntt, ti=ti, occs=occs,)

def estimate_ntt(tmax: float, periods) -> int:
    '''
    calculate the max number of transit times
    '''
    periods_np = jnp.asarray(periods)

    ratios = tmax / periods_np
    finite = jnp.isfinite(ratios)

    if not jnp.any(finite):
        raise ValueError("Cannot allocate transit arrays: no finite periods.")

    return int(jnp.max(jnp.ceil(jnp.abs(ratios[finite])))+ 3) # Buffer

#Create tranist timing
def create_transit_timing(tmax: float, ic, ti: int = 0, ) -> TransitTiming:
    n = ic.nbody
    # Julia elements[:, 2] -> Python elements[:, 1]
    periods = jnp.asarray(ic.elements)[:, 1]
    ntt = estimate_ntt(tmax, periods)
    dtype = jnp.asarray(ic.elements).dtype

    # Julia setdiff(collect(1:n), ti)
    occs = tuple(i for i in range(n) if i != ti)
    base_state = State.from_ic(ic)

    return TransitTiming(
        tt=jnp.zeros((n, ntt), dtype=dtype,),
        dtdq0=jnp.zeros((n, ntt, 7, n), dtype=dtype,),
        dtdelements=jnp.zeros((n, ntt, 7, n), dtype=dtype,),
        count=jnp.zeros((n,), dtype=jnp.int32,),
        dtdq=jnp.zeros((1, 7, n), dtype=dtype,),
        gsave=jnp.zeros((n,), dtype=dtype,),
        s_prior=base_state, s_transit=base_state,
        ntt=ntt, ti=int(ti), occs=occs,)

def zero_transit_timing(output: TransitTiming,) -> TransitTiming:
    return replace(output, tt=jnp.zeros_like(output.tt),
        dtdq0=jnp.zeros_like(output.dtdq0),
        dtdelements=jnp.zeros_like(output.dtdelements),
        count=jnp.zeros_like(output.count),
        dtdq=jnp.zeros_like(output.dtdq),
        gsave=jnp.zeros_like(output.gsave),)#

# ==========================================
# Transit Parameters class, jitted
# ==========================================
@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True) #immutable data class, cannot reattribute the value
class TransitParameters:
    '''
    Transit times, impact parameters, sky-velocities, and derivatives.
    '''
    #The transit times, impact parameter, and sky-velocity of each body.
    ttbv: jax.Array
    # Derivatives of the transit times, impact parameters, and sky-velocities 
    # with respect to the initial Cartesian coordinates and masses.
    dtbvdq0: jax.Array 
    # Derivatives of the transit times, impact parameters, and sky-velocities 
    # with respect to the initial orbital elements and masses.
    dtbvdelements: jax.Array

    # Runtime arrays
    count: jax.Array
    dtbvdq: jax.Array
    gsave: jax.Array
    s_prior: Any
    s_transit: Any

    # Static metadata
    ntt: int
    ti: int
    occs: tuple[int, ...]

    def tree_flatten(self):
        children = (self.ttbv, self.dtbvdq0, self.dtbvdelements, self.count,
                    self.dtbvdq, self.gsave, self.s_prior, self.s_transit,)
        aux_data = (self.ntt, self.ti, self.occs,)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        ntt, ti, occs = aux_data
        (ttbv, dtbvdq0, dtbvdelements, count, dtbvdq, gsave, s_prior, s_transit,) = children

        return cls(ttbv=ttbv, dtbvdq0=dtbvdq0, dtbvdelements=dtbvdelements, count=count, dtbvdq=dtbvdq,
            gsave=gsave, s_prior=s_prior, s_transit=s_transit, ntt=ntt, ti=ti, occs=occs,)
    
def create_transit_parameters(tmax: float, ic, ti: int = 0) -> TransitParameters:
    n = ic.nbody
    periods = jnp.asarray(ic.elements)[:, 1]
    ntt = estimate_ntt(tmax, periods)
    dtype = jnp.asarray(ic.elements).dtype
    occs = tuple(i for i in range(n) if i != ti)
    base_state = State.from_ic(ic)

    return TransitParameters(ttbv=jnp.zeros((3, n, ntt), dtype=dtype,),
        dtbvdq0=jnp.zeros((3, n, ntt, 7, n), dtype=dtype,),
        dtbvdelements=jnp.zeros((3, n, ntt, 7, n), dtype=dtype,),
        count=jnp.zeros((n,), dtype=jnp.int32,),
        dtbvdq=jnp.zeros((3, 7, n), dtype=dtype,),
        gsave=jnp.zeros((n,), dtype=dtype,),
        s_prior=base_state, s_transit=base_state,
        ntt=ntt, ti=ti, occs=occs,)

def zero_transit_parameters(output: TransitParameters,) -> TransitParameters:
    return replace(output,
        ttbv=jnp.zeros_like(output.ttbv),
        dtbvdq0=jnp.zeros_like(output.dtbvdq0),
        dtbvdelements=jnp.zeros_like(output.dtbvdelements),
        count=jnp.zeros_like(output.count),
        dtbvdq=jnp.zeros_like(output.dtbvdq),
        gsave=jnp.zeros_like(output.gsave),)

# ==========================================
# Transit Snapshot class, jitted
# ==========================================
@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class TransitSnapshot:
    '''
    Given specific times check sky-plane geometry
    '''
    times:          jax.Array
    bsky2:          jax.Array
    vsky:           jax.Array
    dbvdq0:         jax.Array
    dbvdelements:   jax.Array
    
    nt: int

    def tree_flatten(self):
        children = (self.times, self.bsky2, self.vsky, self.dbvdq0, self.dbvdelements,)
        aux_data = (self.nt,)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (nt,) = aux_data
        (times, bsky2, vsky, dbvdq0, dbvdelements,) = children

        return cls(times=times, bsky2=bsky2, vsky=vsky,
            dbvdq0=dbvdq0, dbvdelements=dbvdelements, nt=nt,)

def create_transit_snapshot(times, ic,) -> TransitSnapshot:
    times = jnp.asarray(times)

    n = ic.nbody
    nt = times.shape[0]
    dtype = times.dtype

    return TransitSnapshot(times=times,
        bsky2=jnp.zeros((n, nt), dtype=dtype,),
        vsky=jnp.zeros((n, nt), dtype=dtype,),
        dbvdq0=jnp.zeros((2, n, nt, 7, n), dtype=dtype,),
        dbvdelements=jnp.zeros((2, n, nt, 7, n), dtype=dtype,),
        nt=nt,)

def zero_transit_snapshot(output: TransitSnapshot,) -> TransitSnapshot:
    return replace(output,
        bsky2=jnp.zeros_like(output.bsky2),
        vsky=jnp.zeros_like(output.vsky),
        dbvdq0=jnp.zeros_like(output.dbvdq0),
        dbvdelements=jnp.zeros_like(output.dbvdelements),)

