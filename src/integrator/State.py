from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp

from ..ics.init_nbody import init_nbody
from ..ics.initial_conditions import InitialConditions
from ..pre_alloc_arrays import Derivatives
# ===================================
# State class
# ===================================
Array = jax.Array

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class State:
    """
    Current state of simulation.
    
    Array fields are dynamic PyTree leaves. The integer n is static metadata.
    """
    # User-facing state
    x: Array # Positions of each body [dimension, body].
    v: Array # Velocities of each body [dimension, body].
    t: Array # Current time of simulation.
    m: Array # Masses of each body.
    jac_step: Array # Current Jacobian.
    dqdt: Array  # Derivative with respect to time.
    jac_init: Array

    # Internal state
    pair: tuple  # Static: fixed hierarchy structure, never mutated during integration.
    xerror: Array
    verror: Array
    dqdt_error: Array
    jac_error: Array

    rij: Array
    a: Array
    aij: Array
    x0: Array
    v0: Array
    input: Array
    delxv: Array
    rtmp: Array

    # Static metadata
    n: int

    @classmethod
    def from_ic(cls, ic: InitialConditions) -> "State":
        x, v, jac_init = init_nbody(ic)
        # Convert everything entering JAX kernels to JAX arrays.
        x = jnp.asarray(x)
        v = jnp.asarray(v)
        jac_init = jnp.asarray(jac_init)

        dtype = x.dtype
        n = int(ic.nbody)
        sevn = 7 * n

        return cls(
            x=x, v=v,
            t=jnp.asarray([ic.t0], dtype=dtype),
            m=jnp.asarray(ic.m, dtype=dtype),
            jac_step=jnp.eye(sevn, dtype=dtype),
            dqdt=jnp.zeros(sevn, dtype=dtype),
            jac_init=jac_init,
            pair=tuple(tuple(False for _ in range(n)) for _ in range(n)),
            xerror=jnp.zeros_like(x),
            verror=jnp.zeros_like(v),
            dqdt_error=jnp.zeros(sevn, dtype=dtype),
            jac_error=jnp.zeros((sevn, sevn), dtype=dtype),
            rij=jnp.zeros(3, dtype=dtype),
            a=jnp.zeros((3, n), dtype=dtype),
            aij=jnp.zeros(3, dtype=dtype),
            x0=jnp.zeros(3, dtype=dtype),
            v0=jnp.zeros(3, dtype=dtype),
            input=jnp.zeros(8, dtype=dtype),
            delxv=jnp.zeros(6, dtype=dtype),
            rtmp=jnp.zeros(3, dtype=dtype),
            n=n,
        )

    def tree_flatten(self):
        children = (self.x, self.v, self.t, self.m, self.jac_step, self.dqdt, self.jac_init,
                    self.xerror, self.verror, self.dqdt_error, self.jac_error, self.rij,
                    self.a, self.aij, self.x0, self.v0, self.input, self.delxv, self.rtmp,)

        # pair (fixed hierarchy structure, never mutated during integration)
        # and n (determines array interpretation) are static metadata.
        aux_data = (self.pair, self.n)

        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        pair, n = aux_data

        (x, v, t, m, jac_step, dqdt, jac_init,
            xerror, verror, dqdt_error, jac_error,
            rij, a, aij, x0, v0,
            input_array, delxv, rtmp,) = children

        return cls(x=x, v=v, t=t, m=m, jac_step=jac_step, dqdt=dqdt,
            jac_init=jac_init, pair=pair, xerror=xerror, verror=verror,
            dqdt_error=dqdt_error, jac_error=jac_error, rij=rij, a=a,
            aij=aij, x0=x0, v0=v0, input=input_array,
            delxv=delxv, rtmp=rtmp, n=n,)

    def copy_from(self, other: "State") -> "State":
        """
        JAX replacement for Julia set_state!.

        Since State is immutable, return a new State.
        """
        return replace(
            self, t=other.t, x=other.x, v=other.v, jac_step=other.jac_step,
            xerror=other.xerror, verror=other.verror, jac_error=other.jac_error,
            dqdt=other.dqdt, dqdt_error=other.dqdt_error,)

    def initialize(self, ic: InitialConditions) -> "State":
        """
        Reset physical initial conditions without mutating this State.
        """
        x, v, jac_init = init_nbody(ic)

        return replace(
                        self,
                        x=jnp.asarray(x, dtype=self.x.dtype),
                        v=jnp.asarray(v, dtype=self.v.dtype),
                        t=jnp.asarray([ic.t0], dtype=self.t.dtype),
                        m=jnp.asarray(ic.m, dtype=self.m.dtype),
                        jac_init=jnp.asarray(
                            jac_init,
                            dtype=self.jac_init.dtype,
                        ),
                        xerror=jnp.zeros_like(self.xerror),
                        verror=jnp.zeros_like(self.verror),
                    )
    
def dState(ic: InitialConditions):
    return State.from_ic(ic), Derivatives.create(ic.nbody)