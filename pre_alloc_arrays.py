from __future__ import annotations
from abc import ABC
from dataclasses import dataclass, fields, replace

import jax
import jax.numpy as jnp

Array = jax.Array

# Abstract type
class PreAllocArrays(ABC):
    pass

class AbstractDerivatives(PreAllocArrays, ABC):
    pass

# Struct Jacobian
@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Jacobian(AbstractDerivatives):
    jac_phi     :   Array # Phi_n to Phi_m: the mapping matrix
    jac_kick    :   Array # for kick step
    jac_copy    :   Array
    jac_ij      :   Array # A pair (i, j) the jacobian matrix
    jac_tmp1    :   Array # chain rule - tmp matrix
    jac_tmp2    :   Array # chain rule - tmp matrix
    jac_err1    :   Array # save the error

    dqdt_ij     :   Array
    dqdt_phi    :   Array
    dqdt_kick   :   Array

    @classmethod
    def create(cls, n:int, dtype = jnp.float64) -> "Jacobian":
        sevn = 7 * n
        return cls(
            jac_phi   = jnp.zeros((sevn, sevn), dtype),
            jac_kick  = jnp.zeros((sevn, sevn), dtype),
            jac_copy  = jnp.zeros((sevn, sevn), dtype),
            jac_ij    = jnp.zeros((14,   14),   dtype),
            jac_tmp1  = jnp.zeros((14,   sevn), dtype),
            jac_tmp2  = jnp.zeros((14,   sevn), dtype),
            jac_err1  = jnp.zeros((14,   sevn), dtype),
            dqdt_ij   = jnp.zeros((14,),        dtype),
            dqdt_phi  = jnp.zeros((sevn,),      dtype),
            dqdt_kick = jnp.zeros((sevn,),      dtype),
        )
    

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class dTime(AbstractDerivatives):

    jac_phi     :   Array
    jac_kick    :   Array
    jac_ij      :   Array
    dqdt_phi    :   Array
    dqdt_kick   :   Array
    dqdt_ij     :   Array
    dqdt_tmp1   :   Array
    dqdt_tmp2   :   Array
    jac_kepler  :   Array
    jac_mass    :   Array

    @classmethod
    def create(cls, n: int, dtype = jnp.float64) -> "dTime":
        sevn = 7 * n
        return cls(
            jac_phi    = jnp.zeros((sevn, sevn), dtype),
            jac_kick   = jnp.zeros((sevn, sevn), dtype),
            jac_ij     = jnp.zeros((14,   14),   dtype),
            dqdt_phi   = jnp.zeros(sevn,          dtype),
            dqdt_kick  = jnp.zeros(sevn,          dtype),
            dqdt_ij    = jnp.zeros(14,            dtype),
            dqdt_tmp1  = jnp.zeros(14,            dtype),
            dqdt_tmp2  = jnp.zeros(14,            dtype),
            jac_kepler = jnp.zeros((6, 8),        dtype),
            jac_mass   = jnp.zeros(6,             dtype),
        )

@jax.tree_util.register_dataclass    
@dataclass(frozen=True)
class Derivatives(AbstractDerivatives):
    jac_phi:   Array
    jac_kick:  Array
    jac_copy:  Array
    jac_ij:    Array
    jac_tmp1:  Array
    jac_tmp2:  Array
    jac_err1:  Array

    dqdt_phi:  Array
    dqdt_kick: Array
    dqdt_ij:   Array
    dqdt_tmp1: Array
    dqdt_tmp2: Array

    jac_kepler:Array
    jac_mass:  Array
    
    dadq:      Array   
    dotdadq:   Array   

    tmp7n:     Array
    tmp14:     Array
    ctime:     Array

    @classmethod
    def create(cls, n: int, dtype=jnp.float64) -> "Derivatives":
        sevn = 7 * n
        return cls(
            jac_phi    = jnp.zeros((sevn, sevn), dtype),
            jac_kick   = jnp.zeros((sevn, sevn), dtype),
            jac_copy   = jnp.zeros((sevn, sevn), dtype),
            jac_ij     = jnp.zeros((14,   14),   dtype),
            jac_tmp1   = jnp.zeros((14,   sevn), dtype),
            jac_tmp2   = jnp.zeros((14,   sevn), dtype),
            jac_err1   = jnp.zeros((14,   sevn), dtype),
            dqdt_phi   = jnp.zeros((sevn,),      dtype),
            dqdt_kick  = jnp.zeros((sevn,),      dtype),
            dqdt_ij    = jnp.zeros((14,),        dtype),
            dqdt_tmp1  = jnp.zeros((14,),        dtype),
            dqdt_tmp2  = jnp.zeros((14,),        dtype),
            jac_kepler = jnp.zeros((6, 8),       dtype),  
            jac_mass   = jnp.zeros((6,),         dtype),
            dadq       = jnp.zeros((3, n, 4, n), dtype),
            dotdadq    = jnp.zeros((4, n),       dtype),
            tmp7n      = jnp.zeros((sevn,),      dtype),
            tmp14      = jnp.zeros((14,),        dtype),
            ctime      = jnp.zeros((1,),         dtype),)
    
    def zero_out(self) -> "Derivatives":
        n = self.tmp7n.shape[0] // 7
        return Derivatives.create(n=n, dtype=self.jac_phi.dtype,)
