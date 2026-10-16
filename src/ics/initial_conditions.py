from __future__ import annotations
from abc import ABC
from dataclasses import dataclass, field, fields
from typing import ClassVar
import numpy as np
import warnings
import jax.numpy as jnp
import jax
import math


# Abstract type
class AbstractInitialConditions(ABC):
    pass

class InitialConditions(ABC):
    pass

# Elements class
@dataclass
class Elements(AbstractInitialConditions):
    '''
    - m: Mass of outer body.
    - P: Period [Days].
    - t0: Initial time of transit [Days].
    - ecos_omega: Eccentricity vector x-component (eccentricity times cosine of the argument of periastron)
    - esin_omega: Eccentricity vector y-component (eccentricity times sine of the argument of periastron)
    - I: Inclination, as measured from sky-plane [Radians].
    - Omega: Longitude of ascending node, as measured from +x-axis [Radians].
    - a: Orbital semi-major axis [AU].
    - e: Eccentricity.
    - omega: Argument of periastron [Radians].
    - tp: Time of periastron passage [Days].
    '''

    m:              float
    P:              float = 0.0
    t0:             float = 0.0
    ecos_omega:     float = 0.0
    esin_omega:     float = 0.0
    I:              float = 0.0
    Omega:          float = 0.0
    a:              float = 0.0
    e:              float = 0.0
    omega:          float = 0.0
    tp:             float = 0.0

    @classmethod
    def from_elements(
        cls,
        m: float, P: float, t0: float, ecos_omega: float, esin_omega: float, I: float, Omega: float,) -> "Elements":
        e     = math.hypot(ecos_omega, esin_omega)
        omega =  math.atan2(esin_omega, ecos_omega)
        return cls(m = float(m),  P = float(P), t0 = float(t0), ecos_omega = float(ecos_omega), esin_omega = float(esin_omega),
                   I = float(I), Omega = float(Omega), a = 0.0, e = e, omega = omega, tp = 0.0)
    
    @classmethod
    def from_kwargs(
        cls, *, m: float, P: float = float("nan"), t0: float = float("nan"), ecos_omega: float = float("nan"), esin_omega: float = float("nan"),
        I: float = float("nan"), Omega: float = float("nan"), a: float = float("nan"), omega: float = float("nan"), e: float = float("nan"), 
        tp: float = float("nan"),) -> "Elements":
        def _isnan(x): return math.isnan(x) #math ? or jnp?
        def _nanall(*xs): return all(_isnan(x) for x in xs)
        def _replacenan(x): return 0.0 if _isnan(x) else x
        
        #Host star
        if _nanall(P, t0, ecos_omega, esin_omega, I, Omega, a, omega, e, tp): #Host star
            return cls(m = m, P = 0.0, t0 = 0.0, ecos_omega = 0.0, esin_omega = 0.0,
                       I = 0.0, Omega = 0.0, a = 0.0, e = 0.0, omega = 0.0, tp = 0.0)

        # Check the parameters
        if _nanall(P, a):
            raise ValueError("Must specify either P (period)  or a (semi-major axis) !")
        if not _isnan(P) and not _isnan(a):
            raise ValueError("Only one of P (period) or a (semi-major axis) can be defined !")
        if not _isnan(P) and P < 0:
            raise ValueError("Period must be positive !")
        if not _isnan(a) and a < 0:
            raise ValueError("Orbital semi-major axis must be positive !")
        if not _isnan(e) and not (0 <= e < 1):
            raise ValueError(f"Eccentricity must be in [0,1), e={e} !")
        if not (_isnan(ecos_omega) and _isnan(esin_omega)) and not _isnan(e):
            raise ValueError("Must specify either e and ecosω/esinω, but not both.")
        if not _isnan(tp) and not _isnan(t0):
            raise ValueError("Must specify either initial transit time or time of periastron passage, but not both.")

        omega = _replacenan(omega)
        
        if _nanall(e, ecos_omega, esin_omega):
            e, ecos_omega, esin_omega = 0.0, 0.0, 0.0

        elif _isnan(e):
            ecos_omega = _replacenan(ecos_omega)
            esin_omega = _replacenan(esin_omega)
            e = math.sqrt(ecos_omega ** 2 + esin_omega ** 2)
            assert 0 <= e < 1, "Eccentricity must be in [0,1)"
            omega = math.atan2(esin_omega, ecos_omega)
        
        else:
            ecos_omega = e * math.cos(omega)
            esin_omega = e * math.sin(omega)
            eps =  1e-15
            ecos_omega = 0.0 if abs(ecos_omega) < eps else ecos_omega
            esin_omega = 0.0 if abs(esin_omega) < eps else esin_omega
        
        t0 = _replacenan(t0)
        n = 2 * math.pi / P
        if _isnan(tp) and e != 0:
            tp = (t0 - math.sqrt(1.0-e*e)*ecos_omega/(n*(1.0-esin_omega)) 
                  - (2.0/n)*math.atan2(math.sqrt(1.0-e)*(esin_omega+ecos_omega+e), math.sqrt(1.0+e)*(esin_omega-ecos_omega-e))) % P
        elif _isnan(tp):
            theta = math.pi / 2 + omega
            tp = theta / n 
        
        a = _replacenan(a)
        I = _replacenan(I)
        Omega = _replacenan(Omega)
        
        return cls(m = float(m), P = float(P), t0 = float(t0), ecos_omega = float(ecos_omega), esin_omega = float(esin_omega), 
                   I = float(I), Omega = float(Omega), a = float(a), e = float(e), omega = float(omega), tp = float(tp))
  

# ElementIC : initial conditions specified by a hierarchy vector and oribital elements
class ElementsIC(InitialConditions):
    FIELD_ORDER = ["m", "P", "t0", "ecos_omega", "esin_omega", "I", "Omega"]

    def __init__(self, t0 : float, H: jnp.ndarray, elements : jnp.ndarray, der: bool = True):
        '''
        t0: initial time
        H: hierarchy matrix
        elements: (nbody, 7) orbital parameters, with field order as show 
        der: switch to grad (Jacobian)
        '''
        H_jnp = jnp.asarray(H, dtype = jnp.float64)
        elements_jnp = jnp.asarray(elements, dtype = jnp.float64)

        if H_jnp.ndim != 2:
            raise ValueError("H must be a two-dimensional matrix")

        nbody = H_jnp.shape[0]
        
        if H_jnp.shape != (nbody, nbody):
            raise ValueError("H must be a square matrix")

        if elements_jnp.shape != (nbody, 7):
            raise ValueError("elements must have shape " + f"({nbody}, 7), got " + f"{elements_jnp.shape}")
        
        nbody = H.shape[0]
        masses = elements[:,0]
        A = amatrix(H, masses)  #already return A jnp matrix
        
        self.elements = elements_jnp
        self.epsilon = H_jnp
        self.amat = A
        self.m = masses
        self.t0 = jnp.asarray(t0, dtype = elements_jnp.dtype)
        
        # static:
        self.nbody = int(nbody)
        self.der = bool(der)

    @classmethod
    def from_elements(cls, t0:float, H: jnp.ndarray, *elems: Elements, der: bool = True) -> "ElementsIC":
        nbody = H.shape[0]
        if len(elems) != nbody:
            raise ValueError(f"Expected {nbody} Elements objects, "+f"got {len(elems)}")

        elements = np.asarray([[getattr(elem, field_name)for field_name in cls.FIELD_ORDER] for elem in elems], dtype = np.float64)
        return cls(t0 = t0, H=H, elements=elements, der = der)
    
    @classmethod
    def from_elements_list(cls, t0: float, H: jnp.ndarray, elems: list[Elements],der: bool = True) -> "ElementsIC":
        return cls.from_elements(t0, H, *elems, der=der)

    @classmethod
    def from_file(cls, t0: float, H: jnp.ndarray, filepath: str, der: bool = True) -> "ElementsIC":
        elements = np.loadtxt(filepath, delimiter=",", comments="#")
        elements = elements[:H.shape[0]].astype(np.float64)
        return cls(t0 = t0, H = H, elements = elements, der=der)

    def __repr__(self) -> str:
        return f"ElementsIC[float64]\n Orbital Elements: \n{self.elements}"

@dataclass
class CartesianIC(InitialConditions):
    x:     jax.Array   # Position matrix (3, nbody)
    v:     jax.Array   # Velocity matrix (3, nbody)
    m:     jax.Array   # Mass (nbody,)
    nbody: int
    t0:    float

    @classmethod
    def from_matrix(cls, t0: float, N:int, coords: jnp.ndarray) -> "CartesianIC":
        coords = jnp.asarray(coords,dtype=jnp.float64,)
        if coords.ndim != 2:
            raise ValueError("coords must be a two-dimensional array")

        if coords.shape[1] != 7:
            raise ValueError("coords must have 7 columns: [m, x, y, z, vx, vy, vz]")

        if N <= 0:
            raise ValueError("N must be positive")

        if coords.shape[0] < N:
            raise ValueError(
                f"coords contains {coords.shape[0]} rows, "
                f"but N={N}")
        m = coords[:N, 0]
        x = coords[:N, 1:4].T
        v = coords[:N, 4:7].T
        return cls(x = x ,v = v, m = m, nbody = int(N), t0 = float(t0))
    
    @classmethod
    def from_file(cls, t0: float, N:int, filepath: str,) -> "CartesianIC":
        coords = np.loadtxt(filepath, delimiter=",")
        return cls.from_matrix(t0, N, coords)
    

#Functions used here, originally from init_nbody
def sigma_m(mass, i, j, epsilon):
    '''
    Sums masses in current Keplerian system
    epsilon - from Jacobi coordinates, copy from hierarchy matrix
    e.g.
    [[-1.  1.  0.]
    [-1. -1.  1.]
    [-1. -1. -1.]]

    '''
    mask = epsilon[i, :] == epsilon[i,j]
    return jnp.sum(mass[mask])

def amatrix(epsilon, masses):
    '''
    
    Create A matrix presented in Hamers & Portegies Zwart 2016
    Questions: seems need to use -epsilon[i,j] * masses[j] / denom
    '''
    N = len(masses)
    A = jnp.zeros_like(epsilon)

    for i in range(N):
        for j in range(N):
            denom = sigma_m(masses, i, j, epsilon)
            if denom != 0:
                A = A.at[i, j].set(epsilon[i, j] * masses[j] / denom)
    A_jax = jnp.asarray(A)
    return A_jax
