import numpy as np
import jax
import jax.numpy as jnp
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .State import State
    from ..ics.initial_conditions import InitialConditions

# Convert the integrator's current Cartesian state (s.x, s.v, s.m) 
# back into orbital parameters [m, P, t_0, ecos_omega, esin_omega, I, Omega]
YEAR    = 365.242
GNEWT   = 39.4845 / (YEAR * YEAR)
Array = jax.Array
# ================
# Supporting functions
# ================
def calc_a(R, V, Gm):
    # Semimajor axis
    return 1.0 / (2.0 / R - V ** 2 / Gm)

def calc_e(h, Gm, a):
    # Eccentricity
    return jnp.sqrt(jnp.maximum(1.0 - h **2 /(Gm * a), 0.0))

def calc_I(hz, h):
    # Inclination
    return jnp.arccos(hz/h)

def calc_Omega(hx, hy):
    # longitude of ascending node Omega
    return jnp.arctan2(hx, hy)

def calc_varpi(x, R, Rdot, Omega, I, e, h, a):
    # Longitude of periapsis
    X, _, Z = x
    sin_Omega, cos_Omega = jnp.sin(Omega), jnp.cos(Omega)
    sin_I, cos_I = jnp.sin(I), jnp.cos(I)
    
    #omega + f
    sin_wpf = Z / ( R * sin_I )
    cos_wpf = ((X / R) + sin_Omega * sin_wpf * cos_I) / cos_Omega 
    wpf = jnp.arctan2(sin_wpf, cos_wpf)

    sin_f = a * Rdot * (1.0 - e ** 2) / (h * e)
    cos_f = (a * (1.0 - e**2)/R - 1.0) / e
    f = jnp.arctan2(sin_f, cos_f)

    return Omega + (wpf - f - jnp.pi)

def calc_t0(R, a, e, Gm, t):
    # Time of periastron 
    E = jnp.arccos((1.0 - R/a)/e)
    return t - (E - e * jnp.sin(E))/jnp.sqrt( Gm / a**3 )

def calc_P(a, Gm):
    # Orbital period
    return 2.0 * jnp.pi * jnp.sqrt( a**3 / Gm)

# ====================
# Other functions
# ====================

def calc_X(s, ic):
    '''
    Convert barycentric Cartesian coordinates to the relative coordinates of each Kepler subsystem.
    '''
    n = s.x.shape[1]
    X = (ic.amat[:n-1, :] @ s.x.T).T #(3, n-1)
    V = (ic.amat[:n-1, :] @ s.v.T).T #(3, n-1)

    return X, V

def gmass(m, epsilon):
    '''
    Calculate g mass of each kepler system M[i] = Sum_j epsilon[i, j] * m[j]
    Returns:
        Gm: (n-1, )
    '''
    N = len(m)
    M = jnp.abs(epsilon[:N-1, :]) @ m #(N-1,)
    return GNEWT * M 

def get_orbital_elements(s, ic):
    '''
    Cartesian -> Matrix of orbital parameters
    Output:
    [mass, orbit period, t0, e cos_omega, e sin_omge, I, Omega]
    '''
    X, Xdot = calc_X(s, ic) #(3, n-1)
    Gm = gmass(s.m, ic.epsilon) #(n-1, )

    R = jnp.linalg.norm(X, axis = 0) # |r|, (n-1,)
    V = jnp.linalg.norm(Xdot, axis = 0) # |v|, (n-1,)

    h_vecs = jnp.cross(X.T, Xdot.T).T
    hx = h_vecs[0]
    hy = h_vecs[1]
    hz = h_vecs[2]
    h = jnp.linalg.norm(h_vecs, axis = 0) #shape (n-1,)

    # Radial velocity, r * v / |r|
    Rdot = jnp.sum(X * Xdot,axis=0,) / R

    # Orbital parameters
    a = calc_a(R, V, Gm)
    e = calc_e(h, Gm, a)
    I = calc_I(hz, h)
    Omega = calc_Omega(hx, hy)
    P = calc_P(a, Gm)

    calc_vmap_varpi = jax.vmap(calc_varpi, in_axes=(1, 0, 0, 0, 0, 0, 0, 0))
    varpi = calc_vmap_varpi(X, R, Rdot, Omega, I, e, h, a,) 

    ecos_omega = e * jnp.cos(varpi) ##??
    esin_omega = e * jnp.sin(varpi) ##??

    t0 = ic.elements[1:, 2] #except the host star

    body_elements = jnp.column_stack([s.m[1:], P, t0, ecos_omega, esin_omega, I, Omega])
    return jnp.concatenate((ic.elements[0:1], body_elements), axis = 0,)

get_orbital_elements_jit = jax.jit(get_orbital_elements)