from __future__ import annotations

from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax

YEAR    = 365.242
GNEWT   = 39.4845 / (YEAR * YEAR)
Array = jax.Array

"""
Convert Cartesian coordinates to orbital elements.

Array conventions
-----------------
x[:, i] : position of body i, shape (3,)
v[:, i] : velocity of body i, shape (3,)

Full position and velocity arrays have shape (3, n).
"""
def point(x: Any) -> Array:
    """
    Construct a 3D point/vector:
    Scalar input: point(1.0) -> [1.0, 1.0, 1.0]
    Vector input:point([x, y, z]) -> shape (3,)
    """
    array = jnp.asarray(x)

    if array.ndim == 0:
        return jnp.full((3,), array)

    if array.shape != (3,):
        raise ValueError(f"A point must have shape (3,), got {array.shape}.")

    return array


def unpack(x: Array):
    """Return the three coordinates of a point."""
    return x[0], x[1], x[2]


def dot_self(x: Array) -> Array:
    """Equivalent to the Julia overload dot(x::Point)."""
    return jnp.dot(x, x)

def get_relative_positions(x: Array, v: Array,ic) -> tuple[Array, Array]:
    """
    Calculate relative positions and velocities using the A matrix.

    Parameters
    ----------
    x, v: Cartesian coordinates, shape (3, n).
    ic.amat: A matrix, normally shape (n, n).

    Returns
    -------
    X, V: Relative positions and velocities, shape (3, n).
    """
    amat = jnp.asarray(ic.amat,dtype=x.dtype,)

    X = x @ amat.T
    V = v @ amat.T

    return X, V

def get_relative_masses(ic) -> Array:
    """
    Calculate relative gravitational parameters G*M.

    Equivalent to:

        M[i] = sum_j abs(epsilon[i, j]) * m[j]

    for i = 0, ..., N-2.
    """
    masses = jnp.asarray(ic.m)

    # Prefer the ASCII Python attribute name.
    if hasattr(ic, "epsilon"):
        epsilon = jnp.asarray(ic.epsilon, dtype=masses.dtype,)
    else:
        raise AttributeError("InitialConditions must contain `epsilon`.")

    relative_mass = jnp.sum(jnp.abs(epsilon[:-1, :]) * masses[None, :], axis=1)
    gravitational_constant = jnp.asarray(GNEWT, dtype=masses.dtype,)

    return gravitational_constant * relative_mass


def mag(x: Array) -> Array:
    """Magnitude of a three-dimensional vector."""
    return jnp.sqrt(jnp.dot(x, x))


def mag_dot(x: Array, v: Array,) -> Array:
    """
    Equivalent to the original Julia mag(x, v).
    Note that this computes sqrt(dot(x, v)), not a vector norm.
    """
    return jnp.sqrt(jnp.dot(x, v))


def rdot_magnitude(R: Array, V: Array, h: Array) -> Array:
    """
    Magnitude of the radial velocity.
        |Rdot| = sqrt(V^2 - (h/R)^2)
    """
    argument = V * V - (h / R) ** 2
    # Small negative values can arise from floating-point roundoff.
    return jnp.sqrt(jnp.maximum(argument, 0.0))

def hvec(r: Array,rdot: Array,) -> Array:
    """
    Calculate the modified angular-momentum vector used by the
    original NbodyGradient orbital-element convention.

    This intentionally preserves the original sign changes:

        hz >= 0 -> hy *= -1
        hz < 0  -> hx *= -1
    """
    h_raw = jnp.cross(r, rdot)
    hx = h_raw[0]
    hy = h_raw[1]
    hz = h_raw[2]
    hx = jnp.where(hz >= 0.0,hx,-hx,)
    hy = jnp.where(hz >= 0.0,-hy,hy,)
    return jnp.stack((hx, hy, hz))

def calc_longitude_ascending_node(hx: Array, hy: Array, h: Array, inclination: Array,) -> Array:
    """Calculate the longitude of ascending node, Ω."""
    sin_inclination = jnp.sin(inclination)

    sin_Omega = hx / (h * sin_inclination)
    cos_Omega = hy / (h * sin_inclination)

    return jnp.arctan2(sin_Omega, cos_Omega,)

def calc_argument_of_periapsis(x: Array, R: Array, Rdot: Array, inclination: Array,
    Omega: Array, semi_major_axis: Array, eccentricity: Array, angular_momentum: Array,) -> Array:
    """
    Calculate the argument of periapsis, omega.
    """
    dtype = x.dtype
    zero = jnp.asarray(0.0, dtype=dtype)

    def calculate_w_plus_f(_):
        sin_wpf = (x[2] / (R * jnp.sin(inclination)))
        cos_wpf = (x[0] / R + jnp.sin(Omega) * sin_wpf * jnp.cos(inclination)) / jnp.cos(Omega)

        return jnp.arctan2(sin_wpf, cos_wpf,)

    w_plus_f = lax.cond(inclination != zero, calculate_w_plus_f, lambda _: zero, operand=None,)
    sin_f = (semi_major_axis * Rdot * (1.0 - eccentricity**2) / (angular_momentum * eccentricity))
    cos_f = (semi_major_axis * (1.0 - eccentricity**2) / R - 1.0) / eccentricity

    true_anomaly = jnp.arctan2(sin_f, cos_f,)

    return w_plus_f - true_anomaly

def convert_to_elements(x: Array,v: Array,gravitational_parameter: Array,) -> Array:
    """
    Convert one relative Cartesian state to orbital elements.

    Parameters
    ----------
    x, v: Relative position and velocity, shape (3,).

    gravitational_parameter: G * M for this relative coordinate.

    Returns
    -------
    elements: Shape (10,), containing:
        [P, t0, e*cos(omega), e*sin(omega), I, Omega, a, e, omega, tp]
    """
    dtype = x.dtype
    zero = jnp.asarray(0.0, dtype=dtype)

    R = mag(x)
    V = mag(v)

    angular_momentum_vector = hvec(x, v)
    angular_momentum = mag(angular_momentum_vector)

    hx, hy, hz = unpack(angular_momentum_vector)

    radial_sign = jnp.sign(jnp.dot(x, v))

    Rdot = radial_sign * rdot_magnitude(R, V, angular_momentum,)

    mu = gravitational_parameter

    semi_major_axis = 1.0 / (2.0 / R - V * V / mu)

    eccentricity_squared = (1.0 - angular_momentum**2 / (mu * semi_major_axis))

    eccentricity = jnp.sqrt(jnp.maximum(eccentricity_squared, zero,))
    cosine_inclination = hz / angular_momentum

    # Avoid acos domain errors caused by roundoff.
    inclination = jnp.arccos(jnp.clip(cosine_inclination, -1.0, 1.0,))

    Omega = lax.cond(
        inclination != zero,
        lambda _: calc_longitude_ascending_node(hx,hy,angular_momentum, inclination,),
        lambda _: zero,
        operand=None,
    )

    omega = calc_argument_of_periapsis(
        x=x, R=R, Rdot=Rdot, inclination=inclination,
        Omega=Omega, semi_major_axis=semi_major_axis,
        eccentricity=eccentricity, angular_momentum=angular_momentum,)

    period = (2.0 * jnp.pi * jnp.sqrt(semi_major_axis**3 / mu))
    mean_motion = 2.0 * jnp.pi / period

    sin_omega = jnp.sin(omega)
    cos_omega = jnp.cos(omega)

    ecos_omega = eccentricity * cos_omega
    esin_omega = eccentricity * sin_omega

    sqrt_one_minus_e2 = jnp.sqrt(jnp.maximum(1.0 - eccentricity**2,zero,))

    first_term = (-sqrt_one_minus_e2 * ecos_omega/ ( mean_motion * (1.0 - esin_omega)))

    atan_numerator = (jnp.sqrt(1.0 - eccentricity)* (esin_omega + ecos_omega + eccentricity))
    atan_denominator = (jnp.sqrt(1.0 + eccentricity) * (esin_omega - ecos_omega - eccentricity))
    second_term = (-2.0 / mean_motion * jnp.arctan2( atan_numerator, atan_denominator,))
    time_of_periapsis = jnp.mod(first_term + second_term,period, )

    return jnp.stack((period, zero, ecos_omega, esin_omega, inclination,
            Omega, semi_major_axis, eccentricity, omega, time_of_periapsis,))

def convert_relative_states_to_elements(X: Array,V: Array,mu: Array,) -> Array:
    """
    Convert several relative states simultaneously.

    Parameters
    ----------
    X, V: Shape (3, n_relative).
    mu:   Shape (n_relative,).

    Returns
    -------
    elements: Shape (n_relative, 10).
    """
    return jax.vmap(convert_to_elements,in_axes=(1, 1, 0),out_axes=0,)(X, V, mu)

def get_orbital_elements(state, ic, Elements,):
    """
    Return orbital Elements objects for all bodies.

    This is an output-layer function and returns a Python list, so the
    function as a whole is not intended to be jitted.

    The numerical conversion inside convert_to_elements remains fully
    JAX-compatible.
    """
    mu = get_relative_masses(ic)
    X, V = get_relative_positions(state.x, state.v, ic,)
    masses = jnp.asarray(ic.m)

    if hasattr(ic, "epsilon"):
        epsilon = jnp.asarray(ic.epsilon)
    else:
        raise AttributeError("InitialConditions must contain `epsilon`.")

    nbody = int(ic.nbody)
    elements = [Elements(m=masses[0])]

    # Python equivalent is i = 0 for the first relative orbit/body after the central body.
    i = 0
    binary_offset = 0

    while i < nbody - 1:
        if float(epsilon[i, 0]) == 0.0:
            binary_offset += 1

        relative_index = i + binary_offset

        new_elements = convert_to_elements(X[:, relative_index],V[:, relative_index],mu[relative_index],)

        elements.append(
            Elements(m=masses[i + 1],
                P=new_elements[0],
                t0=new_elements[1],
                ecosomega=new_elements[2],
                esinomega=new_elements[3],
                I=new_elements[4],
                Omega=new_elements[5],
                a=new_elements[6],
                e=new_elements[7],
                omega=new_elements[8],
                tp=new_elements[9], ))
        if binary_offset > 0:
            binary_offset -= 2
        elif binary_offset < 0:
            i += 1

        i += 1
    return elements