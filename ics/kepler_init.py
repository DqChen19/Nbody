import jax
import jax.numpy as jnp
from .kepler import kepler
from functools import partial

YEAR    = 365.242
GNEWT   = 39.4845 / (YEAR * YEAR)
        
def kepler_cartesian(time, params):
    """
    Convert Keplerian parameters to Cartesian state.

    Parameters
    ----------
    time : scalar
        Time at which the state is evaluated.
    params : array, shape (7,) [period, t0, ecos_omega, esin_omega, inclination, Omega, mass]

    Returns
    -------
    state : array, shape (7,) [x, y, z, vx, vy, vz, mass]
    """
    period, t0, ecos_omega, esin_omega, I, Omega, mass = params

    semi = jnp.cbrt(GNEWT * mass * period **2 / (4 * jnp.pi**2))
    ecc = jnp.hypot(ecos_omega, esin_omega)
    omega = jnp.arctan2(esin_omega, ecos_omega)

    f1 = 1.5 * jnp.pi - omega
    sqrt1mecc2 = jnp.sqrt(1.0 - ecc ** 2) # sqrt(1 - e^2)

    tp = (t0 + period * sqrt1mecc2 / (2 * jnp.pi) * (ecc * jnp.sin(f1) / (1.0 + ecc * jnp.cos(f1)) 
             - 2.0 / sqrt1mecc2 * jnp.arctan2(sqrt1mecc2 * jnp.tan(0.5 *f1), 1.0 + ecc)))

    mean_motion = 2 * jnp.pi / period #mean_motion, n
    mean_anomaly = mean_motion * (time - tp) #mean_anomly, m

    true_anomaly = kepler(mean_anomaly, ecc)

    ecosfp1 = 1.0 + ecc * jnp.cos(true_anomaly) # one plus e cos(true anomaly)
    fdot = mean_motion * ecosfp1 ** 2 / sqrt1mecc2 ** 3
    radius = semi * (1.0 - ecc**2)/ecosfp1 #r
    radial_velocity = semi * mean_motion / sqrt1mecc2 * ecc * jnp.sin(true_anomaly) #rdot

    cos_Omega = jnp.cos(Omega)
    sin_Omega = jnp.sin(Omega)

    cos_I = jnp.cos(I)
    sin_I = jnp.sin(I)

    argument = omega + true_anomaly
    cos_omega_pf = jnp.cos(argument)
    sin_omega_pf = jnp.sin(argument)

    x = radius * jnp.array([
        (cos_Omega * cos_omega_pf - sin_Omega * sin_omega_pf * cos_I),
        (sin_Omega * cos_omega_pf + cos_Omega * sin_omega_pf * cos_I),
        sin_omega_pf * sin_I]) #Position

    radial_ratio = radial_velocity / radius 
    tangential_velocity = radius * fdot

    v = jnp.array([
        x[0] * radial_ratio  + tangential_velocity * (-cos_Omega * sin_omega_pf - sin_Omega * cos_omega_pf * cos_I),
        x[1] * radial_ratio  + tangential_velocity * (-sin_Omega * sin_omega_pf + cos_Omega * cos_omega_pf * cos_I),
        x[2] * radial_ratio  + tangential_velocity * cos_omega_pf * sin_I,]) # Velocity


    return jnp.concatenate([x,v,jnp.reshape(mass, (1,)),]) #[x, v, m]

@partial(jax.jit, static_argnames=("der",))
def kepler_init(time, mass, elements, der):
    """
    Parameters
    ----------
    elements : array, shape (6,)
        [period, t0, ecos_omega, esin_omega, inclination, Omega]

    Returns:
    x, v
    jac_init is the derivative of (x,v,m) with respect to (elements,m).
        shape (7, 7)
            Derivative of (x, v, mass) with respect to
            (period, t0, e*cos(omega), e*sin(omega),
             inclination, Omega, mass).
    """
    elements = jnp.asarray(elements)
    mass = jnp.asarray(mass, dtype=elements.dtype)
    params = jnp.concatenate([elements,jnp.reshape(mass, (1,)),])
    state = kepler_cartesian(time, params)# [x, v]
    x = state[:3]
    v = state[3:6]
    
    if der:
        return x,v, jax.jacfwd(kepler_cartesian, argnums=1,)(time, params) 

    return x, v, jnp.nan