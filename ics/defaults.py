import warnings
import jax
import jax.numpy as jnp
from .initial_conditions import ElementsIC
from .setup_hierarchy import hierarchy

def get_trappist1_elements(t0 = 0.0, n = 0):
    elements = jnp.array([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [2.5901135977661885e-5, 1.510880055106516,  7257.547487248826,   0.02436651768325364,    0.018169884000968452,  1.5707963267948966, 0.0],
        [5.7871255112412840e-5, 2.4218013609356652, 7258.592163817471,   0.020060810686211832,   0.011189705094395375,  1.5707963267948966, 0.0],
        [1.4602772830539989e-6, 4.0503542353950355, 7257.023855669221,   0.007411490159357976,  -0.02016424872931776 ,  1.5707963267948966, 0.0],
        [1.9235328222249013e-5, 6.099281590191818,  7257.816770447013,   0.0011801938769616127,  0.000731913417670215 , 1.5707963267948966, 0.0],
        [2.7302687390082730e-5, 9.20618480814173,   7257.1228936246725, -699952921060827e-19,    0.0002252519365921506, 1.5707963267948966, 0.0],
        [3.5331017018761430e-5, 12.353988709624156, 7257.667328639113,  -0.0009722026578612578,  0.001276000403979281,  1.5707963267948966, 0.0],
        [1.6410627049780406e-6, 18.733535095576702, 7250.524231929195,  -0.010402303111464135 , -0.014289870200773339,  1.5707963267948966, 0.0],      
    ])
    return elements

def get_kepler36_elements(t0 = 0.0, n = 0):
    elements = jnp.array([[1.0,                    0.0,      0.0,                 0.0,   0.0,   1.5707963267948966, 0.0],
        [1.2479484222582662e-5, 13.83989,  0.0,                 0.002, -0.004, 1.5707963267948966, 0.0],
        [2.2659378094037732e-5, 16.23855,  5.062100000213832,  -0.01,   0.007, 1.5707963267948966, 0.0]])
    return elements

###
AVAILABLE_SYSTEMS: tuple[tuple[str, ...], ...] = (
    ("trappist-1", "trappist 1"),
    ("kepler-36", "kepler 36")
)

ELEMENT_FUNCTIONS: dict[str, callable] = {
    "trappist-1": get_trappist1_elements,
    "kepler-36": get_kepler36_elements
}
### Other useful functions:

def _available_systems() -> tuple:
    return AVAILABLE_SYSTEMS

def available_systems() -> None:
    print('Currently available systems include:')
    for name in AVAILABLE_SYSTEMS:
        key = name[0]
        rest = name[1:]
        print(key)

def get_default_ICs(system_name, t0 = 0.0, n = 0):
    system_key = ""
    for names in AVAILABLE_SYSTEMS:
        if system_name.lower() in names:
            system_key = names[0]
            break
    if system_key == "":
        raise ValueError(f'The input system is not currently available')
    
    elements = ELEMENT_FUNCTIONS[system_key]()
    n_max = elements.shape[0]

    if n == 0:
        n = n_max
    elif n > n_max:
        warnings.warn('MAX n is {n_max}, User asked for {n}. Setting to ${n_max}')
        n = n_max
    # build hierarchy matrix
    H = hierarchy([n] + [1] * (n-1)) #why 
    return ElementsIC(t0, H, elements)