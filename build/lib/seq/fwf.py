""" Various functions related to calculations with arbirary gradient waveforms. """

import numpy as np
import numpy.typing as npt
from ivim.constants import y

def q_from_g(g: npt.NDArray[np.float64], dt: float) -> npt.NDArray[np.float64]:
    """
    Calculate the dephasing vector for gradient waveform g.

    Arguments:
        g:  n x 3 array where each column describes a gradient channel and n is the number of gradient waveform samples
        dt: time between gradient samples

    Output:
        q:  dephasing vector

    Note! Gyromagnetic ration (gamma) is given in units [rad/T/s]. Use reasonable units for input arguments to get preferred output unit. E.g.:
    q [1/mm] -- g [T/mm] and dt [s] 
    q [1/m]  -- g [T/m] and dt [s] 
    """

    g_shift = np.zeros_like(g)
    g_shift[1:, :] = g[:-1, :]
    q = y * np.cumsum((g + g_shift)/2, axis=0) * dt
    return q

def B_from_q(q: npt.NDArray[np.float64], dt: float) -> npt.NDArray[np.float64]:
    """
    Calculate B-tensor from dephasing vector.

    Arguments:
        q:  n x 3 array where each column describes a gradient channel and n is the number of gradient waveform samples
        dt: time between gradient samples

    Output:
        B:  B tensor

    Note! Units of input arguments will decide the output unit. E.g.:
    B [s/mm2]  -- q [1/mm] and dt [s]
    B [ms/um2] -- q [1/um] and dt [ms]
    B [s/m2]   -- q [1/m] and dt [s]
    """
    
    B = q.T @ q * dt
    return B

def b_from_q(q: npt.NDArray[np.float64], dt: float) -> float:
    """
    Calculate b-value from dephasing vector.

    Arguments:
        q:  n x 3 array where each column describes a gradient channel and n is the number of gradient waveform samples
        dt: time between gradient samples

    Output:
        b:  b-value

    Note! Units of input arguments will decide the output unit. E.g.:
    b [s/mm2]  -- q [1/mm] and dt [s]
    b [ms/um2] -- q [1/um] and dt [ms]
    b [s/m2]   -- q [1/m] and dt [s]
    """
        
    b = np.trace(B_from_q(q, dt))
    return b

def C_from_q(q: npt.NDArray[np.float64], dt: float) -> npt.NDArray[np.float64]:
    """
    Calculate C-vector from dephasing vector.

    Arguments:
        q:  n x 3 array where each column describes a gradient channel and n is the number of gradient waveform samples
        dt: time between gradient samples

    Output:
        C:  C vector

    Note! Units of input arguments will decide the output unit. E.g.:
    C [s/mm]  -- q [1/mm] and dt [s]
    C [ms/um] -- q [1/um] and dt [ms]
    C [s/m]   -- q [1/m] and dt [s]
    """

    C = np.sum(q, axis=0) * dt
    return C

def c_from_q(q: npt.NDArray[np.float64], dt: float) -> float:
    """
    Calculate c-value from dephasing vector.

    Arguments:
        q:  n x 3 array where each column describes a gradient channel and n is the number of gradient waveform samples
        dt: time between gradient samples

    Output:
        c:  c-value

    Note! Units of input arguments will decide the output unit. E.g.:
    c [s/mm]  -- q [1/mm] and dt [s]
    c [ms/um] -- q [1/um] and dt [ms]
    c [s/m]   -- q [1/m] and dt [s]
    """
        
    c = np.sum(C_from_q(q, dt))
    return c

def B_from_g(g: npt.NDArray[np.float64], dt: float) -> npt.NDArray[np.float64]:
    """
    Calculate B-tensor from gradient waveforms.

    Arguments:
        g:  n x 3 array where each column describes a gradient channel and n is the number of gradient waveform samples
        dt: time between gradient samples

    Output:
        B:  B tensor

    Note! Units of input arguments will decide the output unit. E.g.:
    B [s/mm2]  -- g [T/mm] and dt [s]
    B [ms/um2] -- g [T/um] and dt [ms]
    B [s/m2]   -- g [T/m] and dt [s]
    """

    B = B_from_q(q_from_g(g, dt), dt)
    return B

def b_from_g(g: npt.NDArray[np.float64], dt: float) -> float:
    """
    Calculate b-value from gradient waveforms.

    Arguments:
        g:  n x 3 array where each column describes a gradient channel and n is the number of gradient waveform samples
        dt: time between gradient samples

    Output:
        b:  b-value

    Note! Units of input arguments will decide the output unit. E.g.:
    b [s/mm2]  -- g [T/mm] and dt [s]
    b [ms/um2] -- g [T/um] and dt [ms]
    b [s/m2]   -- g [T/m] and dt [s]
    """

    b = np.trace(B_from_g(g, dt))
    return b

def C_from_g(g: npt.NDArray[np.float64], dt: float) -> npt.NDArray[np.float64]:
    """
    Calculate C-vector from gradient waveforms.

    Arguments:
        g:  n x 3 array where each column describes a gradient channel and n is the number of gradient waveform samples
        dt: time between gradient samples

    Output:
        C:  C-vector

    Note! Units of input arguments will decide the output unit. E.g.:
    C [s/mm]  -- g [T/mm] and dt [s]
    C [ms/um] -- g [T/um] and dt [ms]
    C [s/m]   -- g [T/m] and dt [s]
    """
    
    C = C_from_q(q_from_g(g, dt), dt)
    return C

def c_from_g(g: npt.NDArray[np.float64], dt: float) -> float:
    """
    Calculate c-value from gradient waveforms.

    Arguments:
        g:  n x 3 array where each column describes a gradient channel and n is the number of gradient waveform samples
        dt: time between gradient samples

    Output:
        c:  c-value

    Note! Units of input arguments will decide the output unit. E.g.:
    c [s/mm]  -- g [T/mm] and dt [s]
    c [ms/um] -- g [T/um] and dt [ms]
    c [s/m]   -- g [T/m] and dt [s]
    """    
    
    c = np.sum(C_from_g(g, dt))
    return c

def G_from_b(b: npt.NDArray[np.float64], gnorm: npt.NDArray[np.float64], dt: float) -> npt.NDArray[np.float64]:
    """
    Calculate gradient strength needed to achive a given b-value.

    Arguments:
        b:     b-values
        gnorm: normalized gradient waveform
        dt:    time between gradient samples

    Output:
        G:     gradient strength 

    Note! Units of input arguments will decide the output unit. E.g.:
    G [T/m] -- b [s/m2] and dt [s]
    """
    
    bnorm = b_from_g(gnorm, dt)
    G = np.sqrt(b/bnorm)
    return G

def c_from_b(b: npt.NDArray[np.float64], gnorm: npt.NDArray[np.float64], dt: float) -> npt.NDArray[np.float64]:
    """
    Calculate c-value for a given b-value and an arbirary gradient waveform.

    Arguments:
        b:     b-values
        gnorm: normalized gradient waveform
        dt:    time between gradient samples
    
    Output:
        c:     c-values

    Note! Units of input arguments will decide the output unit. E.g.:
    c [s/mm] -- b [s/mm2] and dt [s]
    """
    
    b = np.atleast_1d(b)
    Gs = G_from_b(b, gnorm, dt)

    c = np.zeros_like(b)
    for i, G in enumerate(Gs):
        c[i] = c_from_g(G*gnorm, dt)

    return c