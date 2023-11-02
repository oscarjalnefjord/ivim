""" Functions to generate MR signal and corresponding Jacobians based on IVIM parameters. """

import numpy as np
import numpy.typing as npt
from ivim.constants import Db, y
from ivim.seq.sde import MONOPOLAR, BIPOLAR, G_from_b

NO_REGIME = 'no'
DIFFUSIVE_REGIME = 'diffusive'
BALLISTIC_REGIME = 'ballistic'
INTERMEDIATE_REGIME = 'intermediate'

def monoexp(b: npt.NDArray[np.float64], D: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Return the monoexponential e^(-b*D).
    
    Arguments:
        b: vector of b-values [s/mm2]
        D: ND array of diffusion coefficients [mm2/s]

    Output:
        S: (N+1)D array of signal values
    """

    [b, D] = at_least_1d([b, D])
    S = np.exp(-np.outer(D, b))
    return np.reshape(S, list(D.shape) + [b.size]) # reshape as np.outer flattens D if ndim > 1

def kurtosis(b: npt.NDArray[np.float64], D: npt.NDArray[np.float64], K: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Return the kurtosis signal representation.
    
    Arguments: 
        b: vector of b-values [s/mm2]
        D: ND array of diffusion coefficients [mm2/s]
        K: ND array of kurtosis coefficients (same shape as D or scalar)

    Output:
        S: (N+1)D array of signal values
    """
    
    [b, D, K] = at_least_1d([b, D, K])
    Slin = monoexp(b, D)
    Squad = np.exp(np.reshape(np.outer(D, b)**2, list(D.shape) + [b.size]) * K[..., np.newaxis]/6)
    return Slin * Squad

def sIVIM(b: npt.NDArray[np.float64], D: npt.NDArray[np.float64], f: npt.NDArray[np.float64], S0: npt.NDArray[np.float64] = 1, K: npt.NDArray[np.float64] = 0) -> npt.NDArray[np.float64]:
    """
    Return MR signal based on the simplified IVIM (sIVIM) model.
    
    Arguments: 
        b:  vector of b-values [s/mm2]
        D:  ND array of diffusion coefficients [mm2/s]
        f:  ND array of perfusion fractions (same shape as D or scalar)
        S0: (optional) ND array of signal values at b == 0 (same shape as D or scalar)
        K:  (optional) ND array of kurtosis coefficients (same shape as D or scalar)

    Output:
        S:  (N+1)D array of signal values
    """
    
    [b, D, f, S0] = at_least_1d([b, D, f, S0])
    return S0[..., np.newaxis] * ((1-f[..., np.newaxis]) * kurtosis(b, D, K) + np.reshape(np.outer(f, b==0), list(f.shape) + [b.size]))

def ballistic(b: npt.NDArray[np.float64], c: npt.NDArray[np.float64], D: npt.NDArray[np.float64], f: npt.NDArray[np.float64], vd: npt.NDArray[np.float64], S0: npt.NDArray[np.float64] = 1, K: npt.NDArray[np.float64] = 0) -> npt.NDArray[np.float64]:
    """
    Return MR signal based on the ballistic IVIM model.
    
    Arguments: 
        b:  vector of b-values [s/mm2]
        c:  vector of c-values [s/mm]
        D:  ND array of diffusion coefficients [mm2/s]
        f:  ND array of perfusion fractions (same shape as D or scalar)
        vd: ND array of velocity disperions [mm/s] (same shape as D or scalar)
        S0: (optional) ND array of signal values at b == 0 (same shape as D or scalar)
        K:  (optional) ND array of kurtosis coefficients (same shape as D or scalar)

    Output:
        S:  (N+1)D array of signal values
    """

    [b, c, D, f, vd, S0] = at_least_1d([b, c, D, f, vd, S0])
    return S0[..., np.newaxis] * ((1-f[..., np.newaxis])*kurtosis(b, D, K) + f[..., np.newaxis]*monoexp(b, Db)*monoexp(c**2, vd**2))

def diffusive(b: npt.NDArray[np.float64], D: npt.NDArray[np.float64], f: npt.NDArray[np.float64], Dstar: npt.NDArray[np.float64], S0: npt.NDArray[np.float64] = 1, K: npt.NDArray[np.float64] = 0) -> npt.NDArray[np.float64]:
    """
    Return MR signal based on the diffusive IVIM model.
    
    Arguments: 
        b:     vector of b-values [s/mm2]
        D:     ND array of diffusion coefficients [mm2/s]
        f:     ND array of perfusion fractions (same shape as D or scalar)
        Dstar: ND array of pseudo-diffusion coefficients [mm2/s] (same shape as D or scalar)
        S0:    (optional) ND array of signal values at b == 0 (same shape as D or scalar)
        K:     (optional) ND array of kurtosis coefficients (same shape as D or scalar)

    Output:
        S:     (N+1)D array of signal values
    """

    [b, D, f, Dstar, S0] = at_least_1d([b, D, f, Dstar, S0])
    return S0[..., np.newaxis] * ((1-f[..., np.newaxis])*kurtosis(b, D, K) + f[..., np.newaxis]*monoexp(b, Dstar))

def intermediate(b: npt.NDArray[np.float64], delta: npt.NDArray[np.float64], Delta: npt.NDArray[np.float64], D: npt.NDArray[np.float64], f: npt.NDArray[np.float64], v: npt.NDArray[np.float64], tau: npt.NDArray[np.float64], S0: npt.NDArray[np.float64] = 1, K: npt.NDArray[np.float64] = 0, seq = MONOPOLAR, T: npt.NDArray[np.float64] | None = None, k: npt.NDArray[np.float64] | None = None) -> npt.NDArray[np.float64]:
    """
    Return MR signal based on the intermediate IVIM model.
    
    Arguments: 
        b:     vector of b-values [s/mm2]
        delta: vector of gradient durations [s]
        Delta: vector of gradient separations [s]
        D:     ND array of diffusion coefficients [mm2/s]
        f:     ND array of perfusion fractions (same shape as D or scalar)
        Dstar: ND array of pseudo-diffusion coefficients [mm2/s] (same shape as D or scalar)
        S0:    (optional) ND array of signal values at b == 0 (same shape as D or scalar)
        K:     (optional) ND array of kurtosis coefficients (same shape as D or scalar)
        seq:   (optional) pulse sequence used (monopolar or bipolar)
        T:     (optional) vector of encoding times [s]
        k:     (optional) vector indicating if bipolar pulse sequence is flow compensated or not [-1/1] 

    Output:
        S:     (N+1)D array of signal values
    """

    [b, delta, Delta, T, k, D, f, v, tau, S0] = at_least_1d([b, delta, Delta, T, k, D, f, v, tau, S0])

    G = G_from_b(b, Delta, delta, seq)

    Deltam = np.reshape(np.outer(np.ones_like(tau), Delta), list(tau.shape) + [Delta.size])
    deltam = np.reshape(np.outer(np.ones_like(tau), delta), list(tau.shape) + [Delta.size])
    Gm     = np.reshape(np.outer(np.ones_like(tau), G), list(tau.shape) + [Delta.size])
    if seq == BIPOLAR:
        Tm     = np.reshape(np.outer(np.ones_like(tau), T), list(tau.shape) + [Delta.size])
        km     = np.reshape(np.outer(np.ones_like(tau), k), list(tau.shape) + [Delta.size])
    taum   = np.reshape(np.outer(tau, np.ones_like(Delta)), list(tau.shape) + [Delta.size])

    t1 = taum * deltam**2 * (Deltam - deltam/3)
    t3 = -2*taum**3 * deltam
    t4 = -taum**4 * (2*np.exp(-Deltam/taum) + 2*np.exp(-deltam/taum) - np.exp(-(Deltam+deltam)/taum) - np.exp(-(Deltam-deltam)/taum) - 2)
    if seq == BIPOLAR:
        t1 *= 2
        t3 *= 2
        t4 *= 2
        t4 += taum**4 * km * np.exp(-Tm/taum)*(np.exp((2*Deltam+2*deltam)/taum) - 2*np.exp((2*Deltam+deltam)/taum) + np.exp(2*Deltam/taum) - 2*np.exp((Deltam+2*deltam)/taum) + 4*np.exp((Deltam+deltam)/taum) - 2*np.exp(Deltam/taum) + np.exp(2*deltam/taum) - 2*np.exp(deltam/taum) + 1)

    Fp = np.exp(-y**2*(v**2/3)[..., np.newaxis]*Gm**2*(t1+t3+t4))
    return S0[..., np.newaxis] * ((1-f[..., np.newaxis])*kurtosis(b, D, K) + f[..., np.newaxis]*monoexp(b, Db)*Fp)


def monoexp_jacobian(b: npt.NDArray[np.float64], D: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """ 
    Return the Jacobian matrix for the monoexponential expression.
    
    S(b) = exp(-b*D)

    Arguments:
        b: vector of b-values [s/mm2]
        D: ND array of diffusion coefficients [mm2/s]

    Output: 
        J: Jacobian matrix
    """
    # warning! alternative to b[np.newaxis,:] may be needed
    J = (monoexp(b, D) * -b[np.newaxis, :])[...,np.newaxis] # D is the only parameter, but we still want the last dimension
    return J

def kurtosis_jacobian(b: npt.NDArray[np.float64], D: npt.NDArray[np.float64], K: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """ 
    Return the Jacobian matrix for the monoexponential expression.
    
    S(b) = exp(-b*D + b**2*D**2*K/6)

    Arguments:
        b: vector of b-values [s/mm2]
        D: ND array of diffusion coefficients [mm2/s]
        K: ND array of kurtosis coefficients (same shape as D or scalar)

    Output: 
        J: Jacobian matrix
    """

    [b,D,K] = at_least_1d([b,D,K])

    J = np.stack([
                  kurtosis(b,D,K)*(-b[np.newaxis, :]+2*np.reshape(np.outer(D*K,b**2)/6,list(D.shape) + [b.size])),
                  kurtosis(b,D,K)*np.reshape(np.outer(D, b)**2/6, list(D.shape) + [b.size])
                  ], axis=-1)
    return J
    
def sIVIM_jacobian(b: npt.NDArray[np.float64], D: npt.NDArray[np.float64], f: npt.NDArray[np.float64], S0: npt.NDArray[np.float64] | None = None, K: npt.NDArray[np.float64] | None = None) -> npt.NDArray[np.float64]:
    """
    Return the Jacobian matrix for the simplified IVIM (sIVIM) model.
    
    S(b) = S0((1-f)*exp(-b*D+b^2*D^2*K/6)+fδ(b))

    Arguments: 
        b:  vector of b-values [s/mm2]
        D:  ND array of diffusion coefficients [mm2/s]
        f:  ND array of perfusion fractions (same shape as D or scalar)
        S0: (optional) ND array of signal values at b == 0 (same shape as D or scalar)
        K:  (optional) ND array of kurtosis coefficients (same shape as D or scalar)

    Output:
        J:  Jacobian matrix
    """

    [b, D, f] = at_least_1d([b, D, f])

    if K is None:
        dSdD = (1-f)[..., np.newaxis] * monoexp_jacobian(b,D)[..., 0]
        dSdf = -monoexp(b,D) + (b==0)[np.newaxis, :]
    else:
        [K] = at_least_1d([K])
        dSdD = (1-f)[..., np.newaxis] * kurtosis_jacobian(b,D,K)[..., 0]
        dSdf = -kurtosis(b, D, K) + (b==0)[np.newaxis, :] 
        dSdK = (1-f)[..., np.newaxis] * kurtosis_jacobian(b,D,K)[..., 1]

    if S0 is None:
        if K is None:
            J_list = [dSdD, dSdf]
        else:
            J_list = [dSdD, dSdf, dSdK]
    else:
        [S0] = at_least_1d([S0])
        if K is None:
            dSdS0 = sIVIM(b, D, f)
        else:
            dSdS0 = sIVIM(b, D, f, K=K)
        dSdD *= S0[..., np.newaxis]
        dSdf *= S0[..., np.newaxis]
        if K is None:
            J_list = [dSdD, dSdf, dSdS0]
        else:
            J_list = [dSdD, dSdf, dSdS0, dSdK * S0[..., np.newaxis]]

    J = np.stack(J_list, axis=-1)
    
    return J

def ballistic_jacobian(b:  npt.NDArray[np.float64], c: npt.NDArray[np.float64], D: npt.NDArray[np.float64], f: npt.NDArray[np.float64], vd: npt.NDArray[np.float64], S0: npt.NDArray[np.float64] | None = None, K: npt.NDArray[np.float64] | None = None) -> npt.NDArray[np.float64]:
    """
    Return the Jacobian matrix for the ballistic IVIM model.
    
    S(b) = S0((1-f)*exp(-b*D+b^2*D^2*K/6)+f*exp(-b*Db-vd^2*c*2))

    Arguments: 
        b:  vector of b-values [s/mm2]
        c:  vector of c-values [s/mm]
        D:  ND array of diffusion coefficients [mm2/s]
        f:  ND array of perfusion fractions (same shape as D or scalar)
        vd: ND array of velocity dispersions [mm/s] (same shape as D or scalar)
        S0: (optional) ND array of signal values at b == 0 (same shape as D or scalar)
        K:  (optional) ND array of kurtosis coefficients (same shape as D or scalar)

    Output:
        J:  Jacobian matrix
    """

    [b, D, f, vd] = at_least_1d([b, D, f, vd])
    if S0 is not None:
        [S0] = at_least_1d([S0])
    exp2 = monoexp(b,np.atleast_1d(Db)) * monoexp(c**2,vd**2)

    J_sIVIM = sIVIM_jacobian(b,D,f,S0,K)
    dSdD  = J_sIVIM[..., 0]
    dSdvd = f[..., np.newaxis] * exp2 * (-2*vd[..., np.newaxis]@((c**2)[np.newaxis, :]))
    if S0 is None:
        dSdf  = J_sIVIM[..., 1] - np.ones_like(f)[..., np.newaxis]@(b==0)[np.newaxis, :] + exp2    
    else:
        dSdf  = J_sIVIM[..., 1] - S0[..., np.newaxis]@(b==0)[np.newaxis, :] + S0[..., np.newaxis]*exp2
        dSdvd *= S0[..., np.newaxis]

    if S0 is None:
        if K is None:
            J_list = [dSdD, dSdf, dSdvd]
        else:
            J_list = [dSdD, dSdf, dSdvd, J_sIVIM[..., 2]]
    else:
        if K is None:
            dSdS0 = ballistic(b,c,D,f,vd)
            J_list = [dSdD, dSdf, dSdvd, dSdS0]
        else:
            dSdS0 = ballistic(b,c,D,f,vd,K=K)
            J_list = [dSdD, dSdf, dSdvd, dSdS0, J_sIVIM[..., 3]]

    J = np.stack(J_list, axis=-1)

    return J

def diffusive_jacobian(b: npt.NDArray[np.float64], D: npt.NDArray[np.float64], f: npt.NDArray[np.float64], Dstar: npt.NDArray[np.float64], S0: npt.NDArray[np.float64] | None = None, K: npt.NDArray[np.float64] | None = None) -> npt.NDArray[np.float64]:
    """
    Return the Jacobian matrix for the diffusive IVIM model.
    
    S(b) = S0((1-f)*exp(-b*D+b^2*D^2*K/6)+f*exp(-b*D*))

    Arguments: 
        b:     vector of b-values [s/mm2]
        D:     ND array of diffusion coefficients [mm2/s]
        f:     ND array of perfusion fractions (same shape as D or scalar)
        Dstar: ND array of perfusion fractions (same shape as D or scalar)
        S0:    (optional) ND array of signal values at b == 0 (same shape as D or scalar)
        K:     (optional) ND array of kurtosis coefficients (same shape as D or scalar)

    Output:
        J:     Jacobian matrix
    """

    [b, D, f, Dstar] = at_least_1d([b, D, f, Dstar])
    if S0 is not None:
        [S0] = at_least_1d([S0])

    J_sIVIM = sIVIM_jacobian(b,D,f,S0,K)
    dSdD  = J_sIVIM[..., 0]
    dSdDstar = f[..., np.newaxis] * monoexp(b,Dstar) * -(np.ones_like(f)[..., np.newaxis]@b[np.newaxis, :])
    if S0 is None:
        dSdf  = J_sIVIM[..., 1] - np.ones_like(f)[..., np.newaxis]@(b==0)[np.newaxis, :] + monoexp(b,Dstar)
    else:
        dSdf  = J_sIVIM[..., 1] - S0[..., np.newaxis]@(b==0)[np.newaxis, :] + S0[..., np.newaxis]*monoexp(b,Dstar)
        dSdDstar *= S0[..., np.newaxis]

    if S0 is None:
        if K is None:
            J_list = [dSdD, dSdf, dSdDstar]
        else:
            J_list = [dSdD, dSdf, dSdDstar, J_sIVIM[..., 2]]
    else:
        [S0] = at_least_1d([S0])
        if K is None:
            dSdS0 = diffusive(b,D,f,Dstar)
            J_list = [dSdD, dSdf, dSdDstar, dSdS0]
        else:
            dSdS0 = diffusive(b,D,f,Dstar,K=K)
            J_list = [dSdD, dSdf, dSdDstar, dSdS0, J_sIVIM[..., 3]]

    J = np.stack(J_list, axis=-1)

    return J

def at_least_1d(pars: list) -> list:
    """ Check that each parameter is atleast one dimension in shape. """
    for i, par in enumerate(pars):
        pars[i] = np.atleast_1d(par)
    return pars

def check_regime(regime: str) -> None:
    """ Check that the regime is valid. """
    if regime not in [NO_REGIME, DIFFUSIVE_REGIME, BALLISTIC_REGIME, INTERMEDIATE_REGIME]:
        raise ValueError(f'Invalid regime "{regime}". Valid regimes are "{NO_REGIME}", "{DIFFUSIVE_REGIME}", "{BALLISTIC_REGIME}" and "{INTERMEDIATE_REGIME}".')