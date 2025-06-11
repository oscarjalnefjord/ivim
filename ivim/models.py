""" 
This module provides functions to generate MR signal and corresponding Jacobians based on IVIM 
parameters. 
"""

import numpy as np
import numpy.typing as npt
from ivim.constants import Db, y
from ivim.seq.lte import MONOPOLAR, BIPOLAR, G_from_b

NO_REGIME = 'no'
DIFFUSIVE_REGIME = 'diffusive'
BALLISTIC_REGIME = 'ballistic'
INTERMEDIATE_REGIME = 'intermediate'

def monoexp(b: npt.NDArray[np.float64], D: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Return the monoexponential :math:`e^{-bD}`.
    
    Arguments:
        b: vector of b-values [s/mm\ :sup:`2`]
        D: ND array of diffusion coefficients [mm\ :sup:`2`/s]

    Output:
        S: (N+1)D array of signal values
    """

    [b, D] = at_least_1d([b, D])
    S = np.exp(-np.outer(D, b))
    return np.reshape(S, list(D.shape) + [b.size]) # reshape as np.outer flattens D if ndim > 1

def kurtosis(b: npt.NDArray[np.float64], D: npt.NDArray[np.float64], K: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Return the kurtosis signal representation :math:`e^{-bD+b^2D^2K/6}`.
    
    Arguments: 
        b: vector of b-values [s/mm\ :sup:`2`]
        D: ND array of diffusion coefficients [mm\ :sup:`2`/s]
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
        b:  vector of b-values [s/mm\ :sup:`2`]
        D:  ND array of diffusion coefficients [mm\ :sup:`2`/s]
        f:  ND array of perfusion fractions (same shape as D or scalar)
        S0: (optional) ND array of signal values at b == 0 (same shape as D or scalar)
        K:  (optional) ND array of kurtosis coefficients (same shape as D or scalar)

    Output:
        S:  (N+1)D array of signal values
    """
    
    [b, D, f, S0] = at_least_1d([b, D, f, S0])
    return S0[..., np.newaxis] * ((1-f[..., np.newaxis]) * kurtosis(b, D, K) + np.reshape(np.outer(f, b==0), list(f.shape) + [b.size]))

def ballistic(b: npt.NDArray[np.float64], c: npt.NDArray[np.float64], D: npt.NDArray[np.float64], f: npt.NDArray[np.float64], v: npt.NDArray[np.float64], S0: npt.NDArray[np.float64] = 1, K: npt.NDArray[np.float64] = 0) -> npt.NDArray[np.float64]:
    """
    Return MR signal based on the ballistic IVIM model with :math:`F_P = e^{-c^2v^2/6}`.
    
    Arguments: 
        b:  vector of b-values [s/mm\ :sup:`2`]
        c:  vector of c-values [s/mm]
        D:  ND array of diffusion coefficients [mm\ :sup:`2`/s]
        f:  ND array of perfusion fractions (same shape as D or scalar)
        v:  ND array of velocities [mm/s] (same shape as D or scalar)
        S0: (optional) ND array of signal values at b == 0 (same shape as D or scalar)
        K:  (optional) ND array of kurtosis coefficients (same shape as D or scalar)

    Output:
        S:  (N+1)D array of signal values
    """

    [b, c, D, f, v, S0] = at_least_1d([b, c, D, f, v, S0])
    return S0[..., np.newaxis] * ((1-f[..., np.newaxis])*kurtosis(b, D, K) + f[..., np.newaxis]*monoexp(b, Db)*monoexp(c**2, v**2/6))

def diffusive(b: npt.NDArray[np.float64], D: npt.NDArray[np.float64], f: npt.NDArray[np.float64], Dstar: npt.NDArray[np.float64], S0: npt.NDArray[np.float64] = 1, K: npt.NDArray[np.float64] = 0) -> npt.NDArray[np.float64]:
    """
    Return MR signal based on the diffusive IVIM model with :math:`F_P = e^{-bD^*}`.
    
    Arguments: 
        b:     vector of b-values [s/mm\ :sup:`2`]
        D:     ND array of diffusion coefficients [mm\ :sup:`2`/s]
        f:     ND array of perfusion fractions (same shape as D or scalar)
        Dstar: ND array of pseudo-diffusion coefficients [mm2/s] (same shape as D or scalar)
        S0:    (optional) ND array of signal values at b == 0 (same shape as D or scalar)
        K:     (optional) ND array of kurtosis coefficients (same shape as D or scalar)

    Output:
        S:     (N+1)D array of signal values
    """

    [b, D, f, Dstar, S0] = at_least_1d([b, D, f, Dstar, S0])
    return S0[..., np.newaxis] * ((1-f[..., np.newaxis])*kurtosis(b, D, K) + f[..., np.newaxis]*monoexp(b, Dstar+Db))

def intermediate(b: npt.NDArray[np.float64], delta: npt.NDArray[np.float64], Delta: npt.NDArray[np.float64], D: npt.NDArray[np.float64], f: npt.NDArray[np.float64], v: npt.NDArray[np.float64], tau: npt.NDArray[np.float64], S0: npt.NDArray[np.float64] = 1, K: npt.NDArray[np.float64] = 0, seq = MONOPOLAR, T: npt.NDArray[np.float64] | None = None, k: npt.NDArray[np.float64] | None = None) -> npt.NDArray[np.float64]:
    """
    Return MR signal based on the intermediate IVIM model.
    
    Arguments: 
        b:     vector of b-values [s/mm\ :sup:`2`]
        delta: vector of gradient durations [s]
        Delta: vector of gradient separations [s]
        D:     ND array of diffusion coefficients [mm\ :sup:`2`/s]
        f:     ND array of perfusion fractions (same shape as D or scalar)
        v:     ND array of velocities [mm/s] (same shape as D or scalar)
        tau:   ND array of correlation times [s] (same shape as D or scalar)
        S0:    (optional) ND array of signal values at b == 0 (same shape as D or scalar)
        K:     (optional) ND array of kurtosis coefficients (same shape as D or scalar)
        seq:   (optional) pulse sequence used (monopolar or bipolar)
        T:     (optional) vector of encoding times [s]
        k:     (optional) vector indicating if bipolar pulse sequence is flow compensated or not [-1/1] 

    Output:
        S:     (N+1)D array of signal values
    """

    [b, delta, Delta, T, k, D, f, v, tau, S0] = at_least_1d([b, delta, Delta, T, k, D, f, v, tau, S0])

    G = G_from_b(b, Delta, delta, seq) # G is in T/mm since b is in s/mm2, but this is fine since v is in units mm/s and we only have y*v*G in Fp -> units 1/s2

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

    Arguments:
        b: vector of b-values [s/mm\ :sup:`2`]
        D: ND array of diffusion coefficients [mm\ :sup:`2`/s]

    Output: 
        J: Jacobian matrix
    """
    # warning! alternative to b[np.newaxis,:] may be needed
    J = (monoexp(b, D) * -b[np.newaxis, :])[...,np.newaxis] # D is the only parameter, but we still want the last dimension
    return J

def kurtosis_jacobian(b: npt.NDArray[np.float64], D: npt.NDArray[np.float64], K: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """ 
    Return the Jacobian matrix for the monoexponential expression.

    Arguments:
        b: vector of b-values [s/mm\ :sup:`2`]
        D: ND array of diffusion coefficients [mm\ :sup:`2`/s]
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

    Arguments: 
        b:  vector of b-values [s/mm\ :sup:`2`]
        D:  ND array of diffusion coefficients [mm\ :sup:`2`/s]
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

def ballistic_jacobian(b:  npt.NDArray[np.float64], c: npt.NDArray[np.float64], D: npt.NDArray[np.float64], f: npt.NDArray[np.float64], v: npt.NDArray[np.float64], S0: npt.NDArray[np.float64] | None = None, K: npt.NDArray[np.float64] | None = None) -> npt.NDArray[np.float64]:
    """
    Return the Jacobian matrix for the ballistic IVIM model.

    Arguments: 
        b:  vector of b-values [s/mm\ :sup:`2`]
        c:  vector of c-values [s/mm]
        D:  ND array of diffusion coefficients [mm\ :sup:`2`/s]
        f:  ND array of perfusion fractions (same shape as D or scalar)
        v:  ND array of velocities [mm/s] (same shape as D or scalar)
        S0: (optional) ND array of signal values at b == 0 (same shape as D or scalar)
        K:  (optional) ND array of kurtosis coefficients (same shape as D or scalar)

    Output:
        J:  Jacobian matrix
    """

    [b, c, D, f, v] = at_least_1d([b, c, D, f, v])
    if S0 is not None:
        [S0] = at_least_1d([S0])
    exp2 = monoexp(b,np.atleast_1d(Db)) * monoexp(c**2,v**2/6)

    J_sIVIM = sIVIM_jacobian(b,D,f,S0,K)
    dSdD  = J_sIVIM[..., 0]
    dSdv = f[..., np.newaxis] * exp2 * (-v[..., np.newaxis]@((c**2)[np.newaxis, :])/3)
    if S0 is None:
        dSdf  = J_sIVIM[..., 1] - np.ones_like(f)[..., np.newaxis]@(b==0)[np.newaxis, :] + exp2    
    else:
        dSdf  = J_sIVIM[..., 1] - S0[..., np.newaxis]@(b==0)[np.newaxis, :] + S0[..., np.newaxis]*exp2
        dSdv *= S0[..., np.newaxis]

    if S0 is None:
        if K is None:
            J_list = [dSdD, dSdf, dSdv]
        else:
            J_list = [dSdD, dSdf, dSdv, J_sIVIM[..., 2]]
    else:
        if K is None:
            dSdS0 = ballistic(b,c,D,f,v)
            J_list = [dSdD, dSdf, dSdv, dSdS0]
        else:
            dSdS0 = ballistic(b,c,D,f,v,K=K)
            J_list = [dSdD, dSdf, dSdv, dSdS0, J_sIVIM[..., 3]]

    J = np.stack(J_list, axis=-1)

    return J

def diffusive_jacobian(b: npt.NDArray[np.float64], D: npt.NDArray[np.float64], f: npt.NDArray[np.float64], Dstar: npt.NDArray[np.float64], S0: npt.NDArray[np.float64] | None = None, K: npt.NDArray[np.float64] | None = None) -> npt.NDArray[np.float64]:
    """
    Return the Jacobian matrix for the diffusive IVIM model.

    Arguments: 
        b:     vector of b-values [s/mm\ :sup:`2`]
        D:     ND array of diffusion coefficients [mm\ :sup:`2`/s]
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
    dSdDstar = f[..., np.newaxis] * monoexp(b,Dstar+Db) * -(np.ones_like(f)[..., np.newaxis]@b[np.newaxis, :])
    if S0 is None:
        dSdf  = J_sIVIM[..., 1] - np.ones_like(f)[..., np.newaxis]@(b==0)[np.newaxis, :] + monoexp(b,Dstar+Db)
    else:
        dSdf  = J_sIVIM[..., 1] - S0[..., np.newaxis]@(b==0)[np.newaxis, :] + S0[..., np.newaxis]*monoexp(b,Dstar+Db)
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

def intermediate_jacobian(b:  npt.NDArray[np.float64], delta: npt.NDArray[np.float64], Delta: npt.NDArray[np.float64], 
                          D: npt.NDArray[np.float64], f: npt.NDArray[np.float64], v: npt.NDArray[np.float64], tau: npt.NDArray[np.float64], 
                          S0: npt.NDArray[np.float64] | None = None, K: npt.NDArray[np.float64] | None = None,
                          seq: str = MONOPOLAR, T: npt.NDArray[np.float64] | None = None, k: npt.NDArray[np.float64] | None = None
                          ) -> npt.NDArray[np.float64]:
    """
    Return the Jacobian matrix for the ballistic IVIM model.

    Arguments: 
        b:      vector of b-values [s/mm\ :sup:`2`]
        delta:  vector of deltas [s]
        Delta:  vector of deltas [s]
        D:      ND array of diffusion coefficients [mm\ :sup:`2`/s]
        f:      ND array of perfusion fractions (same shape as D or scalar)
        v:      ND array of velocity [mm/s] (same shape as D or scalar)
        tau:   ND array of correlation times [s] (same shape as D or scalar)
        S0:     (optional) ND array of signal values at b == 0 (same shape as D or scalar)
        K:      (optional) ND array of kurtosis coefficients (same shape as D or scalar)
        seq:    (optional) pulse sequence used (monopolar or bipolar)
        T:      (optional) vector of encoding times [s]
        k:      (optional) vector indicating if bipolar pulse sequence is flow compensated or not [-1/1] 

    Output:
        J:  Jacobian matrix
    """

    [b, D, f, v, tau] = at_least_1d([b, D, f, v, tau])

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
    def dexpdx(a,x):
        return np.exp(-a/x)*(a)/x**2
    dpsidtau = -(2*dexpdx(Deltam,taum) + 2*dexpdx(deltam,taum) - dexpdx((Deltam+deltam),taum) - dexpdx((Deltam-deltam),taum))
    if seq == BIPOLAR:
        t1 *= 2
        t3 *= 2
        t4 *= 2
        t4part2 = taum**4 * km * np.exp(-Tm/taum)*(np.exp((2*Deltam+2*deltam)/taum) - 2*np.exp((2*Deltam+deltam)/taum) + np.exp(2*Deltam/taum) - 2*np.exp((Deltam+2*deltam)/taum) + 4*np.exp((Deltam+deltam)/taum) - 2*np.exp(Deltam/taum) + np.exp(2*deltam/taum) - 2*np.exp(deltam/taum) + 1)
        t4 += t4part2
        dpsidtau *= 2
        domegadtau = km * (np.exp(-Tm/taum)*(dexpdx(-(2*Deltam+2*deltam),taum) - 2*dexpdx(-(2*Deltam+deltam),taum) + dexpdx(-2*Deltam,taum) - 2*dexpdx(-(Deltam+2*deltam),taum) + 4*dexpdx(-(Deltam+deltam),taum) - 2*dexpdx(-Deltam,taum) + dexpdx(-2*deltam,taum) - 2*dexpdx(-deltam,taum))
                           + dexpdx(Tm,taum)*t4part2/(taum**4*km*np.exp(-Tm/taum)))
    else:
        domegadtau = 0

    if S0 is not None:
        [S0] = at_least_1d([S0])
    expb = monoexp(b,np.atleast_1d(Db))
    if K is None:
        Fp = intermediate(b, delta, Delta, D, 1, v, tau, 1, 0, seq, T, k) / expb
    else:
        Fp = intermediate(b, delta, Delta, D, 1, v, tau, 1, K, seq, T, k) / expb

    J_sIVIM = sIVIM_jacobian(b,D,f,S0,K)
    dSdD  = J_sIVIM[..., 0]
    dSdv = f[..., np.newaxis] * expb * Fp * 2 * np.log(Fp)/v[..., np.newaxis]
    
    dSdtau = f[..., np.newaxis] * expb * Fp * (-y**2*Gm**2*(v**2/3)[..., np.newaxis]) * (
        4 * t4 / taum
        + taum**4 * (dpsidtau + domegadtau)
        + 3 * t3 / taum
        + t1 / taum
    ) 
    
    if S0 is None:
        dSdf  = J_sIVIM[..., 1] - np.ones_like(f)[..., np.newaxis]@(b==0)[np.newaxis, :] + expb * Fp    
    else:
        dSdf  = J_sIVIM[..., 1] - S0[..., np.newaxis]@(b==0)[np.newaxis, :] + S0[..., np.newaxis] * expb * Fp
        dSdv *= S0[..., np.newaxis]
        dSdtau *= S0[..., np.newaxis]

    if S0 is None:
        if K is None:
            J_list = [dSdD, dSdf, dSdv, dSdtau]
        else:
            J_list = [dSdD, dSdf, dSdv, dSdtau, J_sIVIM[..., 2]]
    else:
        if K is None:            
            dSdS0 = intermediate(b,delta,Delta,D,f,v,tau,T=T,k=k,seq=seq)
            J_list = [dSdD, dSdf, dSdv, dSdtau, dSdS0]
        else:
            dSdS0 = intermediate(b,delta,Delta,D,f,v,tau,K=K,T=T,k=k,seq=seq)
            J_list = [dSdD, dSdf, dSdv, dSdtau, dSdS0, J_sIVIM[..., 3]]

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