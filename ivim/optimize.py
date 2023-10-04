"""
Methods for Cramer-Rao lower bounds optmization of b-value schemes.
"""

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize, Bounds, curve_fit
from ivim.models import sIVIM, diffusive, ballistic, sIVIM_jacobian, diffusive_jacobian, ballistic_jacobian, check_regime, NO_REGIME, DIFFUSIVE_REGIME, BALLISTIC_REGIME
from ivim.seq.sde import calc_c, G_from_b, MONOPOLAR, BIPOLAR

def crlb(D: npt.NDArray[np.float64], f: npt.NDArray[np.float64], regime: str, 
         bmax: float = 1000, 
         fitK: bool = False, minbias: bool = False, bias_regime: str = DIFFUSIVE_REGIME, K: npt.NDArray[np.float64] | None = None, SNR: float = 100,
         bthr: float = 200, 
         Dstar: npt.NDArray[np.float64] | None = None,
         vd: npt.NDArray[np.float64] | None = None, seq: str = MONOPOLAR, delta: float | None = None, Delta: float | None = None):
    """
    Optimize b-values (and possibly c-values) using Cramer-Rao lower bounds optmization.

    Arguments:
        D:           diffusion coefficients to optimize over [mm2/s]
        f:           perfusion fractions to optimize over (same size as D)
        regime:      IVIM regime to model: no (= sIVIM), diffusive (long encoding time) or ballistic (short encoding time)
        bmax:        (optional) the largest b-value that can be returned by the optimization
        fitK:        (optional) if True, optimize with the intention to be able to fit K in addition to D and f
        minbias:     (optional) if True, include a bias term in cost function. Requires some of the remaining optional arguments
        bias_regime: (optional) specifies model to use for bias term
        K:           (optional) kurtosis coefficients to optimize over if fitK and for bias term if minbias
        SNR:         (optional) expected SNR level at b = 0 to be used to scale the influence of the bias term
    ---- no regime ----
        bthr:        (optional) the smallest non-zero b-value that can be returned by the optimization
    ---- diffusive regime ----
        Dstar:       (optional) pseudodiffusion coefficients for optimization and/or bias term [mm/s]
    ---- ballistic regime ----
        vd:          (optional) velocity dispersion coefficient for optimization and/or bias term [mm/s]
        seq:         (optional) type of diffusion encoding gradient, 'monopolar' or 'bipolar'
        delta:       (optional) duration of diffusion encoding gradients [s]
        Delta:       (optional) separation of diffusion encoding gradients [s]
    
    Output:
        b:           optimized b-values
        a:           fraction of total acquisition time to spend at each b-value in b 
    ---- ballistic regime ----
        fc:          booleans telling if each b-value should be acquired with a flow-compensated (fc) or non-flow-compensated pulse sequence
    """

    def cost(x, n0 = 0, nfc = 0):
        """ 
        x: vector with b-values and possibly fractions 
        n0: number of b = 0 acquisitions (only relevant for regime = 'no') 
        nfc: number of b-values with flow compensated gradients (only relevant for regime = 'ballistic' and seq = 'bipolar')
        """

        nb = (n0 + x.size) // 2 
        b = np.zeros(nb)
        b[n0:] = x[:-nb]
        a = x[-nb:]

        if (regime == BALLISTIC_REGIME) or (bias_regime == BALLISTIC_REGIME):
            c = np.zeros(nb)
            c[nfc:] = calc_c(G_from_b(b[nfc:], Delta, delta, 'bipolar'), Delta, delta, 'bipolar')

        S0 = np.ones_like(D)
        if regime == DIFFUSIVE_REGIME:
            if fitK:
                J = diffusive_jacobian(b, D, f, Dstar, S0 = S0, K = K)
            else:
                J = diffusive_jacobian(b, D, f, Dstar, S0 = S0)
        elif regime == BALLISTIC_REGIME:
            if fitK:
                J = ballistic_jacobian(b, c, D, f, vd, S0 = S0, K = K)
            else: 
                J = ballistic_jacobian(b, c, D, f, vd, S0 = S0)
        else: # NO_REGIME
            if fitK:
                J = sIVIM_jacobian(b, D, f, S0 = S0, K = K)
            else:
                J = sIVIM_jacobian(b, D, f, S0 = S0)
        Finv = np.linalg.inv((a[np.newaxis,np.newaxis,:]*J.transpose(0,2,1))@J)
        C = np.sum(np.sqrt(Finv[:, 0, 0])/D + np.sqrt(Finv[:, 1, 1])/f)
        if regime == DIFFUSIVE_REGIME:
            C += np.sum(np.sqrt(Finv[:, 2, 2])/Dstar)
            idxK = 4
        elif regime == BALLISTIC_REGIME:
            C += np.sum(np.sqrt(Finv[:, 2, 2])/vd)
            idxK = 4
        else: # NO_REGIME
            idxK = 3 
        if fitK:
            C += np.sum(np.sqrt(Finv[:, idxK, idxK])/K)
        
        if minbias:
            if bias_regime == DIFFUSIVE_REGIME:
                if Dstar is None:
                    raise ValueError('Dstar must be set to calculate bias term.')
                else:
                    Y = diffusive(b, D, f, Dstar, K = K)
            elif bias_regime == BALLISTIC_REGIME:
                if vd is None:
                    raise ValueError('vd must be set to calculate bias term.')
                else:
                    Y = ballistic(b, c, D, f, vd, K = K)
            
            p0 = np.array([1e-3, 0.1, 1])
            bounds = np.array([[0, 0, 0], [3e-3, 1, 2]])
            if regime == DIFFUSIVE_REGIME:
                p0 = np.insert(p0, 2, 10e-3)
                bounds = np.insert(bounds, 2, np.array([0, 1]), axis = 1)
                x = b
                if fitK:
                    def fn(x, D, f, Dstar, S0, K):    
                        return diffusive(x, D, f, Dstar, S0, K).squeeze()
                else:
                    def fn(x, D, f, Dstar, S0):    
                        return diffusive(x, D, f, Dstar, S0).squeeze()
            elif regime == BALLISTIC_REGIME:
                p0 = np.insert(p0, 2, 2)
                bounds = np.insert(bounds, 2, np.array([0, 5]), axis = 1)
                x = np.stack((b, c), axis=1)
                if fitK:
                    def fn(x, D, f, vd, S0, K):
                        b = x[:, 0]
                        c = x[:, 1]    
                        return ballistic(b, c, D, f, vd, S0, K).squeeze()
                else:
                    def fn(x, D, f, vd, S0):    
                        b = x
                        return ballistic(b, c, D, f, vd, S0).squeeze()
            else: # NO_REGIME
                x = b
                if fitK:
                    def fn(x, D, f, S0, K):    
                        return sIVIM(x, D, f, S0, K).squeeze()
                else:
                    def fn(x, D, f, S0):    
                        return sIVIM(x, D, f, S0).squeeze()
            if fitK:
                p0 = np.append(p0, 1)
                bounds = np.append(bounds, np.array([0, 5])[:, np.newaxis], axis = 1)
            P = np.full((Y.shape[0], p0.size), np.nan)
            for i, y in enumerate(Y):
                try:
                    P[i, :],_ = curve_fit(fn, x, y, p0=p0, bounds=bounds)
                except:
                    P[i, :] = 1e5
            C += np.sum(np.abs(D - P[:, 0])/D + np.abs(f - P[:, 1])/f) * SNR
            if regime == DIFFUSIVE_REGIME:
                C += np.sum(np.abs(Dstar - P[:, 2])/Dstar) * SNR
                idxK = 4
            elif regime == BALLISTIC_REGIME:
                C += np.sum(np.abs(vd - P[:, 2])/vd) * SNR
                idxK = 4
            else:
                idxK = 3
            if fitK:
                C += np.sum(np.abs(K - P[:, idxK])/K) * SNR

        return C

    check_regime(regime)

    if regime == NO_REGIME:
        if bias_regime not in [DIFFUSIVE_REGIME, BALLISTIC_REGIME]:
            raise ValueError(f'bias_regime must be "{DIFFUSIVE_REGIME}" or "{BALLISTIC_REGIME}"')
    elif (regime == DIFFUSIVE_REGIME) or (regime == BALLISTIC_REGIME):
        if fitK:
            raise ValueError(f'CRLB optimization in the {regime} regime fit kurtosis fit is not available due to numerical instabilities.')
        if minbias:
            raise ValueError(f'CRLB optiomization in the {regime} regime with a bias term is not available due to numerical instabilities.')
    
    if regime == NO_REGIME:
        bmin = bthr
    else:
        bmin = 0

    nb = 4 + fitK - 2*(regime == NO_REGIME)
    na = 4 + fitK - (regime == NO_REGIME)
    x0 = 1/na * np.ones(nb + na)
    x0[:nb] = np.linspace(bmin+0.01*(bmax-bmin), bmax-0.01*(bmax-bmin), nb)
    lb = bmin * np.ones(nb+na)
    lb[:nb] += np.arange(nb)
    lb[nb:] = 0.01
    ub = bmax * np.ones(nb+na)
    ub[:nb] -= np.arange(nb, 0, -1) - 1
    ub[nb:] = 1.0
    bounds = Bounds(lb, ub, keep_feasible = np.full_like(lb, True))

    constraints = ({'type':'eq',   'fun':lambda x: np.sum(x[nb:]) - 1}) # sum(a) = 1

    mincost = np.inf
    for nfc in range(1+(nb-1)*((regime == BALLISTIC_REGIME) and (seq == BIPOLAR))):
        cost_regime = lambda x: cost(x, int(regime == NO_REGIME), nfc)
        res = minimize(cost_regime, x0, bounds = bounds, constraints = constraints, method = 'SLSQP')
        if res.fun < mincost:
            b = np.zeros(nb+(regime == NO_REGIME))
            b[(regime == NO_REGIME):] = res.x[:nb]
            a = res.x[nb:]
            if regime == BALLISTIC_REGIME:
                fc = np.full(b.size, False)
                fc[:nfc] = True
            mincost = res.fun
    idx = np.argsort(b)

    if regime == BALLISTIC_REGIME:
        return b[idx], a[idx], fc[idx]
    else:
        return b[idx], a[idx]