"""
Methods for Cramer-Rao lower bounds optmization of b-value schemes.
"""

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize, Bounds, curve_fit
from ivim.models import sIVIM, diffusive, ballistic, intermediate, sIVIM_jacobian, diffusive_jacobian, ballistic_jacobian, intermediate_jacobian, check_regime, NO_REGIME, DIFFUSIVE_REGIME, BALLISTIC_REGIME, INTERMEDIATE_REGIME
from ivim.seq.lte import calc_b, calc_c, G_from_b, MONOPOLAR, BIPOLAR, check_seq
from ivim.constants import y

def crlb(D: npt.NDArray[np.float64], f: npt.NDArray[np.float64], regime: str, 
         bmax: float = 1000, 
         fitK: bool = False, K: npt.NDArray[np.float64] | None = None,
         bthr: float = 200, 
         Dstar: npt.NDArray[np.float64] | None = None, v: npt.NDArray[np.float64] | None = None, tau: npt.NDArray[np.float64] | None = None,
         seq: str = MONOPOLAR, system_constraints: dict = {}):
    """
    Optimize b-values (and possibly c-values) using Cramer-Rao lower bounds optmization.

    Arguments:
        D:                  diffusion coefficients to optimize over [mm\ :sup:`2`/s]
        f:                  perfusion fractions to optimize over (same size as D)
        regime:             IVIM regime to model: no (= sIVIM), diffusive (long encoding time) or ballistic (short encoding time)
        bmax:               (optional) the largest b-value that can be returned by the optimization
        fitK:               (optional) if True, optimize with the intention to be able to fit K in addition to D and f
        K:                  (optional) kurtosis coefficients to optimize over if fitK and for bias term if minbias
        bthr:               (optional) the smallest non-zero b-value that can be returned by the optimization
        Dstar:              (optional) pseudodiffusion coefficients [mm\ :sup:`2`/s]
        v:                  (optional) velocity for optimization and/or bias term [mm/s]
        tau:                (optional) correlation times [s]
        seq:                (optional) type of diffusion encoding gradient, 'monopolar' or 'bipolar'
        system_constraints: (optional) dictionary with system constraints (key: 'Gmax','t180','risetime')
    
    Output:
        b:           optimized b-values
        a:           fraction of total acquisition time to spend at each b-value in b 
        fc:          booleans telling if each b-value should be acquired with a flow-compensated (fc) or non-flow-compensated pulse sequence


    Examples
    --------

    >>> D = [1e-3,1.1e-3]
    >>> f = [0.1,0.15]
    >>> Dstar = [20e-3,20e-3]
    >>> bmax = 800
    >>> b,a = crlb(D,f,'diffusive',bmax=bmax,Dstar=Dstar)
    >>> print(b)
    [  0.          39.3715187  199.80682516 800.        ]
    >>> print(a)
    [0.15952579 0.30527184 0.35411006 0.18109231]

    """

    def cost(x, n0 = 0, nfc = 0, nt = 0):
        """ 
        x: vector with b-values and possibly fractions 
        n0: number of b = 0 acquisitions (only relevant for regime = 'no') 
        nfc: number of b-values with flow compensated gradients (only relevant for regime = 'ballistic' and seq = 'bipolar')
        """

        nb = (n0 + x.size - nt) // 2 
        b = np.zeros(nb)
        b[n0:] = x[:nb-n0]
        a = x[nb-n0:2*nb-n0]
        if nt >= 2:
            T = None
            if (regime == INTERMEDIATE_REGIME) and (seq == MONOPOLAR):
                delta = x[2*nb-n0:2*nb-n0+nt//2]
                Delta = x[2*nb-n0+nt//2:]
            else:
                delta = x[2*nb-n0]
                Delta = x[2*nb-n0+1]
                if regime == INTERMEDIATE_REGIME:
                    T = x[-(nt-2):]
                    delta = delta * np.ones_like(b)
                    Delta = Delta * np.ones_like(b)
            
        if regime == BALLISTIC_REGIME:
            c = calc_c(G_from_b(b, Delta, delta, seq), Delta, delta, seq)
            if seq == BIPOLAR:
                if nfc > 0:
                    c[-nfc:] = 0
        if regime == INTERMEDIATE_REGIME:
            k = np.ones_like(b)
            if nfc > 0:
                k[-nfc:] = -1

        S0 = np.ones_like(D)
        if regime == DIFFUSIVE_REGIME:
            if fitK:
                J = diffusive_jacobian(b, D, f, Dstar, S0 = S0, K = K)
            else:
                J = diffusive_jacobian(b, D, f, Dstar, S0 = S0)
        elif regime == BALLISTIC_REGIME:
            if fitK:
                J = ballistic_jacobian(b, c, D, f, v, S0 = S0, K = K)
            else: 
                J = ballistic_jacobian(b, c, D, f, v, S0 = S0)
        elif regime == INTERMEDIATE_REGIME:
            if fitK:
                J = intermediate_jacobian(b, delta, Delta, D, f, v, tau, S0 = S0, K = K, seq = seq, T = T, k = k)
            else: 
                J = intermediate_jacobian(b, delta, Delta, D, f, v, tau, S0 = S0, seq = seq, T = T, k = k)
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
            C += np.sum(np.sqrt(Finv[:, 2, 2])/v)
            idxK = 4
        elif regime == INTERMEDIATE_REGIME:
            C += np.sum(np.sqrt(Finv[:, 2, 2])/v)
            C += np.sum(np.sqrt(Finv[:, 3, 3])/tau)
            idxK = 5
        else: # NO_REGIME
            idxK = 3 
        if fitK:
            C += np.sum(np.sqrt(Finv[:, idxK, idxK])/K)
        
        return C

    check_regime(regime)
    check_seq(seq)
    for key, value in system_constraints.items():
        if key not in ['Gmax','t180','risetime']:
            raise ValueError(f'Unknown system constraint parameter "{key}" given.')

    if ((regime == DIFFUSIVE_REGIME) or (regime == BALLISTIC_REGIME)) and fitK:
        raise ValueError(f'CRLB optimization in the {regime} regime with kurtosis fit is not available due to numerical instabilities.')
    
    # Start values
    if regime == NO_REGIME:
        bmin = bthr
    else:
        bmin = 0

    nb = 4 + fitK - 2*(regime == NO_REGIME) + (regime == INTERMEDIATE_REGIME)
    na = 4 + fitK - (regime == NO_REGIME) + (regime == INTERMEDIATE_REGIME)
    
    if seq == MONOPOLAR:
        if regime == BALLISTIC_REGIME:
            nt = 2 # fixed delta/Delta across b-values
        elif regime == INTERMEDIATE_REGIME:
            nt = 2*nb # variable delta/Delta across b-values is needed
        else:
            nt = 0 # no need to optimize for no or diffusive
    else: # BIPOLAR
        if regime == BALLISTIC_REGIME:
            nt = 2 # fixed delta/Delta across b-values
        elif regime == INTERMEDIATE_REGIME:
            nt = 2 + nb # fixed delta/Delta but variable T across b-values
        else:
            nt = 0 # no need to optimize for no or diffusive

    x0 = 1/na * np.ones(nb + na + nt)
    if regime == BALLISTIC_REGIME:
        def ballistic_bias(x):
            delta = x[-2]
            Delta = x[-1]
            b = np.max(x[:nb])
            c = calc_c(G_from_b(b,Delta,delta,seq),Delta,delta,seq)
            if seq == BIPOLAR:
                b = b * np.ones(2)
                c = np.array([c,0])
                k = np.array([1,-1])
                delta = delta * np.ones(2)
                Delta = Delta * np.ones(2)
                T = 2*(Delta+delta) + system_constraints['t180']
            else:
                k = None
                T = None
            return np.squeeze(0.05 - np.abs(1 - ballistic(b,c,D,1,v)/intermediate(b,delta,Delta,D,1,v,tau,seq=seq,T=T,k=k))).flatten()*100
        
        x0[-2] = 20e-3
        if seq == MONOPOLAR:
            x0[-1] = x0[-2] + 1.05*system_constraints['t180']
        else:
            x0[-1] = x0[-2] + 1.05*system_constraints['risetime']
        
        x0[:nb] = np.linspace(0,calc_b(system_constraints['Gmax'],x0[-1],x0[-2],seq),nb)
        
        while np.any(ballistic_bias(x0) < 0) or (x0[-2] <= 0):
            x0[-1] -= 0.5e-3
            x0[-2] -= 0.5e-3
            x0[:nb] = np.linspace(0,calc_b(system_constraints['Gmax'],x0[-1],x0[-2],seq),nb)
        if x0[-2] <= 0:
            raise ValueError('Unable to provide appropriate initial values. Balllistic regime cannot be reached with current system constraints.')
            
    elif regime == INTERMEDIATE_REGIME:
        if seq == MONOPOLAR:
            delta0 = 4e-3 * np.ones(nb)
            x0[nb+na:nb+na+nt//2] = delta0
            Deltamin = delta0[0] + system_constraints['t180']
            Delta0 = np.linspace(Deltamin*1.05,Deltamin*3,nb)
            x0[nb+na+nt//2:] = Delta0
            b = calc_b(system_constraints['Gmax'],Delta0[0],delta0[0],seq=seq)
            if fitK:
                x0[:nb] = np.array([0,b/2,b,b/3,2*b/3,b])
            else:
                x0[:nb] = np.array([0,b/2,b,b/2,b])
        else:
            delta0 = 4e-3
            x0[nb+na] = delta0
            Delta0 = delta0 + 1.1*system_constraints['risetime']
            x0[nb+na+1] = Delta0
            b = calc_b(system_constraints['Gmax'],Delta0,delta0,seq=seq)
            Tmin = 2*(delta0+Delta0) + system_constraints['t180']
            if fitK:
                x0[:nb] = np.array([0,b/2,b,b/3,2*b/3,b])
                x0[nb+na+2:] = np.array([Tmin,Tmin,Tmin,2*Tmin,2*Tmin,2*Tmin])
            else:
                x0[:nb] = np.array([0,b/2,b,b/2,b])
                x0[nb+na+2:] = np.array([Tmin,Tmin,Tmin,2*Tmin,2*Tmin])
    else:
        x0[:nb] = np.logspace(np.log10(bmin+0.01*(bmax-bmin)), np.log10(bmax-0.01*(bmax-bmin)), nb)
        
    # Bounds
    lb = bmin * np.ones(nb+na+nt)
    lb[:nb] += np.arange(nb)
    lb[nb:nb+na] = 0.01
    lb[nb+na:] = 0.0001
    ub = bmax * np.ones(nb+na+nt)
    ub[:nb] -= np.arange(nb, 0, -1) - 1
    ub[nb:nb+na] = 1.0
    if (regime == INTERMEDIATE_REGIME) and (seq == BIPOLAR):
        ub[nb+na:nb+na+2] = 0.05#0.9
        ub[nb+na+2:] = 0.2 #0.5
    else:
        ub[nb+na:] = 0.2 #0.9
    bounds = Bounds(lb, ub, keep_feasible = np.full_like(lb, True))

    # Constraints
    c_suma = {'type':'eq', 'fun':lambda x: (np.sum(x[nb:nb+na]) - 1)*1e6} # sum(a) = 1, 
    if (regime == BALLISTIC_REGIME) or (regime == INTERMEDIATE_REGIME):
        # Minimum difference Delta - delta
        if seq == MONOPOLAR:
            if regime == BALLISTIC_REGIME:
                c_deltaDelta = {'type':'ineq', 'fun':lambda x: (x[nb+na+1] - (x[nb+na]+system_constraints['t180']))*1e0}
            else:
                c_deltaDelta = {'type':'ineq', 'fun':lambda x: (x[nb+na+nt//2:] - (x[nb+na:nb+na+nt//2]+system_constraints['t180']))*1e0}
        else:
            c_deltaDelta = {'type':'ineq', 'fun':lambda x: (x[nb+na+1] - (x[nb+na]+system_constraints['risetime']))*1e0}

        # Maximum G
        if (regime == INTERMEDIATE_REGIME) and (seq == MONOPOLAR):
            c_Gmax = {'type':'ineq', 'fun':lambda x: (system_constraints['Gmax'] - G_from_b(x[:nb],x[nb+na+nt//2:],x[nb+na:nb+na+nt//2],seq=seq))*1e3}
        else:
            c_Gmax = {'type':'ineq', 'fun':lambda x: (system_constraints['Gmax'] - G_from_b(x[:nb],x[nb+na+1],x[nb+na],seq=seq))*1e3}
        
        # Ballistic regime
        if regime == BALLISTIC_REGIME:
            c_ballistic = {'type':'ineq', 'fun': ballistic_bias}

        # Shortest T
        if (regime == INTERMEDIATE_REGIME) and (seq == BIPOLAR):
            c_T = {'type':'ineq', 'fun':lambda x: (x[nb+na+2:] - (2*(x[nb+na]+x[nb+na+1]) + system_constraints['t180']))*1e0}
            constraints = (c_suma, c_deltaDelta,c_Gmax,c_T)
        elif regime == BALLISTIC_REGIME:
            constraints = (c_suma, c_deltaDelta,c_Gmax,c_ballistic)
        else:
            constraints = (c_suma, c_deltaDelta,c_Gmax)
            
    else: # NO or DIFFUSIVE
        constraints = (c_suma)

    # Optimization
    mincost = np.inf
    for nfc in range(1+(nb-2)*(seq == BIPOLAR)):
        cost_regime = lambda x: cost(x, int(regime == NO_REGIME), nfc, nt)
        try:
            res = minimize(cost_regime, x0, bounds = bounds, constraints = constraints, method = 'SLSQP', jac = '3-point')
        except:
            x = x0
            fun = mincost
            print('Warning: Numerical instabilities')
        else:
            x = res.x
            fun = res.fun
        if fun < mincost:
            b = np.zeros(nb+(regime == NO_REGIME))
            b[(regime == NO_REGIME):] = x[:nb]
            a = x[nb:nb+na]
            if seq == BIPOLAR:
                fc = np.full(b.size, False)
                if nfc > 0:
                    fc[-nfc:] = True
            if (regime == BALLISTIC_REGIME) or (regime == INTERMEDIATE_REGIME):
                if (regime == INTERMEDIATE_REGIME) and (seq == MONOPOLAR):
                    delta = x[nb+na:nb+na+nt//2]
                    Delta = x[nb+na+nt//2:]
                else:
                    delta = x[nb+na]
                    Delta = x[nb+na+1]
                    if regime == INTERMEDIATE_REGIME: # and seq == BIPOLAR
                        T = x[nb+na+2:]
            mincost = fun
    if mincost == np.inf:
        raise Warning('No optimum found. Returning nan')
        return np.nan
    if np.all(res.x == x0):
        print('Optimization returned initial values.')

    idx = np.argsort(b)

    if regime == BALLISTIC_REGIME:
        if seq == MONOPOLAR:
            return b[idx], a[idx], delta, Delta
        else:
            return b[idx], a[idx], fc[idx], delta, Delta
    elif regime == INTERMEDIATE_REGIME:
        if seq == MONOPOLAR:
            return b[idx], a[idx], delta[idx], Delta[idx]
        else:
            return b[idx], a[idx], fc[idx], delta, Delta, T[idx]
    else: 
        return b[idx], a[idx]