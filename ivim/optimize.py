"""
Methods for Cramer-Rao lower bounds optmization of b-value schemes.
"""

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize, Bounds, curve_fit
from ivim.models import sIVIM, diffusive, ballistic, intermediate, sIVIM_jacobian, diffusive_jacobian, ballistic_jacobian, intermediate_jacobian, check_regime, NO_REGIME, DIFFUSIVE_REGIME, BALLISTIC_REGIME, INTERMEDIATE_REGIME
from ivim.seq.sde import calc_b, calc_c, G_from_b, MONOPOLAR, BIPOLAR, check_seq
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
        D:           diffusion coefficients to optimize over [mm2/s]
        f:           perfusion fractions to optimize over (same size as D)
        regime:      IVIM regime to model: no (= sIVIM), diffusive (long encoding time) or ballistic (short encoding time)
        bmax:        (optional) the largest b-value that can be returned by the optimization
        fitK:        (optional) if True, optimize with the intention to be able to fit K in addition to D and f
        minbias:     (optional) if True, include a bias term in cost function. Requires some of the remaining optional arguments
        bias_regime: (optional) specifies model to use for bias term
        K:           (optional) kurtosis coefficients to optimize over if fitK and for bias term if minbias
        SNR:         (optional) expected SNR level at b = 0 to be used to scale the influence of the bias term
        bthr:        (optional) the smallest non-zero b-value that can be returned by the optimization
        Dstar:       (optional) pseudodiffusion coefficients for optimization and/or bias term [mm/s]
        v:           (optional) velocity dispersion coefficient for optimization and/or bias term [mm/s]
        seq:         (optional) type of diffusion encoding gradient, 'monopolar' or 'bipolar'
        delta:       (optional) duration of diffusion encoding gradients [s]
        Delta:       (optional) separation of diffusion encoding gradients [s]
    
    Output:
        b:           optimized b-values
        a:           fraction of total acquisition time to spend at each b-value in b 
        fc:          booleans telling if each b-value should be acquired with a flow-compensated (fc) or non-flow-compensated pulse sequence


    Examples
    --------

    >>> crlb()

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

    def soma(cfun, bounds, idx_a, psize, migrations, constraints = (), step = 0.5, path = 3):
        def const_ok(x):
            ok = True
            for constraint in constraints:
                if constraint['type'] == 'ineq':
                    ok &= np.all(constraint['fun'](x) > 0)
                elif constraint['type'] == 'eq':
                    ok &= np.all(np.abs(constraint['fun'](x)) < 1e-5)
            return ok

        def identify_leader(X):
            cmin = np.inf
            idx_leader = 0
            for i in range(X.shape[0]):
                if const_ok(X[i,:]) and (cfun(X[i,:]) < cmin):
                    idx_leader = i
            return idx_leader, cfun(X[idx_leader])

        n = bounds[0].size
        na = idx_a.size
        nb = np.min(idx_a)
        bmin = np.min(bounds[0][:nb])
        bmax = np.max(bounds[1][:nb])
        tmin = np.min(bounds[0][na+nb:])
        tmax = np.max(bounds[1][na+nb:])
        chist = np.zeros(migrations)

        # generate initial population and find the leader
        X = np.random.rand(psize, n)
        X[:,idx_a] /= np.sum(X[:,idx_a],axis=1)[:,np.newaxis] # a sum to 1
        for idx in range(X.shape[0]):
            X[idx,:nb] = bmin + (bmax-bmin)*np.random.rand(nb)
            if n > (na+nb):
                X[idx,na+nb:] = tmin + (tmax-tmin)*np.random.rand(n-na-nb)
        idx_leader,chist[0] = identify_leader(X)

        # migrate
        for m in range(1,migrations):
            # migrate each individual towards the leader
            for idx in range(X.shape[0]):
                if idx == idx_leader:
                    continue
            
                x0 = X[idx,:].copy()
                x = X[idx,:].copy()
                for pi, p in enumerate(np.arange(path,step)):
                    # take a step in a random (single) direction towards the leader
                    v = np.random.randint(X.shape[1])
                    x[v] = x0[v] + (X[idx_leader,v] - x0)*p
                    x[idx_a] /= np.sum(x[idx_a])

                    # check that the step was valid
                    bmask = (x < bounds[0]) | (x > bounds[1]) 
                    if np.any(bmask):
                        x[bmask] = bounds[0][bmask] + (bounds[1]-bounds[0])[bmask]*np.random.rand(np.sum(bmask))
                    
                    if (cfun(x) < cfun(x0)):
                        X[idx,:] = x


            # find the new leader
            idx_leader, chist[m] = identify_leader(X)

            # potential early stop


        return x, chist

    check_regime(regime)
    check_seq(seq)
    for key, value in system_constraints.items():
        if key not in ['Gmax','t180','risetime']:
            raise ValueError(f'Unknown system constraint parameter "{key}" given.')

#    if regime == NO_REGIME:
#        if bias_regime not in [DIFFUSIVE_REGIME, BALLISTIC_REGIME]:
#            raise ValueError(f'bias_regime must be "{DIFFUSIVE_REGIME}" or "{BALLISTIC_REGIME}"')
#    elif (regime == DIFFUSIVE_REGIME) or (regime == BALLISTIC_REGIME):
#        if fitK:
#            raise ValueError(f'CRLB optimization in the {regime} regime fit kurtosis fit is not available due to numerical instabilities.')
#        if minbias:
#            raise ValueError(f'CRLB optimization in the {regime} regime with a bias term is not available due to numerical instabilities.')
    
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
#        res = soma(cost_regime, [lb,ub], np.arange(nb,nb+na), psize=1000, migrations=20, constraints = constraints, step = 0.5, path = 3)
#        print(res)
        res = minimize(cost_regime, x0, bounds = bounds, constraints = constraints, method = 'SLSQP', jac = '3-point')
        if res.fun < mincost:
            b = np.zeros(nb+(regime == NO_REGIME))
            b[(regime == NO_REGIME):] = res.x[:nb]
            a = res.x[nb:nb+na]
            if seq == BIPOLAR:
                fc = np.full(b.size, False)
                if nfc > 0:
                    fc[-nfc:] = True
            if (regime == BALLISTIC_REGIME) or (regime == INTERMEDIATE_REGIME):
                if (regime == INTERMEDIATE_REGIME) and (seq == MONOPOLAR):
                    delta = res.x[nb+na:nb+na+nt//2]
                    Delta = res.x[nb+na+nt//2:]
                else:
                    delta = res.x[nb+na]
                    Delta = res.x[nb+na+1]
                    if regime == INTERMEDIATE_REGIME: # and seq == BIPOLAR
                        T = res.x[nb+na+2:]
            mincost = res.fun
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