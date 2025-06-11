import numpy as np
from ivim.constants import Db, y
from ivim.seq.lte import G_from_b, calc_c, MONOPOLAR, BIPOLAR
from ivim.models import (monoexp, monoexp_jacobian, kurtosis, kurtosis_jacobian, 
                         sIVIM, sIVIM_jacobian, ballistic, ballistic_jacobian,
                         diffusive, diffusive_jacobian, intermediate, intermediate_jacobian, 
                         at_least_1d)

# Tolerances
rtol_models = 1e-5
atol_models = 0
rtol_jac = 1e-3
atol_jac = 0
hrel = 1e-5

# Example values
b = np.array([0, 10, 20, 50, 100])
Delta = 20e-3
delta = 15e-3
T = 2*(Delta+delta) + 10e-3
c = calc_c(G_from_b(b,Delta,delta),Delta,delta) # non-zero to facilitate numerical derivatives
sz = (2,3,4)
Ds =     [1e-3, np.array([1.2e-3, 1e-3]), 0.5e-3 + 1.0e-3 * np.random.rand(sz[0],sz[1],sz[2])]
fs =     [0.3, np.array([0.2, 0.3]), 0.02 + 0.28 * np.random.rand(sz[0],sz[1],sz[2])]
Dstars = [20e-3, np.array([22e-3, 20e-3]), 5e-3 + 25e-3 * np.random.rand(sz[0],sz[1],sz[2])]
vs =     [1.5, np.array([1.2, 1.5]), 1 + 1.5 * np.random.rand(sz[0],sz[1],sz[2])]
taus =   [0.1, np.array([0.05, 0.1]), 0.075 + 0.05 * np.random.rand(sz[0],sz[1],sz[2])]
Ks =     [1, np.array([2, 1]), 1.5 * np.random.rand(sz[0],sz[1],sz[2])]
S0s =    [1, np.array([2, 2]), 50 + 100*np.random.rand(sz[0],sz[1],sz[2])]

# Test functions
def test_monoexp():
    for D in Ds:
        y = np.reshape(np.exp(-np.outer(np.atleast_1d(D),b)), list(np.atleast_1d(D).shape) + [b.size])
        np.testing.assert_allclose(y, monoexp(b, D),rtol=rtol_models,atol=atol_models)

def test_monoexp_jacobian():
    for D in Ds:
        h = np.atleast_1d(D) * hrel
        y = monoexp(b,D)
        yph = monoexp(b,D+h)
        Japp = (yph - y) / h[..., np.newaxis]
        np.testing.assert_allclose(Japp[...,np.newaxis], monoexp_jacobian(b,D),rtol=rtol_jac,atol=atol_jac)

def test_kurtosis():
    for D, K in zip(Ds, Ks):
        y = np.reshape(np.exp(-np.outer(np.atleast_1d(D),b) + np.outer(np.atleast_1d(D)**2 * np.atleast_1d(K)/6,b**2)), list(np.atleast_1d(D).shape) + [b.size])
        np.testing.assert_allclose(y, kurtosis(b, D, K), rtol=rtol_models, atol=atol_models)

def test_kurtosis_jacobian():
    for D, K in zip(Ds, Ks):
        y = kurtosis(b,D,K)
        Japp = np.stack([
                        (kurtosis(b,D*(1+hrel),K) - y) / (np.atleast_1d(D)[..., np.newaxis]*hrel),
                        (kurtosis(b,D,K*(1+hrel)) - y) / (np.atleast_1d(K)[..., np.newaxis]*hrel)
                        ], axis = -1)
        np.testing.assert_allclose(Japp, kurtosis_jacobian(b,D,K), rtol=rtol_jac, atol=atol_jac)

def test_sIVIM():
    for D, f, S0, K in zip(Ds, fs, S0s, Ks):
        y = np.atleast_1d(S0)[..., np.newaxis] * (np.atleast_1d(1-f)[..., np.newaxis]*kurtosis(b,D,K) + np.atleast_1d(f)[..., np.newaxis]*(np.ones_like(f)[..., np.newaxis] @ (b==0)[np.newaxis, :]))
        np.testing.assert_allclose(y, sIVIM(b,D,f,S0,K), rtol=rtol_models, atol=atol_models)

def test_sIVIM_jacobian():
    for D, f, S0, K in zip(Ds, fs, S0s, Ks):
        for S0test in [S0, None]:
            for Ktest in [K, None]:
                if S0test is None:
                    S0calc = 1
                else:
                    S0calc = S0
                if Ktest is None:
                    Kcalc = 0
                else:
                    Kcalc = K

                y = sIVIM(b,D,f,S0calc,Kcalc)

                Jlist = [
                        (sIVIM(b,D*(1+hrel),f,S0calc,Kcalc) - y) / (np.atleast_1d(D)[..., np.newaxis]*hrel),
                        (sIVIM(b,D,f*(1+hrel),S0calc,Kcalc) - y) / (np.atleast_1d(f)[..., np.newaxis]*hrel)
                        ]
                if S0test is not None:
                    Jlist.append((sIVIM(b,D,f,S0calc*(1+hrel),Kcalc) - y) / (np.atleast_1d(S0calc)[..., np.newaxis]*hrel))
                if Ktest is not None:
                    Jlist.append((sIVIM(b,D,f,S0calc,Kcalc*(1+hrel)) - y) / (np.atleast_1d(Kcalc)[..., np.newaxis]*hrel))
                Japp = np.stack(Jlist, axis = -1)

                np.testing.assert_allclose(Japp, sIVIM_jacobian(b,D,f,S0test,Ktest), rtol=rtol_jac, atol=atol_jac)

def test_ballistic():
    for D, f, v, S0, K in zip(Ds, fs, vs, S0s, Ks):
        y = np.atleast_1d(S0)[..., np.newaxis] * (np.atleast_1d(1-f)[..., np.newaxis] * kurtosis(b, D, K) + np.atleast_1d(f)[..., np.newaxis] * np.exp(-np.ones_like(f)[..., np.newaxis] @ b[np.newaxis, :] * Db - np.reshape(np.outer(v**2/6,c**2), list(np.atleast_1d(f).shape) + [b.size])))
        np.testing.assert_allclose(y, ballistic(b, c, D, f, v, S0,K), rtol=rtol_models, atol=atol_models)

def test_ballistic_jacobian():
    for D, f, v, S0, K in zip(Ds, fs, vs, S0s, Ks):
        for S0test in [S0, None]:
            for Ktest in [K, None]:
                if S0test is None:
                    S0calc = 1
                else:
                    S0calc = S0
                if Ktest is None:
                    Kcalc = 0
                else:
                    Kcalc = K

                y = ballistic(b,c,D,f,v,S0calc,Kcalc)

                Jlist = [
                        (ballistic(b,c,D*(1+hrel),f,v,S0calc,Kcalc) - y) / (np.atleast_1d(D)[..., np.newaxis]*hrel),
                        (ballistic(b,c,D,f*(1+hrel),v,S0calc,Kcalc) - y) / (np.atleast_1d(f)[..., np.newaxis]*hrel),
                        (ballistic(b,c,D,f,v*(1+hrel),S0calc,Kcalc) - y) / (np.atleast_1d(v)[..., np.newaxis]*hrel)
                        ]
                if S0test is not None:
                    Jlist.append((ballistic(b,c,D,f,v,S0calc*(1+hrel),Kcalc) - y) / (np.atleast_1d(S0calc)[..., np.newaxis]*hrel))
                if Ktest is not None:
                    Jlist.append((ballistic(b,c,D,f,v,S0calc,Kcalc*(1+hrel)) - y) / (np.atleast_1d(Kcalc)[..., np.newaxis]*hrel))
                Japp = np.stack(Jlist, axis = -1)

                np.testing.assert_allclose(Japp, ballistic_jacobian(b,c,D,f,v,S0test,Ktest), rtol=rtol_jac, atol=atol_jac)

def test_diffusive():
    for D, f, Dstar, S0, K in zip(Ds, fs, Dstars, S0s, Ks):
        y = np.atleast_1d(S0)[..., np.newaxis] * (np.atleast_1d(1-f)[..., np.newaxis] * kurtosis(b, D, K) + np.atleast_1d(f)[..., np.newaxis] * np.exp(-np.reshape(np.outer(Dstar+Db,b), list(np.atleast_1d(D).shape) + [b.size])))
        np.testing.assert_allclose(y, diffusive(b, D, f, Dstar, S0, K), rtol=rtol_models, atol=atol_models)

def test_diffusive_jacobian():
    for D, f, Dstar, S0, K in zip(Ds, fs, Dstars, S0s, Ks):
        for S0test in [S0, None]:
            for Ktest in [K, None]:
                if S0test is None:
                    S0calc = 1
                else:
                    S0calc = S0
                if Ktest is None:
                    Kcalc = 0
                else:
                    Kcalc = K

                y = diffusive(b,D,f,Dstar,S0calc,Kcalc)

                Jlist = [
                        (diffusive(b,D*(1+hrel),f,Dstar,S0calc,Kcalc) - y) / (np.atleast_1d(D)[..., np.newaxis]*hrel),
                        (diffusive(b,D,f*(1+hrel),Dstar,S0calc,Kcalc) - y) / (np.atleast_1d(f)[..., np.newaxis]*hrel),
                        (diffusive(b,D,f,Dstar*(1+hrel),S0calc,Kcalc) - y) / (np.atleast_1d(Dstar)[..., np.newaxis]*hrel)
                        ]
                if S0test is not None:
                    Jlist.append((diffusive(b,D,f,Dstar,S0calc*(1+hrel),Kcalc) - y) / (np.atleast_1d(S0calc)[..., np.newaxis]*hrel))
                if Ktest is not None:
                    Jlist.append((diffusive(b,D,f,Dstar,S0calc,Kcalc*(1+hrel)) - y) / (np.atleast_1d(Kcalc)[..., np.newaxis]*hrel))
                Japp = np.stack(Jlist, axis = -1)

                np.testing.assert_allclose(Japp, diffusive_jacobian(b,D,f,Dstar,S0test,Ktest), rtol=rtol_jac, atol=atol_jac)

def test_intermediate():
    for D, f, v, tau, S0, K in zip(Ds, fs, vs, taus, S0s, Ks):
        for k in [-1, 1]:
            for seq in [MONOPOLAR, BIPOLAR]:
                sz_true  = list(np.atleast_1d(D).shape) + [b.size]
                sz_given = intermediate(b, delta*np.ones_like(b), Delta*np.ones_like(b), D, f, v, tau, S0, K, seq = seq, T = T*np.ones_like(b), k = k*np.ones_like(b)).shape
                np.testing.assert_array_equal(sz_true, sz_given)
    
                phi2_analytical = -np.log(intermediate(b, delta*np.ones_like(b), Delta*np.ones_like(b), 0, 1, v, tau, seq = seq, T = T*np.ones_like(b), k = k*np.ones_like(b))/monoexp(b,Db))*2

                G = G_from_b(b, Delta, delta, seq = seq)
                n = int(1e3)
                t = np.linspace(0,T,n)
                dt = t[1]
                q = np.zeros_like(t)
                q[t<delta] = t[t<delta]
                q[(t>delta) & (t<Delta)] = delta
                q[(t>Delta) & (t<Delta+delta)] = (Delta+delta) - t[(t>Delta) & (t<Delta+delta)]
                if seq == BIPOLAR:
                    q[(t>T-(Delta+delta)) & (t<T-Delta)] = k*(t[(t>T-(Delta+delta)) & (t<T-Delta)] - (T-(Delta+delta)))
                    q[(t>T-Delta) & (t<T-delta)] = k*delta
                    q[(t>T-delta)] = k*(T - t[t>T-delta])
                phi2_numerical =  np.reshape(np.outer(np.ones_like(v), y**2*G**2), list(np.atleast_1d(v).shape) + [G.size]) * (v**2/3 * np.sum(np.sum(np.reshape(np.outer(np.outer(q,q), np.ones_like(tau)), [n, n] + list(np.atleast_1d(tau).shape))*np.exp(-np.reshape(np.outer(np.abs(t[:,np.newaxis]-t[np.newaxis,:]), np.ones_like(tau)), [n, n] + list(np.atleast_1d(tau).shape))/np.atleast_1d(tau)[np.newaxis, np.newaxis, ...]), axis=0), axis=0)*dt**2)[..., np.newaxis]
                np.testing.assert_almost_equal(np.squeeze(phi2_analytical), np.squeeze(phi2_numerical), 2)

def test_intermediate_jacobian():
    deltav = delta * np.ones_like(b)
    Deltav = Delta * np.ones_like(b)
    Tv = T * np.ones_like(b)
    for D, f, v, tau, S0, K in zip(Ds, fs, vs, taus, S0s, Ks):
        for k in [-1, 1]:
            kv = k * np.ones_like(b)
            for seq in [MONOPOLAR, BIPOLAR]:
                for S0test in [S0, None]:
                    for Ktest in [K, None]:
                        if S0test is None:
                            S0calc = 1
                        else:
                            S0calc = S0
                        if Ktest is None:
                            Kcalc = 0
                        else:
                            Kcalc = K

                        y = intermediate(b,deltav,Deltav,D,f,v,tau,S0calc,Kcalc,seq,Tv,kv)

                        Jlist = [
                                (intermediate(b,deltav,Deltav,D*(1+hrel),f,v,tau,S0calc,Kcalc,seq,Tv,kv) - y) / (np.atleast_1d(D)[..., np.newaxis]*hrel),
                                (intermediate(b,deltav,Deltav,D,f*(1+hrel),v,tau,S0calc,Kcalc,seq,Tv,kv) - y) / (np.atleast_1d(f)[..., np.newaxis]*hrel),
                                (intermediate(b,deltav,Deltav,D,f,v*(1+hrel),tau,S0calc,Kcalc,seq,Tv,kv) - y) / (np.atleast_1d(v)[..., np.newaxis]*hrel),
                                (intermediate(b,deltav,Deltav,D,f,v,tau*(1+hrel),S0calc,Kcalc,seq,Tv,kv) - y) / (np.atleast_1d(tau)[..., np.newaxis]*hrel)
                                ]
                        if S0test is not None:
                            Jlist.append((intermediate(b,deltav,Deltav,D,f,v,tau,S0calc*(1+hrel),Kcalc,seq,Tv,kv) - y) / (np.atleast_1d(S0calc)[..., np.newaxis]*hrel))
                        if Ktest is not None:
                            Jlist.append((intermediate(b,deltav,Deltav,D,f,v,tau,S0calc,Kcalc*(1+hrel),seq,Tv,kv) - y) / (np.atleast_1d(Kcalc)[..., np.newaxis]*hrel))
                        Japp = np.stack(Jlist, axis = -1)

                        np.testing.assert_allclose(Japp, intermediate_jacobian(b,deltav,Deltav,D,f,v,tau,S0test,Ktest,seq,Tv,kv), rtol=rtol_jac, atol=atol_jac)

def test_regimes():
    for D, f, v, S0, K in zip(Ds, fs, vs, S0s, Ks):
        for seq,ks_test in zip([MONOPOLAR,BIPOLAR],[[None],[-1,1]]):
            for k_test in ks_test:
                delta_calc = delta * np.ones_like(b)
                Delta_calc = Delta * np.ones_like(b)
                if seq == MONOPOLAR:
                    T_calc = None
                    k_calc = None
                else:
                    T_calc = (10e-3 + 2*(Delta+delta))*np.ones_like(b)
                    k_calc = k_test * np.ones_like(b)

                # Ballistic limit
                if k_test == -1: # FC
                    c = np.zeros_like(b)
                else:
                    c = calc_c(G_from_b(b,Delta,delta,seq=seq),Delta,delta,seq=seq)
                tau_ball = 5000e-3 # aviod longer taus as the instabilities may appear
                lnFp_ball = np.log(intermediate(b,delta_calc,Delta_calc,D,f=1,v=v,tau=tau_ball,seq=seq,T=T_calc,k=k_calc)/monoexp(b,Db))
                np.testing.assert_allclose(lnFp_ball,np.log(ballistic(b,c,D,f=1,v=v)/monoexp(b,Db)), atol=0.05)

                # Diffusive limit        
                tau_diff = 0.1e-3
                lnFp_diff = np.log(intermediate(b,delta_calc,Delta_calc,D,f=1,v=v,tau=tau_diff,seq=seq,T=T_calc,k=k_calc)/monoexp(b,Db))
                np.testing.assert_allclose(lnFp_diff,np.log(diffusive(b,D,f=1,Dstar=v**2*tau_diff/3)/monoexp(b,Db)), atol=0.05)

def test_at_least_1d():
    for x in at_least_1d([np.ones((1,2,3)),np.ones((1,2)),np.ones(1),1.0]):
        np.testing.assert_array_less(-np.ndim(x),0)

