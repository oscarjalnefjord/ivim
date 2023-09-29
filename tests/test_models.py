import numpy as np
from ivim.constants import Db
from ivim.seq.sde import G_from_b, calc_c
from ivim.models import (monoexp, monoexp_jacobian, kurtosis, kurtosis_jacobian, 
                         sIVIM, sIVIM_jacobian, ballistic, ballistic_jacobian,
                         diffusive, diffusive_jacobian, at_least_1d)

# Tolerances
rtol = 1e-6
atol_models = 1e-8
atol_jac = 1e-4

# Example values
b = np.array([0, 10, 20, 50, 100])
Delta = 20e-3
delta = 15e-3
c = calc_c(G_from_b(b,Delta,delta),Delta,delta) # non-zero to facilitate numerical derivatives
sz = (2,3,4)
Ds =    [1e-3, np.array([1.2e-3, 1e-3]), 0.5e-3 + 1.0e-3 * np.random.rand(sz[0],sz[1],sz[2])]
fs =     [0.3, np.array([0.2, 0.3]), 0.02 + 0.28 * np.random.rand(sz[0],sz[1],sz[2])]
vds =    [1.5, np.array([1.2, 1.5]), 1 + 1.5 * np.random.rand(sz[0],sz[1],sz[2])]
Dstars = [20e-3, np.array([22e-3, 20e-3]), 5e-3 + 25e-3 * np.random.rand(sz[0],sz[1],sz[2])]
Ks =     [1, np.array([2, 1]), 0.5 + 2 * np.random.rand(sz[0],sz[1],sz[2])]
S0s =    [1, np.array([2, 2]), 50 + 100*np.random.rand(sz[0],sz[1],sz[2])]

# Test functions
def test_monoexp():
    for D in Ds:
        y = np.reshape(np.exp(-np.outer(np.atleast_1d(D),b)), list(np.atleast_1d(D).shape) + [b.size])
        np.testing.assert_allclose(y, monoexp(b, D),rtol=rtol,atol=atol_models)

def test_monoexp_jacobian():
    for D in Ds:
        h = np.atleast_1d(D) * rtol
        y = monoexp(b,D)
        yph = monoexp(b,D+h)
        Japp = (yph - y) / h[..., np.newaxis]
        np.testing.assert_allclose(Japp[...,np.newaxis], monoexp_jacobian(b,D),rtol=rtol,atol=atol_jac)

def test_kurtosis():
    for D, K in zip(Ds, Ks):
        y = np.reshape(np.exp(-np.outer(np.atleast_1d(D),b) + np.outer(np.atleast_1d(D)**2 * np.atleast_1d(K)/6,b**2)), list(np.atleast_1d(D).shape) + [b.size])
        np.testing.assert_allclose(y, kurtosis(b, D, K), rtol=rtol, atol=atol_models)

def test_kurtosis_jacobian():
    for D, K in zip(Ds, Ks):
        y = kurtosis(b,D,K)
        Japp = np.stack([
                        (kurtosis(b,D*(1+rtol),K) - y) / (np.atleast_1d(D)[..., np.newaxis]*rtol),
                        (kurtosis(b,D,K*(1+rtol)) - y) / (np.atleast_1d(K)[..., np.newaxis]*rtol)
                        ], axis = -1)
        np.testing.assert_allclose(Japp, kurtosis_jacobian(b,D,K), rtol=rtol, atol=atol_jac)

def test_sIVIM():
    for D, f, S0, K in zip(Ds, fs, S0s, Ks):
        y = np.atleast_1d(S0)[..., np.newaxis] * (np.atleast_1d(1-f)[..., np.newaxis]*kurtosis(b,D,K) + np.atleast_1d(f)[..., np.newaxis]*(np.ones_like(f)[..., np.newaxis] @ (b==0)[np.newaxis, :]))
        np.testing.assert_allclose(y, sIVIM(b,D,f,S0,K), rtol=rtol, atol=atol_models)

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
                        (sIVIM(b,D*(1+rtol),f,S0calc,Kcalc) - y) / (np.atleast_1d(D)[..., np.newaxis]*rtol),
                        (sIVIM(b,D,f*(1+rtol),S0calc,Kcalc) - y) / (np.atleast_1d(f)[..., np.newaxis]*rtol)
                        ]
                if S0test is not None:
                    Jlist.append((sIVIM(b,D,f,S0calc*(1+rtol),Kcalc) - y) / (np.atleast_1d(S0calc)[..., np.newaxis]*rtol))
                if Ktest is not None:
                    Jlist.append((sIVIM(b,D,f,S0calc,Kcalc*(1+rtol)) - y) / (np.atleast_1d(Kcalc)[..., np.newaxis]*rtol))
                Japp = np.stack(Jlist, axis = -1)

                np.testing.assert_allclose(Japp, sIVIM_jacobian(b,D,f,S0test,Ktest), rtol=rtol, atol=atol_jac)

def test_ballistic():
    for D, f, vd, S0, K in zip(Ds, fs, vds, S0s, Ks):
        y = np.atleast_1d(S0)[..., np.newaxis] * (np.atleast_1d(1-f)[..., np.newaxis] * kurtosis(b, D, K) + np.atleast_1d(f)[..., np.newaxis] * np.exp(-np.ones_like(f)[..., np.newaxis] @ b[np.newaxis, :] * Db - np.reshape(np.outer(vd**2,c**2), list(np.atleast_1d(f).shape) + [b.size])))
        np.testing.assert_allclose(y, ballistic(b, c, D, f, vd, S0,K), rtol=rtol, atol=atol_models)

def test_ballistic_jacobian():
    for D, f, vd, S0, K in zip(Ds, fs, vds, S0s, Ks):
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

                y = ballistic(b,c,D,f,vd,S0calc,Kcalc)

                Jlist = [
                        (ballistic(b,c,D*(1+rtol),f,vd,S0calc,Kcalc) - y) / (np.atleast_1d(D)[..., np.newaxis]*rtol),
                        (ballistic(b,c,D,f*(1+rtol),vd,S0calc,Kcalc) - y) / (np.atleast_1d(f)[..., np.newaxis]*rtol),
                        (ballistic(b,c,D,f,vd*(1+rtol),S0calc,Kcalc) - y) / (np.atleast_1d(vd)[..., np.newaxis]*rtol)
                        ]
                if S0test is not None:
                    Jlist.append((ballistic(b,c,D,f,vd,S0calc*(1+rtol),Kcalc) - y) / (np.atleast_1d(S0calc)[..., np.newaxis]*rtol))
                if Ktest is not None:
                    Jlist.append((ballistic(b,c,D,f,vd,S0calc,Kcalc*(1+rtol)) - y) / (np.atleast_1d(Kcalc)[..., np.newaxis]*rtol))
                Japp = np.stack(Jlist, axis = -1)

                np.testing.assert_allclose(Japp, ballistic_jacobian(b,c,D,f,vd,S0test,Ktest), rtol=rtol, atol=atol_jac)

def test_diffusive():
    for D, f, Dstar, S0, K in zip(Ds, fs, Dstars, S0s, Ks):
        y = np.atleast_1d(S0)[..., np.newaxis] * (np.atleast_1d(1-f)[..., np.newaxis] * kurtosis(b, D, K) + np.atleast_1d(f)[..., np.newaxis] * np.exp(-np.reshape(np.outer(Dstar,b), list(np.atleast_1d(D).shape) + [b.size])))
        np.testing.assert_allclose(y, diffusive(b, D, f, Dstar, S0, K), rtol=rtol, atol=atol_models)

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
                        (diffusive(b,D*(1+rtol),f,Dstar,S0calc,Kcalc) - y) / (np.atleast_1d(D)[..., np.newaxis]*rtol),
                        (diffusive(b,D,f*(1+rtol),Dstar,S0calc,Kcalc) - y) / (np.atleast_1d(f)[..., np.newaxis]*rtol),
                        (diffusive(b,D,f,Dstar*(1+rtol),S0calc,Kcalc) - y) / (np.atleast_1d(Dstar)[..., np.newaxis]*rtol)
                        ]
                if S0test is not None:
                    Jlist.append((diffusive(b,D,f,Dstar,S0calc*(1+rtol),Kcalc) - y) / (np.atleast_1d(S0calc)[..., np.newaxis]*rtol))
                if Ktest is not None:
                    Jlist.append((diffusive(b,D,f,Dstar,S0calc,Kcalc*(1+rtol)) - y) / (np.atleast_1d(Kcalc)[..., np.newaxis]*rtol))
                Japp = np.stack(Jlist, axis = -1)

                np.testing.assert_allclose(Japp, diffusive_jacobian(b,D,f,Dstar,S0test,Ktest), rtol=rtol, atol=atol_jac)

def test_at_least_1d():
    for x in at_least_1d([np.ones((1,2,3)),np.ones((1,2)),np.ones(1),1.0]):
        np.testing.assert_array_less(-np.ndim(x),0)