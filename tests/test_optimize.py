import numpy as np
from ivim.models import NO_REGIME, DIFFUSIVE_REGIME, BALLISTIC_REGIME, INTERMEDIATE_REGIME
from ivim.seq.lte import MONOPOLAR, BIPOLAR
from ivim.optimize import crlb

D = np.array([0.5e-3, 1e-3, 1.5e-3])
f = np.array([0.05, 0.15, 0.05])
K = np.array([1.0, 1.2, 2.0])
Dstar = np.array([10e-3, 15e-3, 20e-3])
v = np.array([1.5, 2.0, 2.0])
tau = np.array([100e-3, 150e-3, 200e-3])
bthr = 200

D = [1e-3,1.1e-3]
f = [0.1,0.15]
K = [1,1]
bthr = 200
Dstar = [20e-3,30e-3]
v = [2,3]
tau = [100e-3,150e-3]

system_constraints = {'Gmax':60e-6,'t180':10e-3,'risetime':0.5e-3}

# Test functions
def test_crlb_sIVIM():
    for fitK in [True, False]:
        if fitK:
            bmax = 1500
        else: 
            bmax = 800
        b, a = crlb(D, f, NO_REGIME, bmax = bmax, fitK = fitK, K = K,
                bthr = bthr, Dstar = Dstar)
        
        np.testing.assert_equal(b[0], 0) # First b-value is by design = 0
        np.testing.assert_almost_equal(np.sum(a), 1.0, 2) # a should sum to one
        np.testing.assert_equal(b.size, 3+fitK)
        np.testing.assert_equal(a.size, 3+fitK)
        np.testing.assert_array_less(b, bmax*(1+1e-5))
        np.testing.assert_array_less(bthr, b[1:]*(1+1e-5))
        np.testing.assert_array_less(a, 1.0)
        np.testing.assert_array_less(0.0, a)
                
def test_crlb_diffusive():
    for fitK in [True, False]:
        if fitK:
            bmax = 1500
        else: 
            bmax = 800
        b, a = crlb(D, f, DIFFUSIVE_REGIME, bmax = bmax, fitK = False, K = K, Dstar = Dstar)
        np.testing.assert_almost_equal(np.sum(a), 1.0, 2) # a should sum to one
        np.testing.assert_equal(b.size, 4)
        np.testing.assert_equal(a.size, 4)
        np.testing.assert_array_less(b, bmax*(1+1e-5))
        np.testing.assert_array_less(0, b + 1e-5)
        np.testing.assert_array_less(a, 1.0)
        np.testing.assert_array_less(0.0, a)

def test_crlb_ballistic():
    for fitK in [False]:
        bmax = 1000
        for seq in [MONOPOLAR, BIPOLAR]:
            res = crlb(D, f, BALLISTIC_REGIME, bmax = bmax, fitK = fitK, K = K,
                        v = v, tau = tau, seq = seq, system_constraints=system_constraints)
            if seq == MONOPOLAR:
                b, a, delta, Delta = res
            else:
                b, a, fc, delta, Delta = res
            np.testing.assert_almost_equal(np.sum(a), 1.0, 1) # a should sum to one
            np.testing.assert_equal(b.size, 4+fitK)
            np.testing.assert_equal(a.size, 4+fitK)
            np.testing.assert_array_less(b, bmax*(1+1e-5))
            np.testing.assert_array_less(0, b + 1e-5)
            np.testing.assert_array_less(a, 1.0)
            np.testing.assert_array_less(0.0, a)
            np.testing.assert_array_less(0.0, delta)
            np.testing.assert_array_less(0.0, Delta)
            if seq == BIPOLAR:
                np.testing.assert_equal(fc.size, 4+fitK)
                np.testing.assert_equal(fc.dtype, bool)

def test_crlb_intermediate():
    for fitK in [False]:
        bmax = 1000
        for seq in [MONOPOLAR, BIPOLAR]:
            res = crlb(D, f, INTERMEDIATE_REGIME, bmax = bmax, fitK = fitK, K = K,
                        v = v, tau = tau, seq = seq, system_constraints=system_constraints)
            if seq == MONOPOLAR:
                b, a, delta, Delta = res
            else:
                b, a, fc, delta, Delta, T = res
            np.testing.assert_almost_equal(np.sum(a), 1.0, 1) # a should sum to one
            np.testing.assert_equal(b.size, 5+fitK)
            np.testing.assert_equal(a.size, 5+fitK)
            np.testing.assert_array_less(b, bmax*(1+1e-5))
            np.testing.assert_array_less(0, b + 1e-5)
            np.testing.assert_array_less(a, 1.0)
            np.testing.assert_array_less(0.0, a)
            np.testing.assert_array_less(0.0, delta)
            np.testing.assert_array_less(0.0, Delta)
            if seq == BIPOLAR:
                np.testing.assert_equal(fc.size, 5+fitK)
                np.testing.assert_equal(fc.dtype, bool)
                np.testing.assert_equal(T.size, 5+fitK)

test_crlb_ballistic()