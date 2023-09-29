import os 
import tempfile
import numpy as np
from ivim.constants import y
from ivim.io.base import write_bval, read_cval
from ivim.seq.sde import calc_b, calc_c, G_from_b, cval_from_bval

# Paths to data
temp_folder = tempfile.gettempdir()

# Parameter values 
G = np.array([10e-3, 30e-3, 60e-3]) 
Delta = 20e-3
delta = 10e-3
seqs = ['monopolar', 'bipolar', 'wrong']

# Test functions
def test_calc_b():
    for seq in seqs:
        b = y**2 * G**2 * delta**2 * (Delta - delta/3)
        if seq == 'wrong':
            np.testing.assert_raises(ValueError,calc_b,G, Delta, delta, seq = seq)
        else:    
            if seq == 'bipolar':
                b *= 2
            np.testing.assert_allclose(calc_b(G, Delta, delta, seq = seq), b, rtol = 1e-5, atol = 1e-3)

def test_calc_c():
    for seq in seqs:
        for fc in [True, False]:
            if fc:
                c = np.zeros_like(G)
            else:
                c = y * G * delta * Delta
            if seq == 'wrong':
                np.testing.assert_raises(ValueError,calc_c,G, Delta, delta, seq = seq)
            else:    
                if seq == 'bipolar':
                    c *= 2
                if (seq == 'monopolar') and fc:
                    np.testing.assert_raises(ValueError,calc_c,G, Delta, delta, seq = seq, fc = fc)
                else:   
                    np.testing.assert_allclose(calc_c(G, Delta, delta, seq = seq, fc = fc), c, rtol = 1e-5, atol = 1e-3)


def test_G_from_b():
    for seq in seqs[:2]:
        b = calc_b(G, Delta, delta, seq = seq)
        np.testing.assert_allclose(G_from_b(b, Delta, delta, seq = seq), G, rtol = 1e-5, atol = 1e-3)

def test_cval_from_bval():
    bval_file = os.path.join(temp_folder, 'sde.bval')
    cval_file = os.path.join(temp_folder, 'sde.cval')

    for seq in seqs[:2]:
        for fc in [True, False]:
            if (seq == 'monopolar') and fc:
                continue            
            b = calc_b(G, Delta, delta, seq = seq)
            write_bval(bval_file, b)
            c = calc_c(G, Delta, delta, seq = seq, fc = fc)
            cval_from_bval(bval_file, Delta, delta, seq = seq, fc = fc, cval_file = cval_file)
            np.testing.assert_array_almost_equal(read_cval(cval_file), c, 2)