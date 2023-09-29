import numpy as np
from ivim.constants import ybar, y, Db

# Test functions
def test_y():
    np.testing.assert_approx_equal(ybar, 42.58e6, 4) # Hz/T
    np.testing.assert_approx_equal(y, 268e6, 3)      # rad/T/s

def test_D():
    np.testing.assert_allclose(Db, 1.75e-3, rtol = 0.1, atol = 0.1e-3) # Diffusion coefficient of blood mm2/s