import numpy as np
from ivim.misc import logIo, halfSampleMode

# Test functions
def test_logIo():
    for x in [np.random.rand(3), np.random.rand(3,4), np.random.rand(3,4,5)]:
        np.testing.assert_allclose(logIo(x), np.log(np.i0(x)),rtol=1e-6,atol=1e-6)

def test_halfSampleMode(): 
    X = np.array([[1,2,3,3,4,0], [4,5,3,1,1,2]])
    hsm = halfSampleMode(X)
    np.testing.assert_equal(hsm, [3, 1])


