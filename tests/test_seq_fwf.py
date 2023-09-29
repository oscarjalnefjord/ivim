import numpy as np
from ivim.constants import y
from ivim.seq.sde import calc_b, calc_c
from ivim.seq.fwf import (q_from_g, 
                          B_from_q, b_from_q, C_from_q, c_from_q,
                          B_from_g, b_from_g, C_from_g, c_from_g,
                          G_from_b, c_from_b)

# Variables for testing
t = np.linspace(0, 100e-3, 1000)
dt = t[1] - t[0]
delta = 40e-3
Delta = t[-1] - delta
delta_int = int(delta/dt)

g = np.zeros((t.size, 3))
G = 40e-3
g[1:delta_int+1, 0] = G
g[-delta_int-1:-1, 0] = -G

q = np.zeros_like(g)
q[:delta_int+1, 0] = y * G * t[:delta_int+1]
q[delta_int+1:-delta_int-1, 0] = y * G * t[delta_int]
q[-delta_int-1:, 0] = y * G * (delta + Delta - t[-delta_int-1:])

b = calc_b(G, Delta, delta)
c = calc_c(G, Delta, delta)

atol_q = 1e3
atol_b = 1e8
atol_c = 1e6
atol_g = 0.1e-3
rtol = 1e-2

# Test functions
def test_q_from_g():
    np.testing.assert_allclose(q_from_g(g, dt), q, rtol = rtol, atol = atol_q)

def test_B_from_q():
    B = np.zeros((3,3))
    B[0,0] = b
    np.testing.assert_allclose(B_from_q(q, dt), B, rtol = rtol, atol = atol_b)

def test_b_from_q():
    np.testing.assert_allclose(b_from_q(q, dt), b, rtol = rtol, atol = atol_b)

def test_C_from_q():
    C = np.zeros(3)
    C[0] = c
    np.testing.assert_allclose(C_from_q(q, dt), C, rtol = rtol, atol = atol_c)

def test_c_from_q():
    np.testing.assert_allclose(c_from_q(q, dt), c, rtol = rtol, atol = atol_c)

def test_B_from_g():
    B = np.zeros((3,3))
    B[0,0] = b
    np.testing.assert_allclose(B_from_g(g, dt), B, rtol = rtol, atol = atol_b)

def test_b_from_g():
    np.testing.assert_allclose(b_from_g(g,dt), b, rtol = rtol, atol = atol_b)

def test_C_from_g():
    C = np.zeros(3)
    C[0] = c
    np.testing.assert_allclose(C_from_g(g, dt), C, rtol = rtol, atol = atol_c)

def test_c_from_g():
    np.testing.assert_allclose(c_from_g(g, dt), c, rtol = rtol, atol = atol_c)

def test_G_from_b():
    np.testing.assert_allclose(G_from_b(b, g/G, dt), G, rtol = rtol, atol = atol_g)

def test_c_from_b():
    np.testing.assert_allclose(c_from_b(b, g/G, dt), c, rtol = rtol, atol = atol_c)