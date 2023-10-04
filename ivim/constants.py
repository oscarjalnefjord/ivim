""" Numerical constants relevant for IVIM. """

import numpy as np

# Gyromagnetic constant
ybar = 42.577e6  # Hz/T
y = 2 * np.pi * ybar  # rad/s/T

# Diffusion coefficient of blood
Db = 1.75e-3  # mm2/s, see Ahlgren et al 2016 NMR in Biomedicine