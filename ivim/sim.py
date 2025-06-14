"""Functions for simulations related to IVIM."""

import numpy as np
import numpy.typing as npt
from ivim.models import sIVIM, diffusive, ballistic, intermediate, check_regime, DIFFUSIVE_REGIME, BALLISTIC_REGIME, INTERMEDIATE_REGIME
from ivim.io.base import write_im, read_im, read_bval, read_cval, write_bval, write_cval, read_time, read_k, write_time, write_k
from ivim.seq.lte import MONOPOLAR, BIPOLAR

def noise(D_file: str, f_file: str, regime: str, bval_file: str, 
          noise_sigma: float, outbase: str, S0_file: str | None = None, 
          K_file: str | None = None, Dstar_file: str | None = None, 
          v_file: str | None = None, cval_file: str | None = None,
          tau_file: str | None = None,delta_file: str | None = None, 
          Delta_file: str | None = None, T_file: str | None = None,
          seq: str = MONOPOLAR, k_file: str | None = None):
    """
    Generate noisy data for the IVIM model at a given regime and noise level based on IVIM parameter maps.
     
    Arguments:
        D_file:      path to nifti file with diffusion coefficients
        f_file:      path to nifti file with perfusion fractions
        regime:      IVIM regime to model: no (= sIVIM), diffusive (long encoding time) or ballistic (short encoding time)
        bval_file:   path to .bval file
        noise_sigma: standard deviation of the noise at b = 0
        outbase:     string used to set the file path to out, e.g. '/folder/out' gives '/folder/out.nii.gz' etc.
        S0_file:     (optional) path to nifti file with signal at b = 0, if None S0 = 1
        K_file:      (optional) path to nifti file with kurtosis coefficients
        Dstar_file:  (optional) path to nifti file with pseudo diffusion coefficients
        v_file:      (optional) path to nifti file with velocities
        cval_file:   (optional) path to .cval file
        tau_file:    (optional) path to nifti file with correlation times
        delta_file:  (optional) path to .delta file
        Delta_file:  (optional) path to .Delta file
        T_file:      (optional) path to .T file
        seq:         (optional) pulse sequence used (monopolar or bipolar)
        k_file:      (optional) path to .k file
    """

    check_regime(regime)

    D = read_im(D_file)
    f = read_im(f_file)
    if S0_file is None:
        S0 = np.ones_like(D)
    else:
        S0 = read_im(S0_file)
    if regime == DIFFUSIVE_REGIME:
        if Dstar_file is None:
            raise ValueError(f'Dstar must be set for "{DIFFUSIVE_REGIME}" regime.')
        Dstar = read_im(Dstar_file)
    elif regime == BALLISTIC_REGIME:
        if v_file is None:
            raise ValueError(f'v must be set for "{BALLISTIC_REGIME}" regime.')
        v = read_im(v_file)
    elif regime == INTERMEDIATE_REGIME:
        if v_file is None:
            raise ValueError(f'v must be set for "{INTERMEDIATE_REGIME}" regime.')
        v = read_im(v_file)
        if tau_file is None:
            raise ValueError(f'tau must be set for "{INTERMEDIATE_REGIME}" regime.')
        tau = read_im(tau_file)
    if K_file is None:
        K = np.zeros_like(D)
    else:
        K = read_im(K_file)
    
    b = read_bval(bval_file)
    if regime == BALLISTIC_REGIME:
        c = read_cval(cval_file)
    if regime == INTERMEDIATE_REGIME:
        delta = read_time(delta_file) # specific read functions would be nicer
        Delta = read_time(Delta_file)
        if seq == BIPOLAR:
            T = read_time(T_file)
            k = read_k(k_file)

    seg = ~np.isnan(D)

    if regime == DIFFUSIVE_REGIME:
        Y = diffusive(b, D, f, Dstar, S0, K)
    elif regime == BALLISTIC_REGIME:
        Y = ballistic(b, c, D, f, v, S0, K)
    elif regime == INTERMEDIATE_REGIME:
        if seq == BIPOLAR:
            Y = intermediate(b, delta, Delta, D, f, v, tau, S0, K, seq, T, k)
        elif seq == MONOPOLAR:
            Y = intermediate(b, delta, Delta, D, f, v, tau, S0, K, seq)
        else:
            raise ValueError(f'Invalid pulse sequence "{seq}".')
    else:
        Y = sIVIM(b, D, f, S0, K)

    if Y.ndim > 4:
        raise ValueError('No support for 5D data and above.')
    elif Y.ndim == 3:
        Y = Y[:, :, np.newaxis, :]
    elif Y.ndim == 2:
        Y = Y[:, np.newaxis, np.newaxis, :]
    elif Y.ndim == 1:
        Y = Y[np.newaxis, np.newaxis, np.newaxis, :]
    sz = np.ones(4, dtype=int)    
    sz[:Y.ndim] = np.array(Y.shape)
    n1 = noise_sigma * np.random.randn(sz[0], sz[1], sz[2], sz[3])
    n2 = noise_sigma * np.random.randn(sz[0], sz[1], sz[2], sz[3])
    Ynoise = np.sqrt((Y+n1)**2 + n2**2)

    write_im(outbase + '.nii.gz', Ynoise, imref_file = D_file)
    write_bval(outbase + '.bval',b)
    if regime == BALLISTIC_REGIME:
        write_cval(outbase + '.cval', c)
    if regime == INTERMEDIATE_REGIME:
        write_time(outbase + '.delta', delta)
        write_time(outbase + '.Delta', Delta)
        if seq == BIPOLAR:
            write_time(outbase + '.T', T)
            write_k(outbase + '.k', k)


def langevin(sigma_v: float,tau: float, dt: float, m: int, n: int = 100) -> npt.NDArray[np.float64]:
    """
    Simulation of random walks (velocities) in 3D according
    to the Langevin equation:

    .. math:: dv = -vdt/\\tau + \\sqrt{2\\sigma^2_v/\\tau}d\\xi

    where dv is the velocity increment occuring in the time dt, v is the current
    velocity, tau is the correlation time, :math:`\\sigma_v` is the standard
    deviation of velocities and :math:`d\\xi` is a Wiener process with
    autocorrelation :math:`\\langle d\\xi(t)d\\xi(t')\\rangle = \\delta(t-t')dt`.

    Arguments:
                sigma_v:  standard deviation of the velocities
                tau:      correlation time
                dt:       time step (should be substantially smaller than tau)
                m:        number of time steps
                n:        (optional) number of random walkers

    Output:
                v:        velocities of the walkers (nxmx3 array)


    Examples:
    ---------
    >>> sigma_v, tau, dt, m = 2, 100e-3, 1e-3, 1e3
    >>> v = langevin(sigma_v, tau, dt, m, n = 100)
    
    """

    v = np.zeros((n,m,3))
    v[:,0,:] = sigma_v * np.random.randn(n,3)
    for i in range(1,m):
        v[:,i,:] = v[:,i-1,:]*(1-dt/tau) + np.sqrt(2*sigma_v**2/tau)*np.random.randn(n,3)*np.sqrt(dt)
    return v


