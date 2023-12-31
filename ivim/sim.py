"""Functions for generating noisy image data based on IVIM parameter maps."""

import numpy as np
from ivim.models import sIVIM, diffusive, ballistic, check_regime, DIFFUSIVE_REGIME, BALLISTIC_REGIME
from ivim.io.base import write_im, read_im, read_bval, read_cval, write_bval, write_cval

def noise(D_file: str, f_file: str, regime: str, bval_file: str, 
          noise_sigma: float, outbase: str, S0_file: str | None = None, 
          K_file: str | None = None, Dstar_file: str | None = None, 
          vd_file: str | None = None, cval_file: str | None = None):
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
    ---- diffusive regime ----
        Dstar_file:  (optional) path to nifti file with pseudo diffusion coefficients
    ---- ballistic regime ----
        vd_file:     (optional) path to nifti file with velocity dispersion coefficients
        cval_file:   (optional) path to .cval file
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
        if vd_file is None:
            raise ValueError(f'Dstar must be set for "{BALLISTIC_REGIME}" regime.')
        vd = read_im(vd_file)
    if K_file is None:
        K = np.zeros_like(D)
    else:
        K = read_im(K_file)
    
    b = read_bval(bval_file)
    if regime == BALLISTIC_REGIME:
        c = read_cval(cval_file)

    seg = ~np.isnan(D)

    if regime == DIFFUSIVE_REGIME:
        Y = diffusive(b, D, f, Dstar, S0, K)
    elif regime == BALLISTIC_REGIME:
        Y = ballistic(b, c, D, f, vd, S0, K)
    else:
        Y = sIVIM(b, D, f, S0, K)


    if Y.ndim > 4:
        raise ValueError('No support for 5D data and above.')
    elif Y.ndim == 3:
        Y[..., np.newaxis]
    elif Y.ndim == 2:
        Y[..., np.newaxis, np.newaxis]
    else:
        Y[:, np.newaxis, np.newaxis, np.newaxis]
    sz = np.ones(4, dtype=int)    
    sz[:Y.ndim] = np.array(Y.shape)
    n1 = noise_sigma * np.random.randn(sz[0], sz[1], sz[2], sz[3])
    n2 = noise_sigma * np.random.randn(sz[0], sz[1], sz[2], sz[3])
    Ynoise = np.sqrt((Y+n1)**2 + n2**2)

    write_im(outbase + '.nii.gz', Ynoise, imref_file = D_file)
    write_bval(outbase + '.bval',b)
    if regime == BALLISTIC_REGIME:
        write_cval(outbase + '.cval',c)