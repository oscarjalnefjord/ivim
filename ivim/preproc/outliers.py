""" Methods for outlier removal. """

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
from ivim.io.base import data_from_file, file_from_data, read_im, write_bval, write_cval
from ivim.models import NO_REGIME, BALLISTIC_REGIME, DIFFUSIVE_REGIME, sIVIM, ballistic, diffusive, check_regime

def roi_based(im_file: str, bval_file: str, roi_file: str, outbase: str, regime:str , fig: bool = False, cval_file: str | None = None):
    """
    Identify outliers by fit to ROI average.

    Arguments:
        im_file:    path to nifti image file
        bval_file:  path to .bval file
        roi_file:   path to nifti file defining a region-of-interest (ROI) in which the correction is calculated and applied
        outbase:    basis for output filenames, i.e. filename without file extension to which .nii.gz, .bval, etc. is added
        regime:     IVIM regime to model: no (= sIVIM), diffusive (long encoding time) or ballistic (short encoding time)
        fig:        (optional) if True, a diagnostic figure is output
        cval_file:  (optional) path to .cval file
    """

    check_regime(regime)
    if regime == BALLISTIC_REGIME:
        Y, b, c = data_from_file(im_file, bval_file, cval_file=cval_file, roi_file=roi_file)
    else:
        Y, b = data_from_file(im_file, bval_file, roi_file=roi_file)

    y_avg = np.median(Y, axis=0)

    if regime == NO_REGIME:
        def model(x, idx):
            return sIVIM(b[idx], x[0], x[1], x[2])
        x0 = [1e-3, 0.1, np.max(y_avg)]
        bounds = ((0, 3e-3), (0, 1), (0, 2*np.max(y_avg)))
    elif regime == BALLISTIC_REGIME:
        def model(x, idx):
            return ballistic(b[idx], c[idx], x[0], x[1], x[2], x[3])
        x0 = [1e-3, 0.1, 2, np.max(y_avg)]
        bounds = ((0, 3e-3), (0, 1), (0, 5), (0, 2*np.max(y_avg)))
    else: # diffusive
        def model(x, idx):
            return diffusive(b[idx], x[0], x[1], x[2], x[3])
        x0 = [1e-3, 0.1, 10e-3, np.max(y_avg)]
        bounds = ((0, 3e-3), (0, 1), (0, 1), (0, 2*np.max(y_avg)))
    

    has_outliers = True
    idx = np.full_like(b, True, dtype=bool)
    while has_outliers:
        def fun(x):
            return np.sum(np.abs(model(x, idx)-y_avg[idx]))
        
        x = minimize(fun, x0, bounds=bounds).x
        res = np.squeeze(model(x, idx)) - y_avg[idx]
        iqr = (np.quantile(res, 0.75) - np.quantile(res, 0.25))
        tmp = np.full_like(b, False, dtype=bool)
        tmp[idx] = np.abs(res) < 3*iqr
        if np.all(tmp == idx):
            has_outliers = False
        else:
            idx = tmp
    
    if fig:
        fig, ax = plt.subplots(1, 1)
        b_plot = np.linspace(np.min(b), np.max(b))
        if regime == NO_REGIME:
            y = np.squeeze(sIVIM(b_plot, x[0], x[1], x[2]))
            
        elif regime == BALLISTIC_REGIME:
            c_plot = b_plot * (c/b)[c>0][np.argmax(b[c>0])]
            y = np.squeeze(ballistic(b_plot, c_plot, x[0], x[1], x[2], x[3]))
            if np.any((b>0)&(c==0)):
                ax.plot(b_plot, np.squeeze(ballistic(b_plot, np.zeros_like(b_plot), x[0], x[1], x[2], x[3])))
        else:
            y = np.squeeze(diffusive(b_plot, x[0], x[1], x[2], x[3]))
        ax.plot(b_plot, y)
        ax.plot(b, y_avg, 'ko')
        ax.plot(b[~idx], y_avg[~idx], 'rx')
        ax.set_xlabel(r'b [s/mm$^2$]')
        ax.set_ylabel('Signal [a.u]')
        fig.savefig(outbase+'.png')
        plt.close(fig)

    file_from_data(outbase+'.nii.gz', Y[:, idx], read_im(roi_file).astype(bool), imref_file=im_file)
    write_bval(outbase+'.bval', b[idx])
    if regime == BALLISTIC_REGIME:
        write_cval(outbase+'.cval', c[idx])