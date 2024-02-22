import numpy as np
import os
import tempfile
from ivim.models import NO_REGIME, DIFFUSIVE_REGIME, BALLISTIC_REGIME, sIVIM, diffusive, ballistic
from ivim.seq.sde import calc_c, G_from_b
from ivim.io.base import read_im, write_im, write_bval, write_cval, read_bval, read_cval
from ivim.preproc.outliers import roi_based

# Paths to data
temp_folder = tempfile.gettempdir()
im_file = os.path.join(temp_folder, 'outlier.nii.gz')
bval_file = os.path.join(temp_folder, 'outlier.bval')
cval_file = os.path.join(temp_folder, 'outlier.cval')
roi_file = os.path.join(temp_folder, 'outlier_roi.nii.gz')

sz = (5, 6, 7)
b = np.linspace(0,800,100)
delta = 10e-3
Delta = 20e-3
c = calc_c(G_from_b(b,Delta,delta),Delta,delta)
write_bval(bval_file, b)
write_cval(cval_file, c)
write_im(roi_file, np.ones(sz))

outbase = os.path.join(temp_folder, 'test_outlier')

rtol = 1e-2
atol = 1e-2

def test_roi_based():
    for regime in [NO_REGIME, DIFFUSIVE_REGIME, BALLISTIC_REGIME]:
        if regime == BALLISTIC_REGIME:
            cval_file2use = cval_file
        else:
            cval_file2use = None

        ones = np.ones(sz)
        D = 1e-3
        f = 0.1
        S0 = 1
        Dstar = 10e-3
        vd = 2
        if regime == NO_REGIME:
            im = sIVIM(b, D*ones, f*ones, S0*ones)
        elif regime == BALLISTIC_REGIME:
            im = ballistic(b, c, D, f, vd, S0)
        else:
            im = diffusive(b, D, f, Dstar, S0)

        idx = 45
        outlier = np.full(b.shape, False)
        outlier[idx] = True
        # <create some residuals and an outlier>
        write_im(im_file, im)

        for fig in [True, False]:
            roi_based(im_file,bval_file,roi_file,outbase,regime,fig=fig,cval_file=cval_file2use)
            #np.testing.assert_equal(im[...,~outlier], read_im(outbase+'.nii.gz'))
            #np.testing.assert_equal(b[~outlier], read_bval(outbase+'.bval'))
            if regime == BALLISTIC_REGIME:
                #np.testing.assert_equal(c[~outlier], read_cval(outbase+'.cval'))
                pass