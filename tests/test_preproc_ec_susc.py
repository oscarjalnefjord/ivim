import os
import shutil
import tempfile
import numpy as np
from ivim.io.base import read_im, write_im, write_bval
from ivim.preproc.ec_susc import ec_topup

# Paths to data
temp_folder = tempfile.gettempdir()

# Test functions
if shutil.which('fsl'): # will only run locally
    im_file = os.path.join(temp_folder, 'temp_ec.nii.gz')
    bval_file = os.path.join(temp_folder, 'temp_ec.bval')
    imrev_file = os.path.join(temp_folder, 'temp_ec_rev.nii.gz')
    bvalrev_file = os.path.join(temp_folder, 'temp_ec_rev.bval')
    
    sz = (128, 128, 28)
    b = np.array([0, 0, 0, 100, 100, 100])
    brev = np.array([0, 0, 10])
    write_bval(bval_file, b)
    write_bval(bvalrev_file, brev)
    write_im(im_file, np.random.rand(sz[0], sz[1], sz[2], b.size))
    write_im(imrev_file, np.random.rand(sz[0], sz[1], sz[2], brev.size))

    outbase = os.path.join(temp_folder, 'test_ec_susc_out')

    def test_ec_topup():
        for save_inter in [True, False]:
            for bvalrevtest_file in [bvalrev_file, None]:            
                ec_topup(im_file, bval_file, imrev_file, outbase, bvalrev_file = bvalrevtest_file, save_inter=save_inter)
                np.testing.assert_array_equal(read_im(outbase+'.nii.gz').shape, list(sz)+[b.size])