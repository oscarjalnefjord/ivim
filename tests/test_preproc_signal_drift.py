import numpy as np
import os
import json
import tempfile
from ivim.io.base import read_im, write_im, write_bval
from ivim.preproc.signal_drift import temporal, spatiotemporal

# Paths to data
temp_folder = tempfile.gettempdir()
im_file = os.path.join(temp_folder, 'drift.nii.gz')
bval_file = os.path.join(temp_folder, 'drift.bval')
roi_file = os.path.join(temp_folder, 'drift_roi.nii.gz')

sz = (5, 6, 7)
d = [100, 1]
b = np.array([0, 100, 100, 0, 100, 0, 100, 0, 0])
im = ((np.arange(b.size)*d[1] + d[0])*np.exp(-1e-3*b))[np.newaxis, np.newaxis, np.newaxis, :] * np.ones(list(sz)+[b.size])
im += 1e-2 * np.random.randn(im.shape[0], im.shape[1], im.shape[2], im.shape[3])
write_im(im_file, im)
write_bval(bval_file, b)
write_im(roi_file, np.ones(sz))

outbase = os.path.join(temp_folder, 'test')

rtol = 1e-2
atol = 1e-2

def test_temporal():
    for order in [1, 2, 3]:
        for ctm in ['mean', 'median', 'test']:
            for voxelwise in [True, False]:
                if (order == 3) or ((ctm == 'test') and not voxelwise):
                    np.testing.assert_raises(ValueError, temporal, im_file, bval_file, outbase, roi_file = roi_file, order = order, ctm = ctm, voxelwise = voxelwise)
                else:
                    temporal(im_file, bval_file, outbase, roi_file = roi_file, order = order, ctm = ctm, voxelwise = voxelwise)
                    im_test = read_im(outbase+'_corr.nii.gz')
                    for ub in np.unique(b):
                        np.testing.assert_almost_equal(np.std(im_test[..., b==ub]), 0, 0)

                    if voxelwise:
                        np.testing.assert_allclose(read_im(outbase+'_s0.nii.gz'), np.full(sz, d[0]), rtol = rtol, atol = atol)
                        np.testing.assert_allclose(read_im(outbase+'_d.nii.gz')[..., -1], np.full(sz, d[1]), rtol = rtol, atol = atol)
                        np.testing.assert_allclose(read_im(outbase+'_corrfield.nii.gz'), (np.arange(b.size)*d[1])[np.newaxis, np.newaxis, np.newaxis, :]*np.ones(list(sz)+[b.size])/d[0], rtol = rtol, atol = atol)
                    else:
                        with open(outbase + '.json', 'r') as f:
                            d_test = json.load(f)
                        np.testing.assert_allclose(d_test['s0'], d[0], rtol = rtol, atol = atol)
                        np.testing.assert_allclose(d_test['d'][-1], d[1], rtol = rtol, atol = atol)


def test_spatiotemporal():
    for order in [1, 2, 3]:
        if order == 3:
            np.testing.assert_raises(ValueError, spatiotemporal, im_file, bval_file, outbase, roi_file = roi_file, order = order)
        else:
            spatiotemporal(im_file, bval_file, outbase, roi_file = roi_file, order = order)
            im_test = read_im(outbase+'_corr.nii.gz')
            for ub in np.unique(b):
                np.testing.assert_almost_equal(np.std(im_test[..., b==ub]), 0, 0)

            np.testing.assert_allclose(read_im(outbase+'_corrfield.nii.gz'), (np.arange(b.size)*d[1])[np.newaxis, np.newaxis, np.newaxis, :]*np.ones(list(sz)+[b.size])/d[0], rtol = rtol, atol = atol)
