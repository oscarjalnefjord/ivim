import numpy as np
import os
import tempfile
from ivim.io.base import read_im, read_bval, read_bvec, read_cval, write_im, write_bval, write_bvec, write_cval
from ivim.preproc.base import average, combine, extract

# Paths to data
temp_folder = tempfile.gettempdir()

# Test functions
def test_average():
    bval_file = os.path.join(temp_folder, 'preproc_avg.bval')
    cval_file = os.path.join(temp_folder, 'preproc_avg.cval')
    b = np.array([0, 0, 0, 10, 10, 10, 15, 15, 15])
    c = np.array([0, 1, 1,  0,  0,  1,  1,  0,  1])
    write_bval(bval_file, b)
    write_cval(cval_file, c)

    im_file = os.path.join(temp_folder, 'preproc_avg.nii.gz')
    im = np.random.rand(2,3,4,b.size)
    write_im(im_file, im)

    outbase = os.path.join(temp_folder, 'preproc_avgout')

    for cvaltest_file in [cval_file, None]:
        for avg_type in ['arith', 'geo', 'test']:
            if avg_type == 'test':
                np.testing.assert_raises(ValueError, average,im_file, bval_file, outbase, cval_file = cvaltest_file, avg_type = avg_type)
            else:
                average(im_file, bval_file, outbase, cval_file = cvaltest_file, avg_type = avg_type)

                b_test = read_bval(outbase+'.bval')
                if cvaltest_file == None:
                    np.testing.assert_array_almost_equal(b_test, np.unique(b), 1)
                    c_test = np.zeros_like(b_test)
                    ucs = [0]
                    c_assert = np.zeros_like(c)
                else:
                    c_test = read_cval(outbase+'.cval')
                    ubs = np.unique(np.stack([b, c], axis=0), axis=1)
                    np.testing.assert_array_almost_equal(b_test, ubs[0, :], 1)
                    np.testing.assert_array_almost_equal(c_test, ubs[1, :], 2)
                    ucs = np.unique(c)
                    c_assert = c

                im_test = read_im(outbase + '.nii.gz')
                for ub in np.unique(b):
                    for uc in ucs:
                        mask = (b == ub) & (c_assert == uc)
                        if np.any(mask):
                            if avg_type == 'arith':
                                im_mean = np.mean(im[..., mask], axis=3)
                            else: # avg_type == 'geo'
                                im_mean = np.prod(im[..., mask], axis = 3)**(1/np.sum(mask))
                            np.testing.assert_allclose(np.squeeze(im_test[..., (ub==b_test)&(uc==c_test)]),
                                                    im_mean, rtol=1e-5, atol = 1e-5)

def test_combine():
    n = 3
    sz = (2, 3, 4)
    dwis = []
    bs = []
    vs = []
    cs = []
    dwi_files = []
    bval_files = []
    bvec_files = []
    cval_files = []
    for i in range(n):
        dwis.append(np.random.rand(sz[0], sz[1], sz[2], 6+i))
        bs.append(np.concatenate([np.array([0]), 100*np.random.rand(5+i)], axis = 0))
        vs.append(np.random.rand(3,6+i))
        cs.append(np.random.rand(6+i))
        base = os.path.join(temp_folder, f'preproc_comb{i}')
        dwi_files.append(base+'.nii.gz')
        bval_files.append(base+'.bval')
        bvec_files.append(base+'.bvec')
        cval_files.append(base+'.cval')
        write_im(dwi_files[-1], dwis[-1])
        write_bval(bval_files[-1], bs[-1])
        write_cval(cval_files[-1], cs[-1])
        write_bvec(bvec_files[-1], vs[-1])
    roi = np.random.rand(sz[0], sz[1], sz[2]) > 0.5
    roi_file = os.path.join(temp_folder,'preproc_comb_roi.nii.gz')
    write_im(roi_file, roi.astype(np.int32))

    outbase = os.path.join(temp_folder, 'preproc_combout')
    for bvectest_files in [bvec_files, None]:
        for cvaltest_files in [cval_files, None]:
            for normb in [0, None]:
                for roitest_file in [roi_file, None]:
                    combine(dwi_files, bval_files, outbase, bvec_files = bvectest_files, cval_files = cvaltest_files, normb = normb, roi_file = roitest_file)

                    im_test = read_im(outbase+'.nii.gz')
                    b_test = read_bval(outbase+'.bval')
                    if bvectest_files != None:
                        v_test = read_bvec(outbase+'.bvec')
                    if cvaltest_files != None:
                        c_test = read_cval(outbase+'.cval')

                    start = 0
                    for dwi, b, v, c in zip(dwis, bs, vs, cs):
                        nb = b.size
                        if normb == None:
                            np.testing.assert_array_equal(im_test[..., start:start+nb], dwi)
                        else:
                            if roitest_file == None:
                                np.testing.assert_array_equal(im_test[..., start:start+nb], dwi/np.quantile(dwi[...,b==normb], 0.75))
                            else:
                                np.testing.assert_array_equal(im_test[..., start:start+nb], dwi/np.median(dwi[roi,:][:, b == normb]))

                        np.testing.assert_array_almost_equal(b_test[start:start+nb], b, 0)

                        if bvectest_files != None:
                            np.testing.assert_array_almost_equal(v_test[:, start:start+nb], v, 2)
                        if cvaltest_files != None:
                            np.testing.assert_array_almost_equal(c_test[start:start+nb], c, 2)

                        start += nb

def test_extract():
    im_file = os.path.join(temp_folder, 'preproc_ex.nii.gz')
    bval_file = os.path.join(temp_folder, 'preproc_ex.bval')
    bvec_file = os.path.join(temp_folder, 'preproc_ex.bvec')
    cval_file = os.path.join(temp_folder, 'preproc_ex.cval')
    b = np.array([0, 10, 15, 10, 0])
    c = np.array([0,  0,  0,  0, 1])
    im = np.random.rand(2, 3, 4, b.size)
    v = np.random.rand(3, b.size)
    write_bval(bval_file, b)
    write_im(im_file, im)
    write_bvec(bvec_file, v)
    write_cval(cval_file, c)

    outbase = os.path.join(temp_folder, 'preproc_exout')

    for bvectest_file in [bvec_file, None]:
        for b_ex in [0, 10]:
            for cvaltest_file in [cval_file, None]:
                for c_ex in [0, None]:
                    if (c_ex == None) != (cvaltest_file == None):
                        np.testing.assert_raises(ValueError, extract, im_file, bval_file, outbase, bvec_file = bvectest_file, b_ex = b_ex, cval_file = cvaltest_file, c_ex = c_ex)
                    else:
                        mask = b == b_ex
                        if (c_ex != None) and (cvaltest_file != None):
                            mask &= c == c_ex
                        if not np.any(mask):
                            np.testing.assert_raises(ValueError, extract, im_file, bval_file, outbase, bvec_file = bvectest_file, b_ex = b_ex, cval_file = cvaltest_file, c_ex = c_ex)
                        else:
                            extract(im_file, bval_file, outbase, bvec_file = bvectest_file, b_ex = b_ex, cval_file = cvaltest_file, c_ex = c_ex)
                            im_test = read_im(outbase+'.nii.gz')
                            np.testing.assert_array_equal(im_test, im[..., mask])

                            b_test = read_bval(outbase+'.bval')
                            np.testing.assert_array_equal(b_test, b[mask])

                            if bvectest_file != None:
                                v_test = read_bvec(outbase+'.bvec')
                                np.testing.assert_array_almost_equal(v_test, v[:, mask], 2)

                            if cvaltest_file != None:
                                c_test = read_cval(outbase+'.cval')
                                np.testing.assert_array_almost_equal(c_test, c[mask], 2)