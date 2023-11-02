import nibabel as nib
import numpy as np
import os.path
import tempfile
from ivim.io.base import (data_from_file, file_from_data, 
                          read_im, read_bval, read_cval, read_bvec, 
                          write_im, write_bval, write_cval, write_bvec,
                          check_vector, check_2dmatrix, check_roi,
                          read_time, read_k, write_time, write_k)

# Paths to data
data_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)),'test_data')
temp_folder = tempfile.gettempdir()

# Test functions
def test_data_from_file():
    im_file = os.path.join(data_folder,'test.nii.gz')
    im = read_im(im_file)
    
    bval_file = os.path.join(data_folder,'test.bval')
    b  = read_bval(bval_file)
    
    temp_file = os.path.join(temp_folder,'temp_iodff.nii.gz')
    write_im(temp_file, (np.random.rand(im.shape[0], im.shape[1], im.shape[2]) > 0.5).astype(np.int32))
    roi = read_im(temp_file).astype(bool)

    for bvec_file in [os.path.join(data_folder,'test.bvec'), None]:
        for cval_file in [os.path.join(data_folder,'test.cval'), None]:
            for roi_file in [temp_file, None]:
                out = data_from_file(im_file, bval_file, bvec_file, cval_file, roi_file)
                np.testing.assert_equal(len(out), 2 + (bvec_file is not None) + (cval_file is not None))

                if roi_file is None:
                    Y = np.full(([int(np.prod(im.shape[:3])), b.size]), np.nan)
                    for i in range(b.size):
                        Y[:, i] = im[..., i].flatten()
                else:
                    Y = np.full(([int(np.sum(roi)), b.size]), np.nan)
                    for i in range(b.size):
                        Y[:, i] = im[roi, i]
                np.testing.assert_array_equal(out[0], Y)

                np.testing.assert_array_equal(out[1], b)

                if bvec_file is not None:
                    v = read_bvec(bvec_file)
                if cval_file is not None:
                    c = read_cval(cval_file)
                if bvec_file is None:
                    if cval_file is not None:
                        np.testing.assert_array_equal(out[2], c)
                else:
                    np.testing.assert_array_equal(out[2], v)
                    if cval_file is not None:
                        np.testing.assert_array_equal(out[3], c)

def test_file_from_data():
    sz = (2, 3, 4, 5)
    data = np.random.rand(np.prod(sz[:3]), sz[3])
    filename = os.path.join(temp_folder,'temp_ioffd.nii.gz')
    for imref_file in [os.path.join(data_folder,'test.nii.gz'), None]:
        for roi in [np.random.rand(sz[0],sz[1],sz[2]) > 0.5, None]:
            if (imref_file is None) and (roi is None):
                np.testing.assert_raises(ValueError, file_from_data, filename, data, roi = roi, imref_file = imref_file)
                continue

            if roi is None:
                data2file = data
            else:
                data2file = data[roi.flatten(),:]
            file_from_data(filename, data2file, roi = roi, imref_file = imref_file)

            nii = nib.load(filename)
            if roi is None:
                Y = np.full((np.prod(sz[:3]), sz[3]),np.nan)
                for i in range(sz[3]):
                    Y[:,i] = data[..., i].flatten()
            else:
                Y = np.full((np.sum(roi), sz[3]),np.nan)
                for i in range(sz[3]):
                    Y[:,i] = data[roi.flatten(), i]
            np.testing.assert_equal(Y, data2file)

            if imref_file is None:
                affine = np.eye(4)
            else:
                affine = nib.load(imref_file).affine
            np.testing.assert_equal(nii.affine, affine)

def test_read_im():
    im = np.random.rand(3,4,5)
    nii = nib.Nifti1Image(im, affine = np.eye(4))
    filename = os.path.join(temp_folder,'temp_iori.nii.gz')
    nib.save(nii,filename)
    np.testing.assert_equal(im, read_im(filename))

def test_read_bval():
    b = np.array([0.0, 100.0, 30.2, 180.0])
    np.testing.assert_equal(b, read_cval(os.path.join(data_folder,'test.bval')))

def test_read_cval():
    c = np.array([0.000, 1.234, 2.345, 3.456])
    np.testing.assert_equal(c, read_cval(os.path.join(data_folder,'test.cval')))

def test_read_time():
    t = np.array([0.1, 0.003, 0.023, 0.240, 0.00123])
    np.testing.assert_equal(t, read_time(os.path.join(data_folder,'test.T')))

def test_read_k():
    k = np.array([1, 1, -1, 1, -1])
    np.testing.assert_equal(k, read_time(os.path.join(data_folder,'test.k')))

def test_read_bvec():
    v = np.array([[1.000, -1.000, 0.420, 0.000],
                  [0.000, 0.000, 0.420, 0.123],
                  [0.000, 0.000, 0.000, 0.890]])
    for idx in [0,np.array([0, 1, 2, 3])]: # test limiting case of 1 dir
        np.testing.assert_equal(v[:, idx], read_bvec(os.path.join(data_folder,'test.bvec'))[:, idx])

def test_write_im():
    im = np.random.rand(2,3,4)
    filename = os.path.join(temp_folder,'temp_iowi.nii.gz')
    for imref_file in [os.path.join(data_folder,'test.nii.gz'), None]:
        write_im(filename, im, imref_file)
        nii = nib.load(filename)
        np.testing.assert_equal(im, nii.get_fdata())

        if imref_file is None:
            affine = np.eye(4)
        else:
            affine = nib.load(imref_file).affine
        np.testing.assert_equal(nii.affine, affine)

def test_write_bval():
    b = read_bval(os.path.join(data_folder,'test.bval'))
    temp_file = os.path.join(temp_folder,'temp_iowb.bval')
    write_bval(temp_file, b)
    np.testing.assert_equal(b, read_bval(temp_file))

def test_write_cval():
    c = read_cval(os.path.join(data_folder,'test.cval'))
    temp_file = os.path.join(temp_folder,'temp_iowc.cval')
    write_cval(temp_file, c)
    np.testing.assert_equal(c, read_cval(temp_file))

def test_write_time():
    t = read_time(os.path.join(data_folder,'test.T'))
    temp_file = os.path.join(temp_folder,'temp_iowt.T')
    write_time(temp_file, t)
    np.testing.assert_equal(t, read_time(temp_file))

def test_write_k():
    k = read_k(os.path.join(data_folder,'test.k'))
    temp_file = os.path.join(temp_folder,'temp_iowk.k')
    write_k(temp_file, k)
    np.testing.assert_equal(k, read_k(temp_file))

def test_write_bvec():
    v = read_bvec(os.path.join(data_folder,'test.bvec'))
    temp_file = os.path.join(temp_folder,'temp_iowv.bvec')
    for idx in [0, np.arange(v.shape[1])]: # test limiting case of 1 dir
        write_bvec(temp_file, v[:, idx])
        np.testing.assert_equal(v[:, idx], read_bvec(temp_file)[:, idx])

def test_check_vector():
    np.testing.assert_raises(ValueError,check_vector,np.ones((2,3)),1,'')
    np.testing.assert_raises(ValueError,check_vector,np.ones(5),4,'')

def test_check_2dmatrix():
    np.testing.assert_raises(ValueError,check_2dmatrix,np.ones((3)),1,'')
    np.testing.assert_raises(ValueError,check_2dmatrix,np.ones((3,2)),3,'')
    np.testing.assert_raises(ValueError,check_2dmatrix,np.ones((2,3)),3,'')

def test_check_roi():
    np.testing.assert_raises(ValueError,check_roi,np.full((2,3),True),1)
    np.testing.assert_raises(ValueError,check_roi,np.full((2,3,4),True),[2,3,2])