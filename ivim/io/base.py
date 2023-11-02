"""
Functions to read and write dMRI image data and parameter files.
"""

import numpy as np
import numpy.typing as npt
import nibabel as nib

def data_from_file(im_file: str, bval_file: str, bvec_file: str | None = None, cval_file: str | None = None, roi_file: str | None = None) -> (
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] |
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]] |
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]] |
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64], npt.NDArray[np.float64]
        ):
    """
    Load image data (optionally from an ROI/mask) into a 2D array along with b-values, b-vectors, c-values.

    Arguments:
        im_file:   path to nifti image file
        bval_file: path to .bval file
        bvec_file: (optional) path to .bvec file
        cval_file: (optional) path to .cval file
        roi_file:  (optional) path to nifti file defining a region-of-interest (ROI) from with data is extracted

    Output:
        Y:         2D array containing the image data [number of voxels x number of b-values]
        b:         vector containing the b-values
        v:         matrix containing the diffusion encoding directions
        c:         vector containgin the c-values (flow encoding)
    """

    im = read_im(im_file)

    if im.ndim == 3:
        nb = 1
    elif im.ndim != 4:
        raise ValueError(f'The image must have four dimensions, or in expections three, but loaded image had {im.ndim}.')
    else:
        nb = im.shape[3]

    b = read_bval(bval_file)
    check_vector(b, nb, 'b-values')

    if cval_file is not None:
        c = read_cval(cval_file)
        check_vector(c, nb, 'c-values')

    if bvec_file is not None:
        v = read_bvec(bvec_file)
        check_2dmatrix(v, nb, 'b-vectors')

    if roi_file is not None:
        roi = nib.load(roi_file).get_fdata().astype(bool)
        check_roi(roi, im.shape[:3])
    else:
        roi = np.full(im.shape[:-1], True)

    Y = im[roi, :]

    if cval_file is None:
        if bvec_file is None:
            return Y, b
        else:
            return Y, b, v
    else:
        if bvec_file is None:
            return Y, b, c
        else:
            return Y, b, v, c

def file_from_data(filename:str, data: npt.NDArray[np.float64], roi: npt.NDArray[np.bool_] | None = None, imref_file: str | None = None) -> None:
    """
    Save image data in Nifti format based on input 2D array originating from optional roi.

    Arguments:
        filename:   path to Nifti file
        data:       image data in v x n array (v is number of voxels, n gives size of 4th dimension in final image)
        roi:        (optional) region-of-interest from which data is assumed to originate. The number of True elements must match first dimension of data
        imref_file: (optional) path to nifti file from which header info is obtained
    """

    if imref_file is not None:
        nii_ref = nib.load(imref_file).get_fdata()
        if roi is None:
            roi = np.full(nii_ref.shape[:3], True)
    else:
        if roi is None:
            raise ValueError('Either roi or imref_file must be specified to derive the size of the image.')
    roi = roi.astype(bool)
    sz = roi.shape

    if data.ndim > 1:
        im = np.full(list(sz) + [data.shape[1]], np.nan)
        im[roi, :] = data
    else:
        im = np.full(sz, np.nan)
        im[roi] = data
    
    write_im(filename, im, imref_file=imref_file)

########################
# Write/read functions #
########################
def read_im(filename: str | None) -> (npt.NDArray[np.float64] | None):
    """
    Load image in Nifti format.

    Arguments:
        filename: path to image file

    Output:
        im:       image (or None if filename is None)
    """

    if filename is None:
        im = None
    else:
        im = nib.load(filename).get_fdata()
    return im

def read_bval(filename: str) -> npt.NDArray[np.float64]:
    """
    Load c-values from file in format similar to FSL bval format.

    Arguments:
        filename: path to cval file

    Output:
        b:        b-values
    """

    b = np.atleast_1d(np.loadtxt(filename))
    return b

def read_cval(filename: str) -> npt.NDArray[np.float64]:
    """
    Load c-values from file in format similar to FSL bval format.

    Arguments:
        filename: path to cval file

    Output:
        c:        c-values
    """
    
    c = np.atleast_1d(np.loadtxt(filename))
    return c

def read_time(filename: str) -> npt.NDArray[np.float64]:
    """
    Load time parameter from file in format similar to FSL bval format.

    Arguments:
        filename: path to time parameter file

    Output:
        t:        time parameter
    """
    
    t = np.atleast_1d(np.loadtxt(filename))
    return t

def read_k(filename: str) -> npt.NDArray[np.float64]:
    """
    Load k (for intermediate regime) from file in format similar to FSL bval format.

    Arguments:
        filename: path to k file

    Output:
        k:        k (+/- 1) 
    """
    
    k = np.atleast_1d(np.loadtxt(filename))
    return k

def read_bvec(filename: str) -> npt.NDArray[np.float64]:
    """
    Load encoding directions from file in FSL bvec format.

    Arguments:
        filename: path to bvec file

    Output:
        v:        encoding directions
    """

    v = np.loadtxt(filename)
    if v.ndim < 2: # a single encoding direction
        v = v[:, np.newaxis]
    return v

def write_im(filename: str, im: npt.NDArray[np.float64], imref_file: str | None = None) -> None:
    """
    Save image in Nifti format.

    Arguments:
        filename:   path to nifti file
        im:         image to save
        imref_file: (optional) path to nifti file from which header info is obtained
    """
    
    if imref_file is not None:
        nii_ref = nib.load(imref_file)
        nii = nib.Nifti1Image(im, affine=nii_ref.affine, header=nii_ref.header)
    else:
        nii = nib.Nifti1Image(im, affine=np.eye(4))
    nib.save(nii, filename)

def write_bval(filename: str, b: npt.NDArray[np.float64]) -> None:
    """
    Save encoding directions to file in FSL bval format.

    Arguments:
        filename: path to file
        b:        b-values
    """

    np.savetxt(filename, b, fmt='%.1f', newline=' ')

def write_cval(filename: str, c: npt.NDArray[np.float64]) -> None:
    """
    Save encoding directions to file in format similar to FSL bval.

    Arguments:
        filename: path to file
        c:        c-values
    """

    np.savetxt(filename, c, fmt='%.3f', newline=' ')

def write_time(filename: str, t: npt.NDArray[np.float64]) -> None:
    """
    Save time parameter to file in format similar to FSL bval.

    Arguments:
        filename: path to file
        t:        time parameter
    """

    np.savetxt(filename, t, fmt='%.5f', newline=' ')

def write_k(filename: str, k: npt.NDArray[np.float64]) -> None:
    """
    Save k to file in format similar to FSL bval.

    Arguments:
        filename: path to file
        k:        k
    """

    np.savetxt(filename, k, fmt='%.0f', newline=' ')

def write_bvec(filename: str, v: npt.NDArray[np.float64]) -> None:
    """
    Save encoding directions to file in FSL bvec format.

    Arguments:
        filename: path to file
        v:        3 x n array with diffusion encoding directions
    """
    if v.ndim < 2:
        v = v[:, np.newaxis]
    check_2dmatrix(v,v.size//3,'bvec')
    np.savetxt(filename, v, fmt='%.3f')

####################
# Helper functions #
####################
def check_vector(v: npt.NDArray[np.float64], n: int, name: str) -> None:
    if v.ndim != 1:
        raise ValueError(f'The {name} must be ordered as a 1-dimensional vector.')
    if v.size != n:
        raise ValueError(f'The number of {name} must match the size of the fourth dimension of the nifti file.')

def check_2dmatrix(M: npt.NDArray[np.float64], n: int, name: str) -> None:
    if M.ndim != 2:
        raise ValueError(f'The {name} must be ordered as a 2-dimensional vector.')
    if M.shape[1] != n:
        raise ValueError(f'The size of the 2nd dimesion of the {name} must match the size of the fourth dimension of the nifti file.')
    if M.shape[0] != 3:
        raise ValueError(f'The size of the 1st dimesion of the {name} must be three.')

def check_roi(roi: npt.NDArray[np.float64], sz: npt.NDArray[np.int_]) -> None:
    if roi.ndim != 3:
        raise ValueError('The ROI must be a 3D array.')
    if np.any(roi.shape != sz):
        raise ValueError('The shape of the ROI must match that of the first three dimension of the image.')
