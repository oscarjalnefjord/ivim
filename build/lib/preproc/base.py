""" General functions for preprocessing IVIM image data. """

import numpy as np
import nibabel as nib
from ivim.io.base import data_from_file, file_from_data, read_bval, read_bvec, read_cval, read_im, write_bval, write_cval, write_bvec

def average(im_file: str, bval_file: str, outbase: str, cval_file: str | None = None, avg_type: str = 'geo') -> None:
    """
    Average signal across encoding directions.

    Arguments:
        im_file:   path to nifti image file
        bval_file: path to .bval file
        outbase:   basis for output filenames, i.e. filename without file extension to which .nii.gz, .bval, etc. is added
        cval_file: (optional) path to .cval file
        avg_type:  (optional) kind of averaging: 'geo' for geometric mean or 'arith' for arithmetic mean
    """

    if cval_file is None:
        Y, b = data_from_file(im_file, bval_file)
        c = np.zeros_like(b)
    else:
        Y, b, c = data_from_file(im_file, bval_file, cval_file=cval_file)

    ubs = np.unique(np.stack([b, c], axis=0), axis=1)
    Y_avg = np.zeros((Y.shape[0], ubs.shape[1]))
    for i, (ub, uc) in enumerate(zip(ubs[0, :], ubs[1, :])):
        Y_temp = Y[:, (b==ub) & (c==uc)]
        if Y_temp.ndim > 1:
            if avg_type == 'arith':
                Y_avg[:, i] =  np.mean(Y_temp, axis=1)
            elif avg_type == 'geo':
                mask = np.all(Y_temp > 0, axis=1)
                Y_avg[mask, i] =  np.exp(np.mean(np.log(Y_temp[mask, :]), axis=1))
            else:
                raise ValueError('Given averaging method, ' + avg_type + ', not accepted')
        else:
            Y_avg[:, i] = Y_temp

    file_from_data(outbase + '.nii.gz', Y_avg, imref_file=im_file)
    write_bval(outbase + '.bval', ubs[0, :])
    if not (cval_file is None):
        write_cval(outbase + '.cval', ubs[1, :])

def combine(dwi_files: list, bval_files: list, outbase: str, bvec_files: list | None = None, cval_files: list | None = None, normb: float | None = None, roi_file: str | None = None) -> None :
    """
    Combine a set of dwi nifti files and the corresponding .bval, .bvec and .cval files to single files.

    Arguments:
        dwi_files:  list of paths to nifti image file
        bval_files: list of paths to .bval files
        outbase:    basis for output filenames, i.e. filename without file extension to which .nii.gz, .bval, etc. is added
        bvec_files: (optional) list of paths to .bvec files
        cval_files: (optional) list of paths to .cval files
        normb:      (optional) if set to a value in the data set, each data set is normalized towards the data at this b-value (typically normb = 0)
        roi_file:   (optional) path to nifti file defining a region-of-interest (ROI) used for normalization
    """

    dwis = []
    for dwi_file, bval_file in zip(dwi_files, bval_files):
        dwi = nib.load(dwi_file).get_fdata()
        if dwi.ndim == 3:
            dwi = dwi[..., np.newaxis]
        if normb is not None:
            b = read_bval(bval_file)
            if np.any(b == normb):
                if roi_file is not None:
                    mask = read_im(roi_file).astype(bool)
                    dwi /= np.median(dwi[mask, :][:, b == normb])
                else:
                    dwi /= np.quantile(dwi[..., b == normb], 0.75)
        dwis.append(dwi)
    dwi = np.concatenate(dwis,axis=3)    
    file_from_data(outbase + '.nii.gz', np.reshape(dwi, [-1, dwi.shape[-1]]), imref_file=dwi_files[0])

    bs = []
    for bval_file in bval_files:
        bs.append(read_bval(bval_file))
    b = np.concatenate(bs, axis=0)
    write_bval(outbase + '.bval', b)

    if bvec_files is not None:
        vs = []
        for bvec_file in bvec_files:
            vs.append(read_bvec(bvec_file))
        v = np.concatenate(vs, axis=1)
        write_bvec(outbase + '.bvec', v)

    if cval_files is not None:
        cs = []
        for cval_file in cval_files:
            cs.append(read_cval(cval_file))
        c = np.concatenate(cs, axis=0)
        write_cval(outbase + '.cval', c)

def extract(im_file: str, bval_file: str, outbase: str, bvec_file: str | None = None, b_ex: float = 0, cval_file: str | None = None, c_ex: float | None = None):
    """
    Extract data and b-values (optionally also c-values) equal to specific b-value (and optionally c-value).

    Arguments:
        im_file:   path to nifti image file
        bval_file: path to .bval file
        outbase:   basis for output filenames, i.e. filename without file extension to which .nii.gz, .bval, etc. is added
        bvec_file: (optional) path to .bvec file
        b_ex:      (optional) b-value to extract, i.e. b_ex = 0 will extract data from all frames with b == 0
        cval_file: (optional) path to .cval file
        c_ex:      (optional) c-value to extract in addition to condition given by b_ex, i.e. both b == b_ex and c == c_ex must be fulfilled for extraction
    """

    if (cval_file is None):
        if bvec_file is None:
            Y, b = data_from_file(im_file, bval_file)    
        else:
            Y, b, v = data_from_file(im_file, bval_file, bvec_file)
        c = np.zeros_like(b)
    else:
        if bvec_file is None:
            Y, b, c = data_from_file(im_file, bval_file, cval_file=cval_file)
        else:
            Y, b, v, c = data_from_file(im_file, bval_file, bvec_file=bvec_file, cval_file=cval_file)
    if (c_ex is None) != (cval_file is None):
        raise ValueError('Both or non of cval_file and c_ex must be specified')
    elif c_ex is None:
        c_ex = 0

    mask = (b_ex==b) & (c_ex==c)
    if np.any(mask):
        Y_ex = Y[:, mask]
    else:
        raise ValueError('No data fulfilling the extraction criteria')
    file_from_data(outbase + '.nii.gz', Y_ex, imref_file=im_file)

    write_bval(outbase + '.bval', b[mask])

    if bvec_file is not None:
        write_bvec(outbase + '.bvec', v[:, mask])

    if cval_file is not None:
        write_cval(outbase + '.cval', c[mask])
    