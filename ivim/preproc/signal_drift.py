""" Methods for correcting for signal drift. """

import numpy as np
import json
from ivim.io.base import data_from_file, file_from_data, read_im

def temporal(im_file: str, bval_file: str, outbase: str, roi_file: str | None = None, order: int = 2, ctm: str = 'mean', voxelwise: bool = False) -> None:
    """
    Correct for temporal signal drift by fitting a polynomial in time.

    Arguments:
        im_file:   path to nifti image file
        bval_file: path to .bval file
        outbase:   basis for output filenames, i.e. filename without file extension to which .nii.gz, .bval, etc. is added
        roi_file:  (optional) path to nifti file defining a region-of-interest (ROI) in which the correction is calculated and applied
        order:     (optional) order of estimated polynomial (1 or 2)
        ctm:       (optional) central tendency measure used to summarize the signal in the ROI, 'mean' or 'median'
        voxelwise: (optional) if True, one polynomial is fitted for each voxel, else one polynomial for the mean/median signal in the ROI

    The correction follows the algorithm described by Vos et al. "The Importance of 
    Correcting for Signal Drift in Diffusion MRI"
    https://doi.org/10.1002/mrm.26124
    """

    def _vos_fwd(d, n, s0):
        if d.ndim == 2:  # Voxelwise
            order = d.shape[1]
            if order == 2:
                d1, d2 = d[:, 0][:, np.newaxis], d[:, 1][:, np.newaxis]
            elif order == 1:
                pass
            else:
                raise ValueError('d must be of size 1 or 2')
            s0 = s0[:, np.newaxis]
        elif d.ndim == 1:  # Global
            order = d.size
            if order == 2:
                d1,d2 = d[0], d[1]
            elif order == 1:
                pass
            else:
                raise ValueError('d must be of size 1 or 2')
        else:
            raise ValueError('d can only have 1 or 2 dimensions')
        if order == 2:
            return d1*n**2 + d2*n + s0
        elif order == 1:
            return d*n + s0


    def _vos_inv(Y, d, n, s0, scale=100, trim=True):
        Ycorr = Y*scale/_vos_fwd(d, n, s0)
        if trim:
            q1 = np.quantile(Ycorr, 0.01)
            q99 = np.quantile(Ycorr, 0.99)
            Ycorr[Ycorr > q99] = q99
            Ycorr[Ycorr < q1] = q1
        return Ycorr

    Y, b = data_from_file(im_file, bval_file, roi_file=roi_file)
    if voxelwise:
        y0 = Y[:, b==0].T
    else:
        if ctm == 'mean':
            y0 = np.mean(Y[:, b==0],axis=0)
        elif ctm == 'median':
            y0 = np.median(Y[:, b==0],axis=0)
        else:
            raise ValueError(f'Valid values of ctm are: "mean" and "median", but "{ctm}" was given.')
    
    n = (b==0).nonzero()[0]
    if order == 2:
        N = np.vstack((n**2, n, np.ones_like(n))).T
    elif order == 1:
        N = np.vstack((n, np.ones_like(n))).T
    else:
        raise ValueError('order must be 1 or 2.')
    p = np.linalg.lstsq(N, y0, rcond=None)[0]
    if voxelwise:
        d = p[:-1, :].T
        s0 = p[-1, :].T
    else:
        d = p[:-1]
        s0 = np.array(p[-1])

    Ycorr = _vos_inv(Y, d, np.arange(b.size), s0)

    if roi_file is not None:
        roi = read_im(roi_file).astype(bool)
    else:
        roi = None
    file_from_data(outbase + '_corr.nii.gz', Ycorr, roi=roi, imref_file=im_file)
    if voxelwise:
        file_from_data(outbase + '_d.nii.gz', d, roi=roi, imref_file=im_file)
        file_from_data(outbase + '_s0.nii.gz', s0, roi=roi, imref_file=im_file)
        Yhat = 1 / _vos_inv(np.ones_like(Y), d, np.arange(b.size), s0)
        file_from_data(outbase + '_corrfield.nii.gz', (Yhat-Yhat[:, 0][:, np.newaxis])/Yhat[:, 0][:, np.newaxis], roi=roi, imref_file=im_file)
    else:
        with open(outbase + '.json', 'w') as f:
            json.dump({'d':[float(di) for di in d], 's0':float(s0)}, f)

def spatiotemporal(im_file: str, bval_file: str, outbase: str, roi_file: str | None = None, order: int = 2) -> None:
    """
    Correct for spatiotemporal signal drift by fitting a polynomial in space and time.

    Arguments:
    im_file   -- path to nifti image file
    bval_file -- path to .bval file
    outbase   -- basis for output filenames, i.e. filename without file extension to which .nii.gz, .bval, etc. is added
    roi_file  -- (optional) path to nifti file defining a region-of-interest (ROI) in which the correction is calculated and applied
    order     -- (optional) order of estimated polynomial (1 or 2)

    The correction follows the algorithm described by Hansen et al. in "Characterization 
    and Correlation of Signal Drift in Diffusion Weighted MRI" 
    https://doi.org/10.1016/j.mri.2018.11.009
    """

    def _hansen_A(mask, n, order):
        if not order in [1, 2]:
            raise ValueError('order must be 1 or 2')
        sz = mask.shape
        coords = np.mgrid[0:sz[0], 0:sz[1], 0:sz[2]]
        x, y, z = [coords[i][mask]/sz[i] for i in range(3)]
        xyz = np.vstack((np.ones_like(x), x, y, z, x*y, x*z, y*z, x*y*z)).T
        if order == 2:
            xyz = np.vstack((xyz.T,
                            x**2, y**2, z**2,
                            x*y**2, x*z**2,
                            x*y**2*z, x*y*z**2, x*y**2*z**2,
                            x**2*y, x**2*z, x**2*y*z, x**2*y**2, x**2*z**2,
                            x**2*y**2*z, x**2*y*z**2, x**2*y**2*z**2,
                            y**2*z, y*z**2, y**2*z**2)).T

        A = np.vstack((np.repeat(xyz, n.size, axis=0).T, (np.tile(n, x.size)[:, np.newaxis]*np.repeat(xyz, n.size, axis=0)).T)).T
        if order == 2:
            A = np.vstack((A.T, (np.tile(n**2, x.size)[:, np.newaxis]*np.repeat(xyz, n.size, axis=0)).T)).T

        return A

    def _hansen_fwd(p, A):
        return A@p

    def _hansen_inv(Y, p, A, scale=100):
        Ycorr = Y*scale/_hansen_fwd(p, A).reshape(Y.shape)
        return Ycorr

    Y, b = data_from_file(im_file, bval_file, roi_file=roi_file)

    if roi_file is not None:
        mask = read_im(roi_file).astype(bool)
    else:
        raise ValueError('A mask is required for spatiotemporal drift correction.')

    n = (b==0).nonzero()[0]
    zeromask = ~np.all(Y[:, n]==0, axis=1)
    mask[mask] = zeromask
    Y = Y[zeromask, :]
    Y = Y/np.mean(Y[:, n], axis=1)[:, np.newaxis] # Remove underlying morphological variations.
    y = Y[:, n].flatten()    
    A = _hansen_A(mask, n, order)

    X = np.vstack((np.ones(A.shape[0]), A.T/b.size)).T
    m = np.mean(X, axis=0)
    C = X.T@X / (Y.shape[0]-1)
    h = 1/Y.shape[0] + np.einsum('ij,ji->i', (X-m[np.newaxis, :]), np.linalg.inv(C)@(X-m[np.newaxis, :]).T)/(Y.shape[0]-1)     # https://en.wikipedia.org/wiki/Mahalanobis_distance#Relationship_to_leverage
    w = np.ones_like(h)
    Sold = np.inf
    for i in range(15):
        # Find solution to weighted least squares problem.
        p = np.linalg.lstsq(np.sqrt(w[:, np.newaxis])*A, np.sqrt(w)*y, rcond=None)[0]

        # Update weights.
        r = np.abs(y-_hansen_fwd(p, A))
        S = np.median(np.sort(r)[(p.size-1):]) / 0.6745  # Remove (p-1) smallest residuals.
        u = r / (4.685*S*np.sqrt(1-h))
        w[np.abs(u)>1] = 0
        w[np.abs(u)<=1] = (1 - u[np.abs(u)<=1]**2)**2
        print(f'Iteration {i}, S = {S}')
        if np.abs(S-Sold)/S < 1e-3:
            break
        Sold = S

    Acorr = _hansen_A(mask, np.arange(Y.shape[1]), order)
    Ycorr = _hansen_inv(Y, p, Acorr)

    file_from_data(outbase + '_corr.nii.gz', Ycorr, roi=mask, imref_file=im_file)
    Yhat = 1/_hansen_inv(np.ones_like(Y), p, Acorr)
    file_from_data(outbase + '_corrfield.nii.gz', (Yhat-Yhat[:, 0][:, np.newaxis])/Yhat[:, 0][:, np.newaxis], roi=mask, imref_file=im_file)
    file_from_data(outbase + '_resid.nii.gz', Y[:, n] - _hansen_fwd(p, A).reshape(Y[:, n].shape), roi=mask, imref_file=im_file)
    with open(outbase + '.json', 'w') as f:
        json.dump({'p':list(p)}, f)