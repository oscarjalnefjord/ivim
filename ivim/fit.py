""" Functions for IVIM parameter estimation. """

import numpy as np
import numpy.typing as npt
from scipy.optimize import curve_fit
from ivim.models import sIVIM, diffusive, ballistic, intermediate, NO_REGIME, DIFFUSIVE_REGIME, BALLISTIC_REGIME, INTERMEDIATE_REGIME, check_regime
from ivim.models import monoexp as monoexp_model
from ivim.models import kurtosis as kurtosis_model
from ivim.constants import Db
from ivim.seq.sde import MONOPOLAR, BIPOLAR
from ivim.misc import halfSampleMode
from ivim.io.base import data_from_file, file_from_data, read_im, read_time, read_k

def nlls(im_file: str, bval_file: str, regime: str, roi_file: str | None = None, outbase: str | None = None, verbose: bool = False, fitK: bool = False, cval_file: str | None = None) -> None:
    """
    NLLS fitting of the IVIM model different regimes.

    Arguments:
        im_file:   path to nifti image file
        bval_file: path to .bval file
        regime:    IVIM regime to model: no (= sIVIM), diffusive (long encoding time) or ballistic (short encoding time)
        roi_file:  (optional) path to nifti file defining a region-of-interest (ROI) from with data is extracted
        outbase:   (optional) basis for output filenames to which e.g. '_D.nii.gz' is added 
        verbose:   (optional) if True, diagnostics during fitting is printet to terminal
        fitK:      (optional) if True, the kurtosis signal representation is used instead of a monoexponential one in the first step
        cval_file: (optional) path to .cval file
    """

    check_regime(regime)

    if regime == BALLISTIC_REGIME:
        Y, b, c = data_from_file(im_file, bval_file, cval_file=cval_file, roi_file=roi_file)
    else:
        Y, b = data_from_file(im_file, bval_file, roi_file = roi_file)

    if regime == DIFFUSIVE_REGIME:
        if fitK:
            def fn(X, D, f, Dstar, S0, K):
                b = X
                return diffusive(b, D, f, Dstar, S0, K).squeeze()
        else:
            def fn(X, D, f, Dstar, S0):
                b = X
                return diffusive(b, D, f, Dstar, S0).squeeze()
    elif regime == BALLISTIC_REGIME:
        if fitK:
            def fn(X, D, f, vd, S0, K):
                b = X[:, 0]
                c = X[:, 1]
                return ballistic(b, c, D, f, vd, S0, K).squeeze()
        else:
            def fn(X, D, f, vd, S0):
                b = X[:, 0]
                c = X[:, 1]
                return ballistic(b, c, D, f, vd, S0).squeeze()
    else:
        if fitK:
            def fn(X, D, f, S0, K):    
                b = X
                return sIVIM(b, D, f, S0, K).squeeze()
        else:
            def fn(X, D, f, S0):    
                b = X
                return sIVIM(b, D, f, S0).squeeze()

    if regime == BALLISTIC_REGIME:
        X = np.stack((b, c), axis=1)
    else:
        X = b
    mask = valid_signal(Y)
    npars = 4 + fitK - (regime == NO_REGIME)
    P = np.full((mask.size, npars), np.nan)

    for i, y in enumerate(Y):
        p0 = [1e-3, 0.1, np.max(y)] # start values
        bounds = np.array([[0, 0, 0], [3e-3, 1, 2*p0[2]]]) # fitting bounds
        if regime == DIFFUSIVE_REGIME:
            p0.insert(2, 10e-3)
            bounds = np.insert(bounds, 2, [0, 1], axis = 1)
        if regime == BALLISTIC_REGIME:
            p0.insert(2, 2.0)
            bounds = np.insert(bounds, 2, [0, 5], axis = 1)
        if fitK:
            p0.append(1.0)
            bounds = np.hstack((bounds, np.array([0, 5])[:, np.newaxis]))

        if verbose:
            if i%1000 == 0:
                print(f'Fitting voxel {i}/{Y.shape[0]}')
            elif i == (Y.shape[0] - 1):
                print('Fitting done!')

        if mask[i]:
            try:
                P[i, :],_ = curve_fit(fn, X, y, p0=p0, bounds=bounds, x_scale=p0)
            except:
                P[i, :] = np.full(npars, np.nan)
    
    pars = {'D': P[:, 0], 'f': P[:, 1]}
    if regime == DIFFUSIVE_REGIME:
        pars['Dstar'] = P[:, 2]
        idxS0 = 3
        idxK = 4
    elif regime == BALLISTIC_REGIME:
        pars['vd'] = P[:, 2]
        idxS0 = 3
        idxK = 4
    else:
        idxS0 = 2
        idxK = 3
    pars['S0'] = P[:, idxS0]
    if fitK:
        pars['K'] = P[:, idxK]
    save_parmaps(pars, outbase, im_file, roi_file)

def seg(im_file: str, bval_file: str, regime: str, bthr: float = 200, roi_file: str | None = None, outbase: str | None = None, verbose: bool = False, fitK: bool = False, cval_file: str | None = None, cthr: float = 0) -> None:
    """
    Segmented fitting of the IVIM model different regimes.

    Arguments:
        im_file:   path to nifti image file
        bval_file: path to .bval file
        regime:    IVIM regime to model: no (= sIVIM), diffusive (long encoding time) or ballistic (short encoding time)
        bthr:      (optional) threshold b-value from which signal is included in first fitting step
        roi_file:  (optional) path to nifti file defining a region-of-interest (ROI) from with data is extracted
        outbase:   (optional) basis for output filenames to which e.g. '_D.nii.gz' is added 
        verbose:   (optional) if True, diagnostics during fitting is printet to terminal
        fitK:      (optional) if True, the kurtosis signal representation is used instead of a monoexponential one in the first step
        cval_file: (optional) path to .cval file
        cthr:      (optional) threshold c-value from above which signal is included in first fitting step
    """

    def _monoexp(Y: npt.NDArray[np.float64], b: npt.NDArray[np.float64], lim: list = [0, 3e-3], validate: bool = True, verbose: bool = False) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """ Estimate D and A (y axis intercept) from a monoexponential model. """

        if validate:
            mask = valid_signal(Y) # Avoids signal with obvious errors (non-positive, nan, inf)
        else:
            mask = np.full(Y.shape[0], True)
            
        D = np.full(mask.shape, np.nan)
        A = np.full(mask.shape, np.nan)

        if b.size == 2:
            if b[1] == b[0]:
                raise ZeroDivisionError("Two b-values can't be equal or both zero.")
            D[mask] = (np.log(Y[mask, 0]) - np.log(Y[mask, 1])) / (b[1] - b[0])
            
            D[mask & (D<lim[0])] = lim[0]
            D[mask & (D>lim[1])] = lim[1]

            A[mask] = Y[mask, 0] * np.exp(b[0]*D[mask])
        elif b.size > 2:
            D[mask], A[mask] = _optimizeD(Y[mask, :], b, lim, disp_prog = verbose)
        else:
            raise ValueError('Too few b-values.')

        return D, A


    def _kurtosis(Y: npt.NDArray[np.float64], b: npt.NDArray[np.float64], lim: list = [0, 3e-3]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """ Calculate D, K and A (y axis intercept) from the kurtosis signal representation. """

        mask = valid_signal(Y)
        
        D = np.full(mask.shape, np.nan)
        K = np.full(mask.shape, np.nan)
        A = np.full(mask.shape, np.nan)
        if b.size == 3:
            # Analytic solution
            D[mask] = (
                    (np.log(Y[mask,0])-np.log(Y[mask,1])) / (b[1]-b[0])
                    - (b[1]+b[0]) / (b[2]-b[1]) * (
                        (np.log(Y[mask, 0])-np.log(Y[mask, 2])) / (b[2]-b[0])
                        - (np.log(Y[mask, 0])-np.log(Y[mask, 1])) / (b[1]-b[0])
                        )
                    )  
            K[mask] = 6./D[mask]**2 / (b[2]-b[1]) * (
                    (np.log(Y[mask,0])-np.log(Y[mask,1])) / (b[1]-b[0])
                    - (np.log(Y[mask,0])-np.log(Y[mask,2])) / (b[2]-b[0])
                    )

            # Use monoexponential fit when quadratic fit is not supported.
            D[mask & (K<0)], A[mask & (K<0)] = _monoexp(Y[mask & (K<0), :], b)
            K[mask & (K<0)] = 0

            # Apply limits.
            D[mask & (D<lim[0])] = lim[0]
            D[mask & (D>lim[1])] = lim[1]

            # Extrapolate to intercept.
            A[mask] = Y[mask, 0] * np.exp(b[0] * D[mask] - b[0]**2 * D[mask]**2 * 
                                        K[mask]/6)

        elif b.size > 3:
            x = np.column_stack([np.ones(b.size), b, b**2])
            for i, y in enumerate(Y):
                C2 = np.diag(y**2)  # For heteroscedasticity correction
                tmp = x.T @ C2 @ x
                if mask[i]:
                    m = np.linalg.lstsq(tmp, x.T@C2@np.real(np.log(y)), rcond = None)[0]
                    A[i] = np.exp(m[0])
                    D[i] = -m[1]
                    K[i] = m[2] * 6/D[i]**2 # m[2] = 1/6 * D**2 * K
        else: 
            raise ValueError('Too few b-values')
            
        return D, K, A


    def _get_S0(Y: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """ Return the signal values at b = 0."""
        return np.mean(Y[:, b==0], axis=1)


    def _f_from_intercept(A: npt.NDArray[np.float64], S0: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """ Calculate f from S(b=0) and extrapolated y axis intercept A."""
        f = 1 - A/S0
        f[f<0] = 0
        return f
    
    def _optimizeD(Y: npt.NDArray[np.float64], b: npt.NDArray[np.float64], lim: list, optlim: float = 1e-6, disp_prog: bool = False) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """ Specfically tailored function for finding the least squares solution of monoexponenital fit. """

        n = Y.shape[0]
        D = np.zeros(n)
        yb = Y * np.tile(b, (n, 1))  # Precalculate for speed.
        
        ##############################################
        # Check if a minimum is within the interval. #
        ##############################################
        # Check that all diff < 0 for Dlow.
        Dlow = lim[0] * np.ones(n)
        difflow,_ = _Ddiff(Y, yb, b, Dlow)
        low_check = difflow < 0 # difflow must be < 0 if the optimum is within the interval.
        
        # Check that all diff > 0 for Dhigh
        Dhigh = lim[1] * np.ones(n)
        diffhigh,_ = _Ddiff(Y, yb, b, Dhigh)
        high_check = diffhigh > 0  # diffhigh must be > 0 if the optimum is within the interval.
        
        # Set parameter value with optimum out of bounds.
        D[~low_check] = lim[0]  # difflow > 0 means that the mimimum has been passed .
        D[~high_check] = lim[1]  # diffhigh < 0 means that the minium is beyond the interval.
        
        # Only the voxels with a possible minimum should be estimated.
        mask = low_check & high_check
        if disp_prog:
            print(f'Discarding {np.count_nonzero(~mask)} voxels due to parameters out of bounds.')

        # Allocate all variables.
        D_lin = np.zeros(n)
        diff_lin = np.zeros(n)
        D_mid = np.zeros(n)
        diff_mid = np.zeros(n)
        ratio_lin = np.zeros(n)
        ratio_mid = np.zeros(n)

        ##########################################################
        # Iterative method for finding the point where diff = 0. #
        ##########################################################
        k = 0
        while np.any(mask):  # Continue if there are voxels left to optimize.
            # Assume diff is linear within the search interval [Dlow Dhigh].
            D_lin[mask] = Dlow[mask] - difflow[mask] * (Dhigh[mask]-Dlow[mask]) / (diffhigh[mask]-difflow[mask])
            # Calculate diff in the point of intersection given by the previous expression.
            diff_lin[mask], ratio_lin[mask] = _Ddiff(Y[mask, :], yb[mask, :], b, D_lin[mask])
        
            # As a potential speed up, the mean of Dlow and Dhigh is also calculated.
            D_mid[mask] = (Dlow[mask]+Dhigh[mask]) / 2
            diff_mid[mask], ratio_mid[mask] = _Ddiff(Y[mask, :], yb[mask, :], b, D_mid[mask])
            
            # If diff < 0, then the point of intersection or mean is used as the
            # new Dlow. Only voxels with diff < 0 are updated at this step. Linear
            # interpolation or the mean is used depending of which method that
            # gives the smallest diff.
            updatelow_lin = (diff_lin<0) & ((diff_mid>0) | ((D_lin>D_mid) & (diff_mid<0)))
            updatelow_mid = (diff_mid<0) & ((diff_lin>0) | ((D_mid>D_lin) & (diff_lin<0)))
            Dlow[updatelow_lin] = D_lin[updatelow_lin]
            Dlow[updatelow_mid] = D_mid[updatelow_mid]
            
            # If diff > 0, then the point of intersection or mean is used as the
            # new Dhigh. Only voxels with diff > 0 are updated at this step. 
            # Linear interpolation or the mean is used depending of which method 
            # that gives the smallest diff.
            updatehigh_lin = (diff_lin>0) & ((diff_mid<0) | ((D_lin<D_mid) & (diff_mid>0)))
            updatehigh_mid = (diff_mid>0) & ((diff_lin<0) | ((D_mid<D_lin) & (diff_lin>0)))
            Dhigh[updatehigh_lin] = D_lin[updatehigh_lin]
            Dhigh[updatehigh_mid] = D_mid[updatehigh_mid]
            
            # Update the mask to exclude voxels that fulfills the optimization
            # limit from the mask.
            opt_lin = np.abs(1-ratio_lin) < optlim
            opt_mid = np.abs(1-ratio_mid) < optlim
            
            D[opt_lin] = D_lin[opt_lin]
            D[opt_mid] = D_mid[opt_mid]  
            # Not optimal if both D_lin and D_mean fulfills the optimization limit,
            # but has a small impact on the result as long as optlim is small.
            
            # Update the mask.
            mask = mask & (~(opt_lin|opt_mid))
            
            # Calculate diff for the new bounds.
            if np.any(mask):
                difflow[mask],_ = _Ddiff(Y[mask, :], yb[mask, :], b, Dlow[mask])
                diffhigh[mask],_ = _Ddiff(Y[mask, :], yb[mask, :], b, Dhigh[mask])
            
            k += 1
            if disp_prog:
                print(f'Iteration {k}: {np.count_nonzero(mask)} voxels left.')

        A = np.sum(Y*np.exp(-np.outer(D, b)), axis=1) / np.sum(np.exp(-2*np.outer(b, D)), axis=0)

        return D, A


    def _Ddiff(Y: npt.NDArray[np.float64], yb: npt.NDArray[np.float64], b: npt.NDArray[np.float64], D: npt.NDArray[np.float64]):
        """
        Return the difference between q1 = e^(-2*b*D)*yb*e^(-b*D) and 
        q2 = Y*e^(-b*D)*b*e^(-2*b*D) summed over b as well as the ratio q1/q2
        summed over b, setting divisions by zero as infinite.
        """
        q1 = np.sum(np.exp(-2*np.outer(b, D)), axis=0) * np.sum(yb*np.exp(-np.outer(D, b)), axis=1)
        q2 = np.sum(Y*np.exp(-np.outer(D, b)), axis=1) * np.sum(b[:, np.newaxis]*np.exp(-2*np.outer(b, D)), axis=0)
        diff = q1 - q2
        ratio = np.full(q1.shape, np.inf)
        ratio[q2!=0] = q1[q2!=0] / q2[q2!=0]
        return diff, ratio
    
    check_regime(regime)

    if regime == BALLISTIC_REGIME:
        Y, b, c = data_from_file(im_file, bval_file, cval_file=cval_file, roi_file=roi_file)
        bmask = (b >= bthr) & (np.abs(c) > cthr)
    else:
        Y, b = data_from_file(im_file, bval_file, roi_file = roi_file)
        bmask = b >= bthr

    if fitK:
        D, K, A = _kurtosis(Y[:, bmask], b[bmask])
        Ysub = (Y - A[:, np.newaxis]*kurtosis_model(b, D, K))    # Remove signal related to diffusion
    else:
        D, A = _monoexp(Y[:, bmask], b[bmask], verbose=verbose)
        Ysub = (Y - A[:, np.newaxis]*monoexp_model(b, D))        # Remove signal related to diffusion
    if regime == BALLISTIC_REGIME:
        Ysub /= monoexp_model(b, Db)

    if regime == DIFFUSIVE_REGIME:
        Dstar, Astar = _monoexp(Ysub, b, lim=[3e-3, 0.1], validate = False, verbose = verbose)
        S0 = A + Astar
    elif regime == BALLISTIC_REGIME:
        vd2, Astar = _monoexp(Ysub, c**2, lim=[0, 25], validate=False, verbose=verbose)
        S0 = A + Astar
    else:
        S0 = _get_S0(Y, b)
    
    f = _f_from_intercept(A, S0)

    pars = {'D': D, 'f': f}
    if regime == DIFFUSIVE_REGIME:
        pars['Dstar'] = Dstar
    elif regime == BALLISTIC_REGIME:
        pars['vd'] = np.sqrt(vd2)
    pars['S0'] = S0
    if fitK:
        pars['K'] = K
    save_parmaps(pars, outbase, im_file, roi_file)

def bayes(im_file: str, bval_file: str, regime: str, roi_file: str | None = None, outbase: str | None = None, verbose: bool = False, fitK: bool = False, 
          spatial_prior: bool = False, n: int = 2000, burns: int = 1000, ctm: str = 'mean', output_std: bool = False,
          cval_file: str | None = None, seq: str = MONOPOLAR, delta_file: str | None = None, Delta_file: str | None = None, T_file: str | None = None, k_file: str | None = None):
    """
    Bayesian fitting of the IVIM model in different regimes

    Arguments:
        im_file:       path to nifti image file
        bval_file:     path to .bval file
        regime:        IVIM regime to model: no (= sIVIM), diffusive (long encoding time) or ballistic (short encoding time)
        roi_file:      (optional) path to nifti file defining a region-of-interest (ROI) from with data is extracted
        outbase:       (optional) basis for output filenames to which e.g. '_D.nii.gz' is added 
        verbose:       (optional) if True, diagnostics during fitting is printet to terminal
        fitK:          (optional) if True, the kurtosis signal representation is used instead of a monoexponential one in the first step
        spatial_prior: (optional) if True, a spatial prior enforcing similary between 4-neighbours is applied 
        n:             (optional) number of Markov Chain Monte Carlo (MCMC) iterations
        burns:         (optional) number of MCMC iterations before sampling
        ctm:           (optional) central tendency measure (mean or mode) used to summarize the posterior parameter distributions
        cval_file:     (optional) path to .cval file
    """

    def _estimation(fn, Y, X, P0, lims, n=500, burns=500, ctm='mean', spatial_prior = False, roi=None, verbose=False):
        """
        Perform a Bayesian parameter estimation and return posterior parameter
        values and their standard deviations.
        """

        ##################
        # Error handling #
        ##################
        n = int(n)
        burns = int(burns)

        #########################
        # Parameter preparation #
        #########################
        mask = valid_signal(Y)
        V = np.sum(mask)  # Y.shape[0]
        pars = P0.shape[1]

        if ctm == 'mean':
            meanonly = True
        elif not ((ctm == 'median') or (ctm == 'mode')):
            raise ValueError(f'Unknown central tendency measure "{ctm}".')
        else:
            meanonly = False

        if meanonly:
            thetasum = np.zeros((V, pars))
            theta2sum = np.zeros((V, pars))

        # Burn-in parameters
        burnUpdateInterval = 100
        burnUpdateFraction = 1

        ########################
        # Parameter estimation #
        ########################

        # Initialize parameter vector.
        if meanonly:
            theta = np.zeros((V, pars, 2))
        else:
            theta = np.zeros((V, pars, n))

        theta[..., 0] = P0[mask, :]

        if spatial_prior:
            if roi is None:
                raise ValueError('A mask is required for the spatial prior.')
            else:
                roi = roi.astype(bool)
            roi[roi] &= mask
            neighbour_mask = neighbours(roi)

        # Step length parameter
        w = theta[..., 0]/10
        N = np.zeros_like(w)  # Number of accepted samples

        # Prior from previous iteration
        prior_old = np.ones_like(P0[mask, :])
        
        # Iterate for j = 0, 1, 2,..., n-1.
        for j in range(n + burns):
            # Initialize theta(j).
            if j > 0:
                if meanonly or (j < burns+1):
                    theta[:, :, 1] = theta[:, :, 0]
                    thetanew = theta[:, :, 1]
                    thetaold = theta[:, :, 0]
                else:
                    theta[:, :, j - burns] = theta[:, :, j - 1 - burns]
                    thetanew = theta[:, :, j - burns]
                    thetaold = theta[:, :, j - 1 - burns]
            else:
                # First iteration
                thetanew = theta[:, :, 0]
                thetaold = theta[:, :, 0]

            # Sample each parameter.
            for k in range(pars):
                # Take a step in parameter space.
                if j > 0:
                    thetanew[:, k] = thetanew[:, k] + np.random.randn(V)*w[:, k]

                # Calculate prior probability.
                prior = ((thetanew[:, k] >= lims[0, k]) & (thetanew[:, k] <= lims[1, k])).astype(float)

                if spatial_prior:
                    theta_neighbours = np.concatenate((thetanew[:,k],np.full(1,np.nan)),axis=0)[neighbour_mask]
                    prior *= np.exp(-np.nansum(np.abs((theta_neighbours-thetanew[:,k][:,np.newaxis])),axis=1)/P0[mask,k])

                # Calculate posterior probability ratio.
                post_ratio = np.zeros_like(prior)
                ssq_new = np.sum((Y[mask, :] - fn(X, thetanew))**2, axis=1)
                ssq_old = np.sum((Y[mask, :] - fn(X, thetaold))**2, axis=1)
                nonzero = (ssq_old > 0) #& (prior_old[:,k]>0)
                post_ratio[nonzero] = ((ssq_new[nonzero] / ssq_old[nonzero])**(-X.shape[0]/2) 
                                        * prior[nonzero]/prior_old[nonzero,k])

                # Evaluate parameter step.
                sample_ok = np.random.rand(V) < post_ratio
                thetanew[~sample_ok, k] = thetaold[~sample_ok, k]  # Reject samples.
                N[:, k] = N[:, k] + sample_ok
                prior_old[sample_ok, k] = prior[sample_ok]
            
            # Prepare for next iteration.
            if meanonly or (j < burns):
                theta[:, :, 0] = thetanew
            else:
                theta[:, :, j-burns] = thetanew
            
            # Save parameter value after burn-in phase.
            if meanonly and j > burns:
                thetasum = thetasum + thetanew
                theta2sum = theta2sum + thetanew**2
            
            # Adapt step length.
            if (j <= burns*burnUpdateFraction) and ((j+1)%burnUpdateInterval == 0):
                w = w * (burnUpdateInterval+1) / (2*((burnUpdateInterval+1)-N))
                N[...] = 0

            # Give update.
            if verbose and ((j%100 == 0) or (j == (n+burns-1))):
                print(f'Iteration {j+1}/{n+burns}')
        
        # Saves distribution measures.
        P = np.full(P0.shape, np.nan)
        std = np.full(P0.shape, np.nan)
        if meanonly:
            P[mask, :] = thetasum/n                              # Mean
            std[mask, :] = np.sqrt(theta2sum/n-(thetasum/n)**2)  # Standard deviation
        else:
            for k in range(P.shape[1]):
                if ctm == 'median':
                    P[mask, k] = np.median(theta[:, k, :], axis=1)
                elif ctm == 'mode':
                    P[mask, k] = halfSampleMode(theta[:, k, :])
                std[mask, k] = np.std(theta[:, k, :], axis=1)

        return P,std
    
    check_regime(regime)

    if regime == BALLISTIC_REGIME:
        Y, b, c = data_from_file(im_file, bval_file, cval_file=cval_file, roi_file=roi_file)
    else:
        Y, b = data_from_file(im_file, bval_file, roi_file = roi_file)
    if regime == INTERMEDIATE_REGIME:
        delta = read_time(delta_file)
        Delta = read_time(Delta_file)
        if seq == BIPOLAR:
            T = read_time(T_file)
            k = read_k(k_file)
            
    if regime == DIFFUSIVE_REGIME:
        if fitK:
            def fn(X, P):
                D, f, Dstar, S0, K = P[:, 0], P[:, 1], P[:, 2], P[:, 3], P[:, 4]
                b = X
                return diffusive(b, D, f, Dstar, S0, K)
        else:
            def fn(X, P):
                D, f, Dstar, S0 = P[:, 0], P[:, 1], P[:, 2], P[:, 3]
                b = X
                return diffusive(b, D, f, Dstar, S0)
    elif regime == BALLISTIC_REGIME:
        if fitK:
            def fn(X, P):
                D, f, vd, S0, K = P[:, 0], P[:, 1], P[:, 2], P[:, 3], P[:, 4]
                b, c = X[:, 0], X[:, 1]
                return ballistic(b, c, D, f, vd, S0, K)
        else:
            def fn(X, P):
                D, f, vd, S0 = P[:, 0], P[:, 1], P[:, 2], P[:, 3]
                b, c = X[:, 0], X[:, 1]
                return ballistic(b, c, D, f, vd, S0)
    elif regime == INTERMEDIATE_REGIME:
        if fitK:
            def fn(X, P):
                D, f, v, tau, S0, K = P[:, 0], P[:, 1], P[:, 2], P[:, 3], P[:, 4], P[:, 5]
                if seq == MONOPOLAR:
                    b, delta, Delta = X[:, 0], X[:, 1], X[:, 2]
                    return intermediate(b, delta, Delta, D, f, v, tau, S0, K)
                else:
                    b, delta, Delta, T, k = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
                    return intermediate(b, delta, Delta, D, f, v, tau, S0, K, seq=seq, T=T, k=k)
        else:
            def fn(X, P):
                D, f, v, tau, S0 = P[:, 0], P[:, 1], P[:, 2], P[:, 3], P[:, 4]
                if seq == MONOPOLAR:
                    b, delta, Delta = X[:, 0], X[:, 1], X[:, 2]
                    return intermediate(b, delta, Delta, D, f, v, tau, S0)
                else:
                    b, delta, Delta, T, k = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
                    return intermediate(b, delta, Delta, D, f, v, tau, S0, seq=seq, T=T, k=k)
    else:
        if fitK:
            def fn(X, P):
                D, f, S0, K = P[:, 0], P[:, 1], P[:, 2], P[:, 3]
                b = X
                return sIVIM(b, D, f, S0, K)
        else:
            def fn(X, P):
                D, f, S0 = P[:, 0], P[:, 1], P[:, 2]
                b = X
                return sIVIM(b, D, f, S0)
    
    npars = 4 + fitK - (regime == NO_REGIME) + (regime == INTERMEDIATE_REGIME)
    P0 = np.zeros((Y.shape[0], npars))
    P0[:, 0] = 1e-3
    P0[:, 1] = 0.1
    lims = np.array([[0, 0, 0], [3e-3, 1, 2*np.max(Y)]])
    if regime == DIFFUSIVE_REGIME:
        P0[:, 2] = 10e-3
        lims = np.insert(lims, 2, [0, 1.0], axis = 1)
        idxS0 = 3
        idxK = 4
    elif regime == BALLISTIC_REGIME:
        P0[:, 2] = 2.0
        lims = np.insert(lims, 2, [0, 5.0], axis = 1)
        idxS0 = 3
        idxK = 4
    elif regime == INTERMEDIATE_REGIME:
        P0[:, 2] = 2.0
        P0[:, 3] = 0.1
        lims = np.insert(lims, 2, [0, 5.0], axis = 1)
        lims = np.insert(lims, 3, [0.001, 1.0], axis = 1)
        idxS0 = 4
        idxK = 5
    else:
        idxS0 = 2
        idxK = 3
    P0[:, idxS0] = np.mean(Y[:, b==0], axis = 1)
    if fitK:
        P0[:, idxK] = 1.0
        lims = np.hstack((lims, np.array([0, 5])[:, np.newaxis]))

    if regime == BALLISTIC_REGIME:
        X = np.stack((b, c), axis=1)
    elif regime == INTERMEDIATE_REGIME:
        if seq == MONOPOLAR:
            X = np.stack((b, delta, Delta), axis=1)
        else:
            X = np.stack((b, delta, Delta, T, k), axis=1)
    else:
        X = b
    P,Pstd = _estimation(fn, Y, X, P0, lims, n=n, burns=burns, ctm=ctm, spatial_prior=spatial_prior, roi=read_im(roi_file), verbose=verbose)
  
    pars = {'D': P[:, 0], 'f': P[:, 1], 'S0': P[:, idxS0]}
    std = {'D': Pstd[:, 0], 'f': Pstd[:, 1], 'S0': Pstd[:, idxS0]}
    if regime == DIFFUSIVE_REGIME:
        pars['Dstar'] = P[:, 2]
        std['Dstar'] = Pstd[:, 2]
    if regime == BALLISTIC_REGIME:
        pars['vd'] = P[:, 2]
        std['vd'] = P[:, 2]
    if regime == INTERMEDIATE_REGIME:
        pars['v'] = P[:, 2]
        pars['tau'] = P[:, 3]
        std['v'] = Pstd[:, 2]
        std['tau'] = Pstd[:, 3]
    if fitK:
        pars['K'] = P[:, idxK]
        std['K'] = Pstd[:, idxK]
    save_parmaps(pars, outbase, im_file, roi_file)
    if output_std:
        save_parmaps(std, outbase+'-std', im_file, roi_file)

def save_parmaps(pars: dict, outbase: str | None = None, imref_file: str | None = None, roi_file: str | None = None) -> None:
    """
    Save IVIM parameter data (vector format) as nifti images

    Arguments:
    pars       -- parameter data in format {par_name: par_value}, e.g. {'D': D, 'f': f}
    outbase    -- (optional) basis for output filenames to which e.g. '_D.nii.gz' is added 
    imref_file -- (optional) path to nifti file from which header info is obtained
    roi        -- (optional) region-of-interest from which data is assumed to originate. The number of True elements must match the size of the parameter vector

    Note! A subset of the optional arguments must be given:
    - if outbase is not set, it is derived from imref_file
    - if imref_file is not set, the image size is derived from roi_file
    i.e. valid argument combinations are:
    - outbase + imref_file + roi_file
    - outbase + roi_file
    - outbase + imref_file
    - imref_file + roi_file
    - imref_file
    """
    
    if imref_file == None:
        if outbase == None:
            raise ValueError('Either outbase or imref_file must be set.')
        # Remaining invalid combinations are handled by file_from_data
    else:
        if outbase == None:
            outbase = imref_file.split('.')[0]
    
    for parname, par in pars.items():
        par_trimmed = trim_par(par, parname)
        filename = outbase + '_' + parname + '.nii.gz'
        file_from_data(filename, par_trimmed, roi = read_im(roi_file), imref_file = imref_file)

def trim_par(par: npt.NDArray[np.float64], parname: str) -> npt.NDArray[np.float64]:
    """
    Trim parameter values beyond reasonable limits to avoid numerical error when saving to file.
    
    Arguments:
    par     -- vector with parameter values
    parname -- name of parameter value ('D', 'f', 'Dstar', 'vd' or 'K')
    
    Output:
    par     -- vector with trimmed parameter values
    """

    lims = {'D':10e-3, 'f':1, 'Dstar':1, 'vd':20, 'K':20, 'v':20, 'tau':10}
    if parname in lims.keys():
        par = np.clip(par, -lims[parname], lims[parname])
    return par

def valid_signal(Y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Return a mask representing all rows in Y with valid values (not non-positive, NaN or infinite).

    Arguments:
    Y    -- v x b matrix with data

    Output:
    mask -- vector of size v indicating valid rows in Y
    """

    mask = ~np.any((Y<=0) | np.isnan(Y) | np.isinf(Y), axis=1)
    return mask

def neighbours(mask: npt.NDArray[np.bool_]) -> npt.NDArray[np.int_]:
    """
    Return an array with index of all 4-neighbours for True elements in mask.

    Arguments:
    mask           -- 3D array identifying a mask in an image 

    Output:
    neighbour_mask -- array with index of all 4-neighbours for True elements in mask

    Note! index of neighbours outside the mask is set to the maximum index + 1
    """

    N = np.sum(mask)
    index_map = np.full(np.array(mask.shape)+2, N) # pad by 1 on each side
    index_map[1:-1,1:-1,1:-1][mask] = np.arange(N) 
    neighbour_mask = np.stack((index_map[0:-2,1:-1,1:-1][mask],
                                index_map[2:,1:-1,1:-1][mask],
                                index_map[1:-1,0:-2,1:-1][mask],
                                index_map[1:-1,2:,1:-1][mask],
                                ),axis=1)
    return neighbour_mask