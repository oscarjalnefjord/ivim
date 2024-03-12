import numpy as np
import os.path
import tempfile
from ivim.models import NO_REGIME, DIFFUSIVE_REGIME, BALLISTIC_REGIME, INTERMEDIATE_REGIME
from ivim.sim import noise
from ivim.seq.sde import calc_c, G_from_b, MONOPOLAR, BIPOLAR
from ivim.io.base import write_im, read_im, write_bval, write_cval, write_time, write_k
from ivim.fit import valid_signal, trim_par, neighbours, save_parmaps, nlls, seg, bayes

# Paths to data
temp_folder = tempfile.gettempdir()

# Test functions
mcmc_n = 10000
sz = (2,3,4)
roi_file = os.path.join(temp_folder, 'temp_fitroi.nii.gz')
roi = np.random.rand(sz[0], sz[1], sz[2]) > 0.5
write_im(roi_file, roi.astype(np.int32))
outbase = os.path.join(temp_folder, 'test_fit')
im_file = outbase + '.nii.gz'
bval_file = os.path.join(temp_folder, 'temp_fit.bval')

parmaps = {
            'D': 0.5e-3 + 1e-3 * np.random.rand(sz[0], sz[1], sz[2]),
            'f': 0.15 + 0.1 * np.random.rand(sz[0], sz[1], sz[2]),
            'Dstar': 20e-3 + 20e-3 * np.random.rand(sz[0], sz[1], sz[2]),
            'vd': 1.5 + 1 * np.random.rand(sz[0], sz[1], sz[2]),
            'K': 2 * np.random.rand(sz[0], sz[1], sz[2]),
            'S0': np.ones(sz),
            'v': 1.5 + 1.0 * np.random.rand(sz[0], sz[1], sz[2]),
            'tau': 0.05 + 0.1 * np.random.rand(sz[0], sz[1], sz[2])
            }

parmap_files = {}
for par, parmap in parmaps.items():
    parmap_files[par] = os.path.join(temp_folder, f'temp_fit_{par}.nii.gz')
    write_im(parmap_files[par], parmap)

b = np.array([0, 10, 20, 50, 100, 200, 400, 800, 1500]).astype(float)
delta = 7.3e-3
Delta = 7.5e-3

rtol = 0.05
atol = {'D': 0.1e-3, 'f': 0.01, 'K': 0.1, 'S0': 0.1, 'Dstar': 2e-3, 'vd': 0.1, 'v': 0.1, 'tau': 0.01}

def test_nlls_sIVIM():
    bthr = 200
    for fitK in [True, False]:
        test_pars = ['D','f','S0']
        if fitK:
            test_pars += ['K']
            K_file = parmap_files['K']
            test_b = b[(b==0) | (b>=bthr)]
        else:
            K_file = None
            test_b = b[(b==0) | ((b>=bthr) & (b<1000))]
        write_bval(bval_file, test_b)

        noise(parmap_files['D'], parmap_files['f'], NO_REGIME, bval_file, 0, outbase, parmap_files['S0'], K_file = K_file)

        nlls(im_file, bval_file, NO_REGIME, roi_file = roi_file, outbase = outbase, fitK = fitK)
        for test_par in test_pars:
            np.testing.assert_allclose(read_im(outbase + f'_{test_par}.nii.gz')[roi], parmaps[test_par][roi], rtol = rtol, atol = atol[test_par])

def test_nlls_diffusive():
    for fitK in [True, False]:
        test_pars = ['D','f','Dstar','S0']
        if fitK:
            test_pars += ['K']
            K_file = parmap_files['K']
            test_b = b
        else:
            K_file = None
            test_b = b[b<1000]
        write_bval(bval_file, test_b)

        noise(parmap_files['D'], parmap_files['f'], DIFFUSIVE_REGIME, bval_file, 0, outbase, parmap_files['S0'], K_file = K_file, Dstar_file = parmap_files['Dstar'])

        nlls(im_file, bval_file, DIFFUSIVE_REGIME, roi_file = roi_file, outbase = outbase, fitK = fitK)
        for test_par in test_pars:
            np.testing.assert_allclose(read_im(outbase + f'_{test_par}.nii.gz')[roi], parmaps[test_par][roi], rtol = rtol, atol = atol[test_par])

def test_nlls_ballistic():
    for seq in ['monopolar', 'bipolar']:
        for fitK in [True, False]:
            test_pars = ['D','f','vd','S0']
            if fitK:
                test_pars += ['K']
                K_file = parmap_files['K']
                test_b = b
            else:
                K_file = None
                test_b = b[b<1000]
            if seq == 'bipolar':
                test_b = np.tile(test_b, 2)
                test_c = np.zeros_like(test_b)
                test_c[:test_b.size//2] = calc_c(G_from_b(test_b[:test_b.size//2], Delta, delta, seq = seq), Delta, delta, seq = seq)
            else:
                test_c = calc_c(G_from_b(test_b, Delta, delta, seq = seq), Delta, delta, seq = seq)

            write_bval(bval_file, test_b)
            cval_file = bval_file.replace('bval', 'cval')
            write_cval(cval_file, test_c)

            noise(parmap_files['D'], parmap_files['f'], BALLISTIC_REGIME, bval_file, 0, outbase, parmap_files['S0'], K_file = K_file, vd_file = parmap_files['vd'], cval_file = cval_file)

            nlls(im_file, bval_file, BALLISTIC_REGIME, roi_file = roi_file, outbase = outbase, fitK = fitK, cval_file = cval_file)
            for test_par in test_pars:
                np.testing.assert_allclose(read_im(outbase + f'_{test_par}.nii.gz')[roi], parmaps[test_par][roi], rtol = rtol, atol = atol[test_par])

def test_seg_sIVIM():
    for bthr in [200, 400, b[-1]]:
        for fitK in [True, False]:
            test_pars = ['D','f','S0']
            if fitK:
                test_pars += ['K']
                K_file = parmap_files['K']
                test_b = b[(b==0) | (b>=bthr)]
            else:
                K_file = None
                test_b = b[(b==0) | ((b>=bthr) & (b<1000))]
            write_bval(bval_file, test_b)

            noise(parmap_files['D'], parmap_files['f'], NO_REGIME, bval_file, 0, outbase, parmap_files['S0'], K_file = K_file)
            if bthr == b[-1]:
                np.testing.assert_raises(ValueError, seg, im_file, bval_file, NO_REGIME, bthr = bthr, roi_file = roi_file, outbase = outbase, fitK = fitK)
            else:
                seg(im_file, bval_file, NO_REGIME, bthr = bthr, roi_file = roi_file, outbase = outbase, fitK = fitK)
                for test_par in test_pars:
                    np.testing.assert_allclose(read_im(outbase + f'_{test_par}.nii.gz')[roi], parmaps[test_par][roi], rtol = rtol, atol = atol[test_par])

def test_seg_diffusive():
    for bthr in [200, 400, b[-1]]:
        for fitK in [True, False]:
            test_pars = ['D','f','Dstar','S0']
            if fitK:
                test_pars += ['K']
                K_file = parmap_files['K']
                test_b = b
            else:
                K_file = None
                test_b = b[b<1000]
            write_bval(bval_file, test_b)

            noise(parmap_files['D'], parmap_files['f'], DIFFUSIVE_REGIME, bval_file, 0, outbase, parmap_files['S0'], K_file = K_file, Dstar_file = parmap_files['Dstar'])

            if bthr == b[-1]:
                np.testing.assert_raises(ValueError, seg, im_file, bval_file, DIFFUSIVE_REGIME, bthr = bthr, roi_file = roi_file, outbase = outbase, fitK = fitK)
            else:
                seg(im_file, bval_file, DIFFUSIVE_REGIME, bthr = bthr, roi_file = roi_file, outbase = outbase, fitK = fitK)
                for test_par in test_pars:
                    np.testing.assert_allclose(read_im(outbase + f'_{test_par}.nii.gz')[roi], parmaps[test_par][roi], rtol = rtol, atol = atol[test_par])

def test_seg_ballistic():
    cthr = 0
    for seq in ['monopolar', 'bipolar']:
        for bthr in [200, 400, b[-1]]:
            for fitK in [True, False]:
                test_pars = ['D','f','vd','S0']
                if fitK:
                    test_pars += ['K']
                    K_file = parmap_files['K']
                    test_b = b
                else:
                    K_file = None
                    test_b = b[b<1000]
                if seq == 'bipolar':
                    test_b = np.tile(test_b, 2)
                    test_c = np.zeros_like(test_b)
                    test_c[:test_b.size//2] = calc_c(G_from_b(test_b[:test_b.size//2], Delta, delta, seq = seq), Delta, delta, seq = seq)
                else:
                    test_c = calc_c(G_from_b(test_b, Delta, delta, seq = seq), Delta, delta, seq = seq)

                write_bval(bval_file, test_b)
                cval_file = bval_file.replace('bval', 'cval')
                write_cval(cval_file, test_c)

                noise(parmap_files['D'], parmap_files['f'], BALLISTIC_REGIME, bval_file, 0, outbase, parmap_files['S0'], K_file = K_file, vd_file = parmap_files['vd'], cval_file = cval_file)

                if bthr == b[-1]:
                    np.testing.assert_raises(ValueError, seg, im_file, bval_file, BALLISTIC_REGIME, bthr = bthr, cval_file = cval_file, cthr = cthr, roi_file = roi_file, outbase = outbase, fitK = fitK)
                else:
                    seg(im_file, bval_file, BALLISTIC_REGIME, bthr = bthr, cval_file = cval_file, cthr = cthr, roi_file = roi_file, outbase = outbase, fitK = fitK)
                    for test_par in test_pars:
                        np.testing.assert_allclose(read_im(outbase + f'_{test_par}.nii.gz')[roi], parmaps[test_par][roi], rtol = rtol, atol = atol[test_par])

def test_bayes_sIVIM():
    bthr = 200
    for spatial_prior in [True, False]:
        for ctm in ['mean', 'mode']:
            for fitK in [True, False]:
                test_pars = ['D','f','S0']
                if fitK:
                    test_pars += ['K']
                    K_file = parmap_files['K']
                    test_b = b[(b==0) | (b>=bthr)]
                else:
                    K_file = None
                    test_b = b[(b==0) | ((b>=bthr) & (b<1000))]
                write_bval(bval_file, test_b)

                noise(parmap_files['D'], parmap_files['f'], NO_REGIME, bval_file, 0, outbase, parmap_files['S0'], K_file = K_file)

                bayes(im_file, bval_file, NO_REGIME, roi_file = roi_file, outbase = outbase, fitK = fitK, n = mcmc_n, ctm = ctm, spatial_prior = spatial_prior)
                for test_par in test_pars:
                    np.testing.assert_allclose(read_im(outbase + f'_{test_par}.nii.gz')[roi], parmaps[test_par][roi], rtol = rtol, atol = atol[test_par])
                
def test_bayes_diffusive():
    for spatial_prior in [True, False]:
        for ctm in ['mean', 'mode']:
            for fitK in [True, False]:
                for output_std in [True, False]:
                    test_pars = ['D','f','Dstar','S0']
                    if fitK:
                        test_pars += ['K']
                        K_file = parmap_files['K']
                        test_b = b
                    else:
                        K_file = None
                        test_b = b[b<1000]
                    write_bval(bval_file, test_b)

                    noise(parmap_files['D'], parmap_files['f'], DIFFUSIVE_REGIME, bval_file, 0, outbase, parmap_files['S0'], K_file = K_file, Dstar_file = parmap_files['Dstar'])

                    bayes(im_file, bval_file, DIFFUSIVE_REGIME, roi_file = roi_file, outbase = outbase, fitK = fitK, n = mcmc_n, ctm = ctm, spatial_prior = spatial_prior, output_std = output_std)
                    for test_par in test_pars:
                        np.testing.assert_allclose(read_im(outbase + f'_{test_par}.nii.gz')[roi], parmaps[test_par][roi], rtol = rtol, atol = atol[test_par])
                        if output_std:
                            np.testing.assert_equal(read_im(outbase + f'-std_{test_par}.nii.gz').shape,parmaps[test_par].shape)

def test_bayes_ballistic():
    for spatial_prior in [True, False]:
        for ctm in ['mean', 'mode']:
            for seq in ['monopolar', 'bipolar']:
                for fitK in [True, False]:
                    for output_std in [True, False]:
                        test_pars = ['D','f','vd','S0']
                        if fitK:
                            test_pars += ['K']
                            K_file = parmap_files['K']
                            test_b = b
                        else:
                            K_file = None
                            test_b = b[b<1000]
                        if seq == 'bipolar':
                            test_b = np.tile(test_b, 2)
                            test_c = np.zeros_like(test_b)
                            test_c[:test_b.size//2] = calc_c(G_from_b(test_b[:test_b.size//2], Delta, delta, seq = seq), Delta, delta, seq = seq)
                        else:
                            test_c = calc_c(G_from_b(test_b, Delta, delta, seq = seq), Delta, delta, seq = seq)

                        write_bval(bval_file, test_b)
                        cval_file = bval_file.replace('bval', 'cval')
                        write_cval(cval_file, test_c)

                        noise(parmap_files['D'], parmap_files['f'], BALLISTIC_REGIME, bval_file, 0, outbase, parmap_files['S0'], K_file = K_file, vd_file = parmap_files['vd'], cval_file = cval_file)

                        bayes(im_file, bval_file, BALLISTIC_REGIME, cval_file = cval_file, roi_file = roi_file, outbase = outbase, fitK = fitK, n = mcmc_n, ctm = ctm, spatial_prior = spatial_prior, output_std = output_std)
                        for test_par in test_pars:
                            np.testing.assert_allclose(read_im(outbase + f'_{test_par}.nii.gz')[roi], parmaps[test_par][roi], rtol = rtol, atol = atol[test_par])
                            if output_std:
                                np.testing.assert_equal(read_im(outbase + f'-std_{test_par}.nii.gz').shape,parmaps[test_par].shape)

def test_bayes_intermediate():
    if False:
        test_b = np.tile(b[b<1000], 2)
        test_delta = delta*np.ones_like(test_b)
        test_Delta = Delta*np.ones_like(test_b)
        write_bval(bval_file, test_b)
        delta_file = bval_file.replace('bval', 'delta')
        write_time(delta_file, test_delta)
        Delta_file = bval_file.replace('bval', 'Delta')
        write_time(Delta_file, test_Delta)

        test_k = np.ones_like(test_b)
        test_k[:test_b.size//2] = -1
        test_T = 2*(delta+Delta) + 0.1*np.random.rand(test_b.size)
        k_file = bval_file.replace('bval', 'k')
        write_k(k_file, test_k)
        T_file = bval_file.replace('bval', 'T')
        write_time(T_file, test_T)

        seq = BIPOLAR
        K_file = None

        noise(parmap_files['D'], parmap_files['f'], INTERMEDIATE_REGIME, bval_file, 0, outbase, 
            parmap_files['S0'], K_file = K_file, v_file = parmap_files['v'], tau_file=parmap_files['tau'], 
            delta_file=delta_file, Delta_file=Delta_file, T_file=T_file, k_file=k_file, seq=seq)
        
        spatial_prior = False
        ctm = 'mean'
        fitK = False
        bayes(im_file, bval_file, INTERMEDIATE_REGIME, roi_file = roi_file, outbase = outbase, fitK = fitK, n = mcmc_n, ctm = ctm, spatial_prior = spatial_prior,
                seq=seq, delta_file=delta_file, Delta_file=Delta_file, T_file=T_file, k_file=k_file)
        
        test_pars = ['D','f','v','tau','S0']
        for test_par in test_pars:
            np.testing.assert_allclose(read_im(outbase + f'_{test_par}.nii.gz')[roi], parmaps[test_par][roi], rtol = rtol, atol = atol[test_par])
    
    for spatial_prior in [True, False]:
        for ctm in ['mean', 'mode']:
            for seq in [BIPOLAR]:
                for fitK in [True, False]:
                    for output_std in [True, False]:
                        test_pars = ['D','f','v','tau','S0']
                        if fitK:
                            test_pars += ['K']
                            K_file = parmap_files['K']
                            test_b = b
                        else:
                            K_file = None
                            test_b = b[b<1000]

                        if True:
                            test_b = np.tile(test_b, 2)
                            test_delta = delta*np.ones_like(test_b)
                            test_Delta = Delta*np.ones_like(test_b)
                            write_bval(bval_file, test_b)
                            delta_file = bval_file.replace('bval', 'delta')
                            write_time(delta_file, test_delta)
                            Delta_file = bval_file.replace('bval', 'Delta')
                            write_time(Delta_file, test_Delta)

                            if seq == 'bipolar':
                                test_k = np.ones_like(test_b)
                                test_k[:test_b.size//2] = -1
                                test_T = 2*(delta+Delta) + 0.1*np.random.rand(test_b.size)
                                k_file = bval_file.replace('bval', 'k')
                                write_k(k_file, test_k)
                                T_file = bval_file.replace('bval', 'T')
                                write_time(T_file, test_T)
                            else:
                                k_file = None
                                T_file = None
                        
                            noise(parmap_files['D'], parmap_files['f'], INTERMEDIATE_REGIME, bval_file, 0, outbase, parmap_files['S0'], K_file = K_file, v_file = parmap_files['v'], tau_file=parmap_files['tau'], 
                                delta_file=delta_file, Delta_file=Delta_file, T_file=T_file, k_file=k_file, seq=seq)

                            bayes(im_file, bval_file, INTERMEDIATE_REGIME, roi_file = roi_file, outbase = outbase, fitK = fitK, n = mcmc_n, ctm = ctm, spatial_prior = spatial_prior, output_std = output_std,
                                seq=seq, delta_file=delta_file, Delta_file=Delta_file, T_file=T_file, k_file=k_file)
                            for test_par in test_pars:
                                np.testing.assert_allclose(read_im(outbase + f'_{test_par}.nii.gz')[roi], parmaps[test_par][roi], rtol = rtol, atol = atol[test_par])
                                if output_std:
                                    np.testing.assert_equal(read_im(outbase + f'-std_{test_par}.nii.gz').shape,parmaps[test_par].shape)

def test_save_parmaps():

    sz = (3,4,5)
    roi = np.random.rand(sz[0], sz[1], sz[2]) > 0.5
    temp_file = os.path.join(temp_folder,'temp.nii.gz')
    write_im(temp_file, roi.astype(np.int32))

    Dmap = 10e-3 * np.random.rand(sz[0], sz[1], sz[2]) + 1e-3
    fmap = 1.0 * np.random.rand(sz[0], sz[1], sz[2]) + 0.1
    atol = {'D': 1e-6, 'f': 1e-5}

    for outbase in [os.path.join(temp_folder,'temp'), None]:
        for imref_file in [temp_file, None]:
            for roi_file in [temp_file, None]:
                if ((outbase == None) or (roi_file == None)) and (imref_file == None):
                    np.testing.assert_raises(ValueError, save_parmaps, pars, outbase = outbase, imref_file = imref_file, roi_file = roi_file)
                else:
                    if roi_file == None:
                        test_roi = Dmap > 0
                    else:
                        test_roi = roi.copy()
                    pars = {'D': Dmap[test_roi], 'f': fmap[test_roi]}
                    
                    save_parmaps(pars, outbase = outbase, imref_file = imref_file, roi_file = roi_file)
                    for par, values in pars.items():
                        np.testing.assert_allclose(read_im(os.path.join(temp_folder, 'temp_' + par + '.nii.gz'))[test_roi], trim_par(values, par), rtol = 1e-3, atol = atol[par])

def test_trim_par():
    lims = {'D':10e-3, 'f':1, 'Dstar':1, 'vd':20, 'K':20}
    x = [1, -2, 0.5, 20, 100]
    for parname, lim in lims.items():
        np.testing.assert_array_less(trim_par(x, parname),lim+1e-10)

def test_valid_signal():
    Y = np.ones((6,3))
    Y[1, 0] = 0
    Y[2, 1] = -1
    Y[3, 2] = np.nan
    Y[4, 1] = np.inf
    np.testing.assert_array_equal(valid_signal(Y), np.array([True, False, False, False, False, True]))

def test_neighbours():
    mask = np.full((2, 3, 4), False)
    mask[0, 0, 0] = True
    mask[1, 0, 0] = True
    mask[0, 1, 0] = True
    N = np.sum(mask)

    expected_mask = np.full((N, 4), N)
    expected_mask[0, 1] = 2
    expected_mask[0, 3] = 1
    expected_mask[1, 2] = 0
    expected_mask[2, 0] = 0

    np.testing.assert_equal(neighbours(mask), expected_mask)

test_bayes_intermediate()