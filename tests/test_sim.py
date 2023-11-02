import numpy as np
import os
import tempfile
from ivim.models import sIVIM, diffusive, ballistic, intermediate, NO_REGIME, DIFFUSIVE_REGIME, BALLISTIC_REGIME, INTERMEDIATE_REGIME
from ivim.sim import noise
from ivim.seq.sde import MONOPOLAR, BIPOLAR
from ivim.io.base import read_im, write_im, write_bval, write_cval, write_time, write_k

# Paths to data
temp_folder = tempfile.gettempdir()

bval_file = os.path.join(temp_folder, 'sim.bval')
cval_file = os.path.join(temp_folder, 'sim.cval')
delta_file = os.path.join(temp_folder, 'sim.delta')
Delta_file = os.path.join(temp_folder, 'sim.Delta')
T_file = os.path.join(temp_folder, 'sim.T')
k_file = os.path.join(temp_folder, 'sim.k')
D_file = os.path.join(temp_folder, 'sim_D.nii.gz')
f_file = os.path.join(temp_folder, 'sim_f.nii.gz')
vd_file = os.path.join(temp_folder, 'sim_vd.nii.gz')
Dstar_file = os.path.join(temp_folder, 'sim_Dstar.nii.gz')
v_file = os.path.join(temp_folder, 'sim_v.nii.gz')
tau_file = os.path.join(temp_folder, 'sim_tau.nii.gz')
K_file = os.path.join(temp_folder, 'sim_K.nii.gz')
S0_file = os.path.join(temp_folder, 'sim_S0.nii.gz')
outbase = os.path.join(tempfile.tempdir, 'sim')

# Gemerate data
b = np.array([0, 10, 20, 100, 300, 500])
write_bval(bval_file, b)
c = np.array([0, 0.3, 0.6, 1.0, 1.5, 2.0])
write_cval(cval_file, c)
delta = 0.01 * np.ones_like(b)
write_time(delta_file, delta)
Delta = 0.015 * np.ones_like(b)
write_time(Delta_file, Delta)
T = np.array([0.05, 0.05, 0.1, 0.1, 0.05, 0.05])
write_time(T_file, T)
k = np.array([1, 1, -1, -1, 1, 1])
write_k(k_file, k)

#sz = (10, 20, 30)
sz = (1,1,1)
for file, intercept, slope in zip([D_file, f_file, vd_file, Dstar_file, v_file, tau_file, K_file, S0_file],
                      [0.5e-3,   0.05,     1.5,      10e-3,    1.5,     0.05,    0.5,    0.95],
                      [1.0e-3,   0.15,     1.0,      20e-3,    1.0,     0.10,    1.5,    0.10]):
    write_im(file, intercept + slope*np.random.rand(sz[0], sz[1], sz[2]))

# Test functions
def test_noise():
    for regime in [NO_REGIME, DIFFUSIVE_REGIME, BALLISTIC_REGIME, INTERMEDIATE_REGIME]:
        if regime == INTERMEDIATE_REGIME:
            seqs = [MONOPOLAR, BIPOLAR]
        else:
            seqs = [MONOPOLAR]
        for seq in seqs:
            for S0test_file in [S0_file, None]:
                for Ktest_file in [K_file, None]:
                    for Dstartest_file in [Dstar_file, None]:
                        if (Dstartest_file is None) and (regime == DIFFUSIVE_REGIME):
                            continue
                        for vdtest_file, cvaltest_file in zip([vd_file, None], [cval_file, None]):
                            if (vdtest_file is None) and (regime == BALLISTIC_REGIME):
                                continue
                            for tautest_file, vtest_file, deltatest_file, Deltatest_file, ktest_file, Ttest_file in zip(
                                [tau_file, None], [v_file, None], [delta_file, None], [Delta_file, None], [k_file, None], [T_file, None]):
                                if (tautest_file is None) and (regime == INTERMEDIATE_REGIME):
                                    continue
                                
                                noise(D_file, f_file, regime, bval_file, noise_sigma = 0.0001, 
                                    outbase=outbase, S0_file=S0test_file, Dstar_file=Dstartest_file,
                                    K_file=Ktest_file, vd_file=vdtest_file,cval_file=cvaltest_file,
                                    tau_file=tautest_file, v_file=vtest_file, delta_file=deltatest_file,
                                    Delta_file=Deltatest_file, T_file=Ttest_file, k_file=ktest_file, seq=seq)

                                # Check that output has correct shape
                                im = read_im(outbase+'.nii.gz')
                                np.testing.assert_equal(np.shape(im),tuple(list(sz)+[b.size]))

                                # Check that output is close to predicted values
                                if S0test_file is None:
                                    S0 = np.ones(sz)
                                else:
                                    S0 = read_im(S0test_file)
                                if Ktest_file is None:
                                    K = np.zeros(sz) 
                                else:
                                    K = read_im(Ktest_file)
                                if regime == DIFFUSIVE_REGIME:
                                    im_nonoise = diffusive(b, read_im(D_file), read_im(f_file), read_im(Dstar_file), S0, K)
                                elif regime == BALLISTIC_REGIME:
                                    im_nonoise = ballistic(b, c, read_im(D_file), read_im(f_file), read_im(vd_file), S0, K)
                                elif regime == INTERMEDIATE_REGIME:
                                    im_nonoise = intermediate(b, delta, Delta, read_im(D_file), read_im(f_file), read_im(v_file), 
                                                            read_im(tau_file), S0, K, seq=seq, T=T, k=k)
                                else:
                                    im_nonoise = sIVIM(b, read_im(D_file), read_im(f_file), S0, K)
                                np.testing.assert_allclose(im, im_nonoise, rtol = .01, atol = 0.01)