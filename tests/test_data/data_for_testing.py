import os
import numpy as np
import nibabel as nib

nii = nib.Nifti2Image(np.random.rand(2,3,4,4), affine = np.diag([4,3,2,1]))
nib.save(nii,os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test.nii.gz'))
