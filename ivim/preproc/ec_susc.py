""" Tools to correct for eddy current and susceptibility induced distorsions. """

import os 
import numpy as np
import shutil
import warnings
import tempfile
from ivim.io.base import read_bval, read_im, write_bval
from ivim.preproc.base import combine, extract

if shutil.which('fsl') is None:
    warnings.warn('FSL installation not found. Correction for motion, eddy currents and susceptibility induced distorsions will not work.')

def ec_topup(im_file: str, bval_file: str, imrev_file: str, outbase: str, bvalrev_file: str | None = None, save_inter: bool = False):
    """
    Run FSL eddy_correct and TOPUP.
    
    Arguments:
        im_file:      path to nifti image file
        bval_file:    path to .bval file
        imrev_file:   path to nifit image file for image with reversed phase encoding direction
        outbase:      basis for output filenames, i.e. filename without file extension to which .nii.gz, .bval, etc. is added
        bvalrev_file: (optional) path to .bval file for the imrev_file. If not given, it is assumed imrev_file only contains b = 0 images 
        save_inter:   (optional) if True, files at intermediate steps are saved to output folder

    To reduce the number of arguments, the following assumptions are made:
    - phase encoding is in the column (up down) dimension, which usually translate to AP for brain imaging
    - phase encoding is in the positive direction for im_file and negative direction for imrev_file. For brain imaging with phase
      encoding in the AP dimension this corresponds to distorsion (see e.g. ear-canals) in the anterior direction for im_file and 
      in the posterior direction for imrev_file

    The user is referred to https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/eddy for additional information.
    Note that FSL EDDY is not suitable for the typical IVIM data as it is designed for "few b-values, many encoding direction".
    According to the FSL webpage (https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/topup -> Introduction) it is preferred to run 
    eddy_correct and then applytopup. 
    """
    
    if save_inter:
        interbase = outbase 
    else:
        interbase = os.path.join(tempfile.gettempdir(), 'temp')

    # eddy_correct
    b = read_bval(bval_file)
    if np.sum(b == 0) == 0:
        raise ValueError('No b = 0 found in data')
    ref = np.argmax(b == 0)
    ecc_file = f'{interbase}_ecc.nii.gz'
    ec_cmd = f'eddy_correct {im_file} {ecc_file} {ref}'
    os.system(ec_cmd)

    # topup
    if bvalrev_file is None:
        bvalrev_file = interbase + '_rev.bval'
        imrev = read_im(imrev_file)
        brev = np.zeros(imrev.shape[3])
        write_bval(bvalrev_file, brev)
    else:
        brev = read_bval(bvalrev_file)
    outbase_combine = interbase + '_comb'
    combine([im_file, imrev_file], [bval_file, bvalrev_file], outbase_combine)
    outbase_extract = interbase + '_b0'
    extract(outbase_combine + '.nii.gz', outbase_combine + '.bval', outbase_extract)

    acqp_file = interbase + '_acqparams.txt'
    with open(acqp_file,'w') as f:
        for _ in range(np.sum(b==0)):
            f.write('0 1 0 0.050\n')
        for _ in range(np.sum(brev==0)):
            f.write('0 -1 0 0.050\n')

    topup_cmd = f'topup --imain={outbase_extract}.nii.gz --datain={acqp_file} --config=b02b0.cnf --out={interbase} --verbose'
    os.system(topup_cmd)

    applytopup_cmd = f'applytopup --imain={ecc_file} --datain={acqp_file} --inindex=1 --topup={interbase} --out={outbase} --method=jac --verbose'
    os.system(applytopup_cmd)             

    
