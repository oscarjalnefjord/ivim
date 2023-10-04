""" Various useful function related to IVIM scanning on Philips MRI systems. """

import os
import numpy as np
import numpy.typing as npt
from ivim.io.base import read_bval, write_cval
from ivim.seq.fwf import c_from_b

def generate_dti_vectors_input(b: npt.NDArray[np.float64], filename: str, dualpol: bool = False, header: str = 'IVIM acquistion scheme') -> None:
    """
    Generate a dti_vectors_input file b-values and encoding directions (x, y, z or +/- x, y, z) throughout the scan.

    Arguments:
        b:        b-values [s/mm2]
        filename: path to dti_vectors_input file
        dualpol:  (optional) if False, three encoding directions are used (x, y, z), if True six encoding directions are used (+/- x, y, z)
        header:   (optional) title written to the dti_vectors_input file
    """
    
    nd = 6 if dualpol else 3  # Encoding directions
    rows = nd*len(b)
    X = np.zeros((rows, 4))
    
    i = 0
    i_off = 0
    for i, bi in enumerate(3*list(b)):
        X[i, (i+i_off)%3] = (-1)**((i%nd)>=3)
        X[i, 3] = bi
        if (i > 0) and ((i+1)%len(b) == 0) and (len(b)%3 == 0):
            i_off += 1  # Avoid repeating the same b-values for the same encoding directions.
    
    if nd == 6:
        X[rows//2:, :3] = -1 * X[:rows//2, :3]
        X[rows//2:, 3] = X[:rows//2, 3]
    
    X[abs(X)<1] = abs(X[abs(X)<1])  # Clean up negative zeroes.
    write_dti_vectors_input(X, filename, header)

def write_dti_vectors_input(X: npt.NDArray[np.float64], filename: str, header: str = '') -> None:
    """
    Save dti_vectors_input file in specified format.

    Arguments:
        X:        2D array were each row defines a 4-element tuple with the encoding direction in the first three elements and the b-value in the fourth.
        filename: path to dti_vectors_input file
        header:   (optional) title written to the dti_vectors_input file 
    """

    np.savetxt(filename, X, fmt='% .4f % .4f % .4f %.1f', header=header)

def read_dti_vectors_input(filename: str) -> npt.NDArray[np.float64]:
    """
    Load dti_vectors_input file.

    Arguments:
        filename: path to dti_vectors_input file

    Output:
        X:        2D array were each row defines a 4-element tuple with the encoding direction in the first three elements and the b-value in the fourth.
    """

    X = np.loadtxt(filename)
    return X

def gnorm_from_txt(examcard_file: str, g_file1: str, g_file2: str) -> tuple[npt.NDArray[np.float64], float]:
    """
    Get normalized gradient waveform and timging from files specific to the Neuromaster patch.

    Arguments:
        examcard_file: path to examcard file (export as txt from scanner)
        gfile1:        path to gradient waveform file 1 (pre 180 pulse)
        gfile2:        path to gradient waveform file 2 (post 180 pulse)

    Output:
        gnorm:         normalized gradient waveforms including the pause for the 180 pulse
        dt:            time between samples in gnorm [s]
    """
    
    with open(examcard_file) as f:
        for line in f:
            if '+WF dur. Pre (ms)' in line:  # for EPI
                T_pre = float(line.split()[-1][:-1]) * 1e-3  # ms -> s
            elif '+WF dur. Post (ms)' in line:  # for EPI
                T_post = float(line.split()[-1][:-1]) * 1e-3
            elif '+Target pause dur. (ms)' in line:
                T_pause = float(line.split()[-1][:-1]) * 1e-3
            elif '+WF dur. Both (ms)' in line: # for TSE
                T_pre = float(line.split()[-1][:-1]) * 1e-3
                T_post = T_pre
    
    gnorms = []
    for gfile in [g_file1, g_file2]:
        gnorms.append(np.loadtxt(gfile, skiprows=1))

    dt = T_pre / (gnorms[0].shape[0]-1)
    n = int((T_pre+T_pause+T_post)/dt + 1)
    gnorm = np.zeros((n, 3))
    gnorm[:gnorms[0].shape[0], :] = gnorms[0]
    gnorm[-gnorms[1].shape[0]:, :] = -gnorms[1] # minus to account for 180 pulse

    return gnorm, dt

def cval_from_files(bval_file, examcard_file: str, g_file1: str, g_file2: str, cval_file: str = '') -> None:
    """
    Generate cval file based on bval file and files specifying the gradient waveform shapes.

    Arguments:
        bval_file:     path to bval file
        examcard_file: path to examcard file (export as txt from scanner)
        gfile1:        path to gradient waveform file 1 (pre 180 pulse)
        gfile2:        path to gradient waveform file 2 (post 180 pulse)
        cval_file:     (optional) path to cval file. Set to bval file name with .cval extension if not specified
    """
    
    gnorm, dt = gnorm_from_txt(examcard_file, g_file1, g_file2)
    c = c_from_b(read_bval(bval_file), gnorm, dt)
    
    if cval_file == '':
        cval_file = os.path.splitext(bval_file)[0] + '.cval'
    write_cval(cval_file, c)


def gnorm_to_txt(filename: str, gnorm: npt.NDArray[np.float64]) -> None:
    """
    Save normalized gradient waveforms in format needed for the Neuromaster patch.

    Arguments:
        filename: path to gradient waveform file
        gnorm:    n x 3 array to be written to the file, with n being the number of samples
    """
    
    if gnorm.ndim != 2:
        raise ValueError('gnorm must be a 2D array.')
    if gnorm.shape[1] != 3:
        raise ValueError('gnorm must be an n x 3 array.')
    
    np.savetxt(filename, gnorm, fmt='% .5f', header=str(gnorm.shape[0]), comments='\t')