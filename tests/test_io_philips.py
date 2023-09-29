import os
import numpy as np
import tempfile
from ivim.io.philips import (generate_dti_vectors_input, write_dti_vectors_input, read_dti_vectors_input,
                             gnorm_from_txt, gnorm_to_txt, cval_from_files)
from ivim.io.base import write_bval, read_cval
from ivim.seq.fwf import c_from_b

# Paths to data
data_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)),'test_data')
temp_folder = tempfile.gettempdir()

# Target data
target_gnorm = np.array([[0, 0, 0], 
                        [1, 0, 0], 
                        [1, 0, 0],
                        [1, 0, 0], 
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0], 
                        [-1, 0, 0],
                        [-1, 0, 0], 
                        [-1, 0, 0], 
                        [0, 0, 0]])

target_dti_vectors = np.array([[1.000, 0.000, 0.000, 10.0],
                               [0.000, -1.000, 0.000, 20.2],
                               [0.000, 0.000, 1.000, 100.1],
                               [0.000, 1.000, 0.000, 200.2]])

# Test functions
def test_generate_dti_vectors_input():
    filename = os.path.join(temp_folder, 'temp_dvi.txt')
    for i in range(2,10):
        b = np.round(1000.0 * np.random.rand(i), 1)
        for dualpol in [True, False]:
            generate_dti_vectors_input(b, filename, dualpol = dualpol)
            X = read_dti_vectors_input(filename)
            for bi in b:
                Xsub = X[bi == X[:,3], :]
                np.testing.assert_equal(Xsub.shape, (3 * 2**dualpol, 4))
                for j in range(3):
                    for k in range(2**dualpol):
                        x = np.zeros(4)
                        x[3] = bi   
                        x[j] = (-1)**k
                        np.testing.assert_array_equal(np.sum(np.all(Xsub == x[np.newaxis, :], axis = 0)), 1)

def test_write_dti_vectors_input():
    dvi_file = os.path.join(temp_folder, 'temp_dvi.txt')
    for header in ['test', '']:
        write_dti_vectors_input(target_dti_vectors, dvi_file, header = header)
        X = read_dti_vectors_input(dvi_file)
        np.testing.assert_array_equal(X, target_dti_vectors)

        with open(dvi_file) as f:
            first_line = f.readline().strip('\n')
        if header == '':
            np.testing.assert_equal(first_line, ' 1.0000  0.0000  0.0000 10.0')
        else:
            np.testing.assert_equal(first_line, '# ' + header)

def test_read_dti_vectors_input():
    X = read_dti_vectors_input(os.path.join(data_folder, 'test_dti_vectors_input.txt'))
    np.testing.assert_array_equal(X, target_dti_vectors)

def test_gnorm_from_text():
    g_file = os.path.join(data_folder, 'test_gnorm.txt')
    
    for i, target_dt in enumerate([50e-3/10, 50e-3/10]):
        examcard_file = os.path.join(data_folder, f'test_examcard{i+1}.txt')
        gnorm, dt = gnorm_from_txt(examcard_file, g_file, g_file)
        np.testing.assert_array_almost_equal(gnorm, target_gnorm, 3)
        np.testing.assert_almost_equal(dt, target_dt, 3)

def test_gnorm_to_text():
    filename = os.path.join(temp_folder, 'temp_gnorm.txt')
    gnorm_to_txt(filename, target_gnorm[:5])
    gnorm, _ = gnorm_from_txt(os.path.join(data_folder,'test_examcard1.txt'),filename,filename)
    np.testing.assert_array_almost_equal(gnorm, target_gnorm, 3)

def test_cval_from_files():
    bval_file = os.path.join(temp_folder, 'temp.bval')
    b = np.array([0.0, 10.0, 20.0, 100.0, 200.0])
    write_bval(bval_file, b)
    examcard_file = os.path.join(data_folder, 'test_examcard1.txt')
    g_file = os.path.join(data_folder, 'test_gnorm.txt')
    for cval_file in [bval_file.replace('bval', 'cval'), '']:
        cval_from_files(bval_file, examcard_file, g_file, g_file, cval_file = cval_file)
        c = read_cval(bval_file.replace('bval', 'cval'))
        np.testing.assert_array_almost_equal(c, c_from_b(b, target_gnorm, 50e-3/10), 3)