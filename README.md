# `ivim` - An end-to-end python package for intravoxel incoherent motion (IVIM) analysis of diffusion MRI data

[![tests passing](https://github.com/oscarjalnefjord/ivim/actions/workflows/tests.yml/badge.svg)](https://github.com/oscarjalnefjord/ivim/actions)
[![codecov](https://codecov.io/gh/oscarjalnefjord/ivim/graph/badge.svg?token=V7LGUH6ZVP)](https://codecov.io/gh/oscarjalnefjord/ivim)
[![Documentation Status](https://readthedocs.org/projects/ivim/badge/?version=latest)](https://ivim.readthedocs.io/en/latest/?badge=latest)

The `ivim` python package provides functionality for the main steps associated with IVIM analysis of diffusion MRI data including:
- Methods for IVIM parameter estimation / model fitting
- Simulation of MRI data based on IVIM models
- Optimization of b-values 
- Preprocessing, both basic operation like extracting image data with a specific b-value or averaging of diffusion encoding directions, and specific methods like correction for signal drift

The file formats used are the ones used by [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSL), i.e. .nii.gz for image data, .bval for b-values and .bvec for diffusion encoding directions. File with ending .cval is also used for c-values in analogy to .bval for b-values to describe flow encoding.

## Dependencies
The `ivim` python package relies heavly on numpy for numerical operations, uses nibabel for reading and writing nifti file, and uses scipy for some specific minimization/optimization tasks. A python version >= 3.10 is required.

The correction for susceptibility and eddy current induced distorsions wraps FSL functionality, thus requiring [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSL) to be installed.

## How to install
The suggested way to install is to first prepare an environment suitable for the `ivim` package using e.g. conda with a compatible python version (>= 3.10):

    conda create -n ivim python == 3.10

Activate the environment and install the prerequisities:

    conda activate ivim
    conda install -c anaconda -c conda-forge nibabel scipy numpy

Download the package with:

    git clone https://github.com/oscarjalnefjord/ivim.git

To install the package, run:

    pip install -e ivim

If you are not in the directory of the repository, run:

    pip install -e /path/to/ivim

where `/path/to/ivim` points to the directory of the repository.

## References
The following work describe some of the algorithms implemented in the package.

1. Jalnefjord et al. Comparison of methods for estimation of the intravoxel incoherent motion (IVIM) diffusion coefficient (D) and perfusion fraction (f). Magnetic Resonance Materials in Physics, Biology and Medicine 2018; 31(6):715-723. https://doi.org/10.1007/s10334-018-0697-5

2. Gustafsson et al. Impact of prior distributions and central tendency measures on Bayesian intravoxel incoherent motion model fitting. Magnetic Resonance in Medicine 2018; 79(3):1674-1683. https://doi.org/10.1002/mrm.26783 

3. Jalnefjord et al. Optimization of b‐value schemes for estimation of the diffusion coefficient and the perfusion fraction with segmented intravoxel incoherent motion model fitting. Magnetic Resonance in Medicine 2019; 82(4):1541-1552. https://doi.org/10.1002/mrm.27826 