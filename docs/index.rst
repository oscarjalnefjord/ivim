**************************************
``ivim`` documentation
**************************************

``ivim`` is an end-to-end python package for intravoxel incoherent motion (IVIM) analysis of diffusion MRI data.
It is designed in a modular fashion where all functions involving processing of data has nifti files as input/output.
This allows for setting up pipelines where functionality from well-known diffusion MRI packages like 
`FSL <https://fsl.fmrib.ox.ac.uk/fsl/docs/#/>`_ and `MRTrix3 <https://www.mrtrix.org/>`_
(Gibbs removal, motion correction, etc) can be combined with IVIM specific functionality available in the current
package. The file formats for diffusion MR images are those of FSL (.nii.gz, .bval, .bvec), with extensions in line
with this where needed (e.g. .cval for c-values describing velocity encoding).

Here you find the documentation of release |release|, with last update |today|.

.. note::
   The package consistently use units mm, s and T. This is in line with the typical units for diffusivity (mm\ :sup:`2`/s)
   and b-value (s/mm\ :sup:`2`). Extending to velocities and c-value this gives suitable units of mm/s and s/mm,
   respectively. It does imply that sequence timing must be given in s rather than ms, and most importantly that 
   gradient strengths are given in T/mm rather than mT/m. The typical user would however rarely deal with the 
   functions involving gradients strengths are input/output.

.. toctree::
   :caption: Getting started
   :maxdepth: 1

   installation
   examples

.. toctree::
   :caption: Detailed information
   :maxdepth: 1

   models
   seqs
   optimize
   sim
   io
   preproc
   fit
   examples_additional