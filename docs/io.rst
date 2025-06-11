**************
Input/Output
**************

The package provides a set of functions useful for input/output 
of data, both general and vendorspecific (currently limited to 
Philips given the interest of the authors):

General
=======
* Reading image and corresponding b-value etc from file (:func:`API <ivim.io.base.data_from_file>`)
* Save data to image file (:func:`API <ivim.io.base.file_from_data>`)
* Read image, preserving dimensionality in contrast to the functions above (:func:`API <ivim.io.base.read_im>`)
* Read b-value and other parameter files (:func:`API <ivim.io.base.read_bval>`)
* Save image (:func:`API <ivim.io.base.write_im>`)
* Save b-value and other parameters to file (:func:`API <ivim.io.base.write_bval>`)

Philips
=======
* Generate dti_vectors_input file for arbitrary diffusion encoding schemes (:func:`API <ivim.io.philips.generate_dti_vectors_input>`)

API
===
.. autofunction:: ivim.io.base.data_from_file
.. autofunction:: ivim.io.base.file_from_data
.. autofunction:: ivim.io.base.read_im
.. autofunction:: ivim.io.base.read_bval
.. autofunction:: ivim.io.base.read_cval
.. autofunction:: ivim.io.base.read_time
.. autofunction:: ivim.io.base.read_k
.. autofunction:: ivim.io.base.read_bvec
.. autofunction:: ivim.io.base.write_im
.. autofunction:: ivim.io.base.write_bval
.. autofunction:: ivim.io.base.write_cval
.. autofunction:: ivim.io.base.write_time
.. autofunction:: ivim.io.base.write_k
.. autofunction:: ivim.io.base.write_bvec
.. autofunction:: ivim.io.philips.generate_dti_vectors_input
.. autofunction:: ivim.io.philips.write_dti_vectors_input
.. autofunction:: ivim.io.philips.read_dti_vectors_input
