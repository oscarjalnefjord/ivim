**************************************
Preprocessing
**************************************

The package provides a set of functions useful for preprocessing 
of data, both general and some specific (currently limited to 
signal drift):

General
=======
* Averaging across diffusion encoding direction (:func:`API <ivim.preproc.base.average>`)
* Combining imaging volume from separate files (:func:`API <ivim.preproc.base.combine>`)
* Extracting a subset of frames from an imaging volume (:func:`API <ivim.preproc.base.extract>`)

Signal drift
============
* Correcting for signal drift with a global correction (:func:`API <ivim.preproc.signal_drift.temporal>`)\ [1]_
* Correcting for signal drift with a spatiotemporal correction (:func:`API <ivim.preproc.signal_drift.spatiotemporal>`)\ [2]_

Both algorithms for signal drift correction relies on acquisition of multiple b=0 imaging volumes
distributed throughout the protocol.

References

.. [1] Vos S, Tax C, Luijten P, Ourselin S, Leemans A, Froeling M. The Importance of 
    Correcting for Signal Drift in Diffusion MRI. Magnetic Resonance in Medicine 2017;77(1):285-299
    https://doi.org/10.1002/mrm.26124
.. [2] Hansen C, Nath V, Hainline A, Schilling K, Parvathaneni P, Bayrak R, Blaber J, Irfanoglu O, 
    Pieropaoli C, Anderson A, Rogers B, Landman B. Characterization and correlation of signal Drift 
    in diffusion weighted MRI. Magnetic Resonance Imaging 2019;57:133-142. 
    https://doi.org/10.1016/j.mri.2018.11.009

API
===
.. autofunction:: ivim.preproc.base.average
.. autofunction:: ivim.preproc.base.combine
.. autofunction:: ivim.preproc.base.extract
.. autofunction:: ivim.preproc.signal_drift.temporal
.. autofunction:: ivim.preproc.signal_drift.spatiotemporal