**************************************
Protocol optimization
**************************************

The precision and accuracy of estimated IVIM parameters depend on the 
imaging parameters, in particular choice of b-values and similar parameters.
Several methods for protocol optimization exist, all with the pros and cons.
The algorithm for protocol optimization provided in this package builds on 
minimization of the Cramér-Rao Lower Bound (CRLB), which is a measure of the 
theoretically lowest variance of an estimated parameter given an experimental
setup.\ [1]_ The algorithm implemented in this package has been used to optimize
b-values for the simplified IVIM model.\ [2]_ The function here does however
provide compatibility with models in all regimes, and both monopolar and bipolar
diffusion encoding (:func:`API <ivim.optimize.crlb>`).

References:

.. [1] Alexander D. A general framework for experiment design in diffusion MRI 
    and its application in measuring direct tissue-microstructure. Magnetic 
    Resonance in Medicine. 2008;60(2):439-448. https://doi.org/10.1002/mrm.21646
.. [2] Jalnefjord O, Montelius M, Starck G, Ljungberg M. Optimization of b-value 
    schemes for estimation of the diffusion coefficient and the perfusion fraction 
    with segmented intravoxel incoherent motion model fitting. Magnetic Resonance
    in Medicine. 2019:82(4):1541-1552. https://doi.org/10.1002/mrm.27826

API
===
``ivim.optimize``

.. automodule:: ivim.optimize
.. autofunction:: ivim.optimize.crlb