**************************************
Model fitting
**************************************

The process of estimating IVIM parameters from a set of diffusion data
has proved to be complicated in the presence of noise, resulting in
several suggested algorithm for model fitting. Three of the most commonly
used algorithm are implemented here: 1) non-linear least squares (NLLS), 
2) segmented model fitting, and 3) Bayesian model fitting.\ [1]_\ :sup:`,` [2]_

NLLS estimates all IVIM parameters of the particular model directly by 
minimizing the squared residuals between the data and the model predictions.
The algorithm builds on the scipy function `curve_fit 
<https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html>`_
which in turn applies a `"Trust Region Reflective" 
<https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares>`_
algorithm (:func:`API <ivim.fit.nlls>`).

A segmented algorithm for model fitting has been developed specifically 
for IVIM for improved robustness based on the prior knowledge that D*>>D.
With this in mind one can assume that for some threshold b-value (b\ :sub:`thr`)
the signal from the perfusion comparment is of negligible size (:math:`F_P \approx 0`).
Accordingly, the algorithm is split into two main steps. First, the model
corresponding to the diffusion comparment is fitted to data with (:math:`b\geq b_{thr}`)
to estimate the diffusion-related parameters and an extrapolated signal at 
b = 0. In a second step, these parameter estimates are fixed when a complete
model is fitted to data for all b-values. The specific algorithm implemented
here is based on analytic solutions of the least squares problem (:func:`API <ivim.fit.seg>`).\ [3]_ 

Bayesian algorithm introduces the ability to include prior knowledge and has
proved to be more robust in the case of IVIM model fitting.\ [1]_\ :sup:`,` [2]_
The algorithm implemented here builds on Markov-Chain Monte Carlo, which numerically
samples the posterior parameter distribution, and also provides the option to 
take information from neighboring voxels into accound (:func:`API <ivim.fit.seg>`).\ [2]_

References

.. [1] Barbieri S, Donati O, Froehlich J, Thoeny H. Impact of the calculation algorithm
    on biexponential fitting of diffusion-weighted mri in upper abdominal organs.
    Magnetic Resonance in Medicine. 2016;75(5):2175-2184. https://doi.org/10.1002/mrm.25765
.. [2] Jalnefjord O, Björkman-Burtscher I. Comparison of methods for intravoxel incoherent 
    motion parameter estimation in the brain from flow-compensated and non-flow-compensated 
    diffusion-encoded data. Magnetic Resonance in Medicine. 2024;92(1):303-318. 
    https://doi.org/10.1002/mrm.30042
.. [3] Jalnefjord O, Andersson M, Montelius M, Starck G, Elf A-K, Johanson V, Svensson J, 
    Ljungberg M. Comparison of methods for estimation of the intravoxel incoherent motion 
    (IVIM) diffusion coefficient (D) and perfusion fraction (f). Magnetic Resonance Materials 
    in Physics, Biology and Medicine. 2018;31:715-723. https://doi.org/10.1007/s10334-018-0697-5

API
===

.. autofunction:: ivim.fit.nlls
.. autofunction:: ivim.fit.seg
.. autofunction:: ivim.fit.bayes

