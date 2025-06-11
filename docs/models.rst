**************************************
Models
**************************************

The models implemented in this package build on the general two-compartment IVIM model

.. math:: S = S_0 ((1-f)F_D + fe^{-bD_b}F_P)

where S and S\ :sub:`0` are the signal with and without diffusion weighting, respectively, 
f is the signal fraction of the perfusion compartment, F\ :sub:`D` describes the signal decay due 
to diffusion weighting in the diffusion (extravascular) compartment, b is the b-value, D\ :sub:`b` 
is the diffusion coefficient of blood, and F\ :sub:`P` describes the signal decay due to diffusion 
weighting in the perfusion (vascular) compartment.\ [1]_\ :sup:`,`\ [2]_

By default is it assumed that sufficiently small b-values are used such that F\ :sub:`D` can be
decribed by a monoexponential (:func:`API <ivim.models.monoexp>`)

.. math:: F_D = e^{-bD}

although there is support for the kurtosis expression (:func:`API <ivim.models.kurtosis>`) if higher b-values are of interest

.. math:: F_D = e^{-bD+b^2D^2K/6}

where D is the extravascular diffusion coefficient, and K is the kurtosis koefficient.\ [3]_

The expression for F\ :sub:`P` depends on the what temporal regime is assumed and choice of 
pulse sequence. Currently, the code supports models in the ballistic (short encoding time),
diffusive (long encoding time), and intermediate (short to long encoding time) regimes. The 
specific expression for F\ :sub:`P` in the ballistic regime (:func:`API <ivim.models.ballistic>`) is 

.. math:: F_P = e^{-c^2v^2/6}

where c describes the velocity encoding and v is the root-mean-squared velocity of blood.\ [4]_ The 
expression for F\ :sub:`P` in the diffusive regime (:func:`API <ivim.models.diffusive>`) is

.. math:: F_P = e^{-bD^*}

where D* is the pseudo-diffusion coefficient.\ [1]_ The expression in the intermediate regime 
(:func:`API <ivim.models.intermediate>`) is

.. math:: F_P = \exp{-\gamma^2G^2v^2\frac{\tau}{3}p[\tau^3(\Psi+\Omega) - 2\delta\tau^2 + \delta^2(\Delta-\frac{\delta}{3})]}

.. math:: \Psi = e^{-(\Delta+\delta)/\tau}[e^{\delta/\tau}-1][2e^{\Delta/\tau}+e^{\delta/\tau}-1]

.. math:: 
    
    \Omega = \begin{cases}
    0 & \text{Monopolar} \\
    \frac{k}{p}[e^{\Delta/\tau}-1]^2[e^{\delta/\tau}-1]^2e^{-T/\tau} & \text{Bipolar}
    \end{cases}

where :math:`\gamma` is the gyromagnetic ratio, G is the gradient strength, tau is the correlation time
of the motion of blood, p takes values 1 for a monopolar pulse sequence and 2 for a bipolar pulse sequence,
k takes values -1 and 1 for flow-compensated and non-flow-compensated diffusion encoding, and T is the total 
encoding time (start of first diffusion encoding gradient to end of last one).\ [5]_

Additionally, if b-values are either zero or sufficiently large such that F\ :sub:`P` is approximately
zero, a simplified expression can be used (:func:`API <ivim.models.sIVIM>`)

.. math::

   F_P = \begin{cases}
   1 & b = 0 \\
   0 & b \geq b_{thr}
   \end{cases}

where b\ :sup:`thr` is the threshold b-value where :math:`F_P \approx 0`.\ [1]_

Analytical jacobian matrices are available for all models (:func:`API <ivim.models.monoexp_jacobian>`).

References:

.. [1] Le Bihan D, Breton E, Lallemand D, Aubin ML, Vignaud J, Laval-Jeantet M. Separation of diffusion 
    and perfusion in intravoxel incoherent motion MR imaging. Radiology. 1988;168(2):497-505. doi: 
    https://doi.org/10.1148/radiology.168.2.3393671 
.. [2] Le Bihan D. What can we see with IVIM MRI? Neuroimage. 2019;187:56-67. doi: 
    https://doi.org/10.1016/j.neuroimage.2017.12.062
.. [3] Iima M, Le Bihan D. Clinical Intravoxel Incoherent Motion and Diffusion MR Imaging: Past, 
    Present, and Future. Radiology. 2016;278(1):13-32. doi: https://doi.org/10.1148/radiol.2015150244
.. [4] Kennan RP, Gao JH, Zhong J, Gore JC. A general model of microcirculatory blood flow effects in 
    gradient sensitized MRI. Med Phys. 1994;21(4):539-45. doi: https://doi.org/10.1118/1.597170
.. [5] Rosenqvist L, Ljungberg M, Jalnefjord O. Modelling the intermediate flow regime in flow-compensated 
    intravoxel incoherent motion MRI. Annual meeting ISMRM. 2024; p0647.

API
===
``ivim.models``

.. automodule:: ivim.models
.. autofunction:: ivim.models.sIVIM
.. autofunction:: ivim.models.diffusive
.. autofunction:: ivim.models.ballistic
.. autofunction:: ivim.models.intermediate
.. autofunction:: ivim.models.monoexp
.. autofunction:: ivim.models.kurtosis
.. autofunction:: ivim.models.monoexp_jacobian
.. autofunction:: ivim.models.kurtosis_jacobian
.. autofunction:: ivim.models.sIVIM_jacobian
.. autofunction:: ivim.models.diffusive_jacobian
.. autofunction:: ivim.models.ballistic_jacobian
.. autofunction:: ivim.models.intermediate_jacobian