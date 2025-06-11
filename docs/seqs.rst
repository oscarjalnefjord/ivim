**************************************
Pulse sequences
**************************************

The package mainly revolves around two pulse sequences for diffusion 
encoding with either 

#. monopolar, or 

#. bipolar diffusion encoding gradients.

.. image:: figs/seqs.png

For these pulse sequences one can calculate the b-value (:func:`API <ivim.seq.lte.calc_b>`)

.. math:: b = p \gamma^2G^2\delta^2(\Delta-\frac{\delta}{3})

where p takes values 1 for a monopolar pulse sequence and 2 for a bipolar pulse sequence,
:math:`\gamma` is the gyromagnetic ratio, G is the gradient strength, and :math:`\delta`
and :math:`\Delta` are the duration and separation of the gradient pulses.\ [1]_

In a similar way, the c-value (flow sensitivity) can be calculated (:func:`API <ivim.seq.lte.calc_c>`)

.. math:: c = p \gamma G\delta\Delta

Note that the equation above applies for non-flow-compensated diffusion encoding.\ [2]_\ 
:sup:`,`\ [3]_ For a flow-compensated diffusion encoding, c = 0 by design.\ [4]_

The module also provides functions for calculating the gradient strength given a b-value (:func:`API <ivim.seq.lte.G_from_b>`)
and generation of cval files similar to the commonly used bval files (:func:`API <ivim.seq.lte.cval_from_bval>`).

References:

.. [1] Le Bihan D, Breton E, Lallemand D, Grenier P, Cabanis E, Laval-Jeantet M. MR imaging 
    of intravoxel incoherent motions: application to diffusion and perfusion in neurologic 
    disorders. Radiology. 1986;161(2):401-407. doi: https://doi.org/10.1148/radiology.161.2.3763909
.. [2] Le Bihan D, Breton E, Lallemand D, Aubin ML, Vignaud J, Laval-Jeantet M. Separation of diffusion 
    and perfusion in intravoxel incoherent motion MR imaging. Radiology. 1988;168(2):497-505. doi: 
    https://doi.org/10.1148/radiology.168.2.3393671 
.. [3] Kennan RP, Gao JH, Zhong J, Gore JC. A general model of microcirculatory blood flow effects in 
    gradient sensitized MRI. Medical Physics. 1994;21(4):539-45. doi: https://doi.org/10.1118/1.597170
.. [4] Ahlgren A, Knutsson L, Wirestam R, Nilsson M, Ståhlberg F, Topgaard D, Lasic S. Quantification 
    of microcirculatory parameters by joint analysis of flow-compensated and non-flow-compensated 
    intravoxel incoherent motion (IVIM) data. NMR in Biomedicine. 2016;29(5):640-649. doi: 
    https://doi.org/10.1002/nbm.3505 

API
===
``ivim.seqs.lte``

.. autofunction:: ivim.seq.lte.calc_b
.. autofunction:: ivim.seq.lte.calc_c
.. autofunction:: ivim.seq.lte.G_from_b
.. autofunction:: ivim.seq.lte.cval_from_bval
