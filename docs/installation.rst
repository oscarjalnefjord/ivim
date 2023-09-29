**************************************
Installation
**************************************

The suggested way to install is to first prepare an environment suitable for the `ivim` package using e.g. conda with a compatible python version (>= 3.10):

.. code-block:: bash
    
    conda create -n ivim python == 3.10

Activate the environment and install the prerequisities:

.. code-block:: bash
    
    conda activate ivim
    conda install -c anaconda -c conda-forge nibabel scipy numpy

Download the package with:

.. code-block:: bash

    git clone https://github.com/oscarjalnefjord/ivim.git

To install the package, run:

.. code-block:: bash
    
    pip install -e ivim

If you are not in the directory of the repository, run:

.. code-block:: bash
    
    pip install -e /path/to/ivim

where `/path/to/ivim` points to the directory of the repository.