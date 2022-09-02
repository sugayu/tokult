Tool of Kinematics on uv-plane for Lensed Targets
=================================================

Tokult is a tool to analyze the kinematics of galaxies.

Tokult is created to fit the 3D data cubes, especially ALMA cube images,
but applicable to other IFU data formats.

Requirements
------------

Required Python version:

-  `Python <https://www.python.org>`__ 3.9 or later

.. warning::

   This code does not work with Python <= 3.8 and Python 2.

Modules:

-  `NumPy <https://numpy.org>`__
-  `SciPy <https://scipy.org>`__
-  `Astropy <https://www.astropy.org>`__
-  `emcee <https://emcee.readthedocs.io/en/stable/>`__ 3.0.0 or later
-  `tqdm <https://tqdm.github.io>`__

Installation
------------

You can install Tokult from the github repository using ``pip``:

.. code:: bash

   $ pip install git+https://github.com/sugayu/tokult.git@dev

Before using this command, it may be good to create a new Python
environment for Tokult.

To update the package, you may need to add the ``--upgrade`` option.

.. code:: bash

   $ pip install --upgrade git+https://github.com/sugayu/tokult.git@dev

.. note::

   This command installs the program code from the branch "dev" (=
   development). We will remove ``@dev`` when the code is open to
   public.
