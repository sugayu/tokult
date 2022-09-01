.. tokult documentation master file, created by
   sphinx-quickstart on Fri Aug 26 19:44:39 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Tokult
======

Tool of Kinematics on uv-plane for Lensed Targets
=================================================


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

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   gettingstarted
   visualization

.. toctree::
   :maxdepth: 2
   :caption: Modules:

   modules
   tokult

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
