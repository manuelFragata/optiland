.. _installation:

Installation
============

Optiland can be installed via `pip` or directly from source. Additional optional dependencies provide enhanced functionality.


Basic Installation
------------------

Install the core Optiland package, excluding optional dependencies:

   .. code-block:: console

      pip install optiland


Installation with PyTorch (CPU-only)
------------------------------------

Install Optiland along with the PyTorch backend for CPU-only operations:

   .. code-block:: console

      pip install optiland[torch]


Installation with PyTorch (GPU support)
---------------------------------------

For GPU acceleration, first manually install PyTorch with CUDA following the instructions at: `PyTorch Get Started <https://pytorch.org/get-started/locally/>`_

Once PyTorch is installed, install Optiland:

   .. code-block:: console

      pip install optiland


Installing from Source
----------------------

To install Optiland from source, follow these steps:

1. Clone the repository from GitHub:

   .. code-block:: console

      git clone https://github.com/HarrisonKramer/optiland.git
      cd optiland

2. Install Optiland and its dependencies:

   .. code-block:: console

      pip install .


Verifying Installation
----------------------

After installation, verify Optiland by importing it in Python:

.. code-block:: python

   import optiland

Optionally, you can generate and visualize a lens system to confirm installation:

.. code-block:: python

   from optiland.samples.objectives import ReverseTelephoto

   lens = ReverseTelephoto()
   lens.draw3D()
