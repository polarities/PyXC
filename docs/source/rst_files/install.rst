=======================
PyXC Installation Guide
=======================

Requirements
------------

- `PyXC` is compatible with Python binaries of version `3.10` or newer.

Dependencies
------------

.. NOTE

For setting up a development environment, see :doc:Contributing Guide <./contributing>.

- NumPy
- SciPy

Installation
------------

Using pip
^^^^^^^^^

.. NOTE

Currently `PyXC` is not available on PyPI. You can install it from source.

.. code-block::

   pip install pyxc

From source
^^^^^^^^^^^

Execute the following codes in your desired folder location

.. code-block:: bash

   git clone https://github.com/polarities/PyXC.git
   cd PyXC
   python -v venv ./venv
   source ./venv/bin/activate
   pip install .
