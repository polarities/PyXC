===========================
Contribute to this project
===========================

Setting up the development environment
--------------------------------------

Execute following codes in your desired folder location.

.. code-block:: bash

   git clone https://github.com/polarities/PyXC.git
   cd PyXC
   python -v venv ./venv
   source ./venv/bin/activate
   pip install .[development]

Setting up the pre-commit hooks
-------------------------------

Please install the pre-commit hooks to ensure the code quality and consistency. Currently the pre-commit hook will run the following checks:

1. Code formatting with black (*.py files under the ./pyxc and ./tests folders & *.ipynb files under the ./docs/source folder).
2. Removing trailing whitespaces.
3. Cleaning notebook outputs and metadata.
4. Checking NumPy-styled docstring validity.

.. code-block:: bash

   pre-commit install

Compiling the documentation
---------------------------

You can compile the documentation by executing the following command in the root folder of this project.

.. code-block:: bash

   python -m sphinx -T -E -b html -d ./docs/build/doctrees -D language=en ./docs/source ./docs/build/html
