Installation
=========================

.. toctree::
   :maxdepth: 3

The code was tested for Python >=3.8 and this is what we recommend. 

Note: If you want to work with a version Python<3.8, that is mostly possible at
the moment. Only the moteus python package needs to be commented in the
setup.py and it will not be possible to operate mjbots hardware.

We recommend using a virtual python environment.

Manual
------

1. Install dependencies::

    # ubuntu20
    sudo apt-get install git unzip libyaml-cpp-dev libeigen3-dev libpython3.8 libx11-6 libsm6 libxt6 libglib2.0-0 python3-pip python3-sphinx python3-numpydoc python3-sphinx-rtd-theme ffmpeg

    # ubuntu22
    sudo apt-get install git unzip libyaml-cpp-dev libeigen3-dev libx11-6 libsm6 libglib2.0-0 libboost-dev python3-pip python3-sphinx python3-numpydoc python3-sphinx-rtd-theme ffmpeg

2. (Optional, only for C++ support, iLQR and PRX controllers) Install `Eigen <https://eigen.tuxfamily.org/index.php?title=Main_Page>`__::

        wget -O Eigen.zip https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
        unzip Eigen.zip
        sudo mv eigen-*/Eigen /usr/local/include/

3. (Optional, only for Acados controller) Install `Acados <https://docs.acados.org/>`__::

        git clone https://github.com/acados/acados.git
        cd acados
        git submodule update --recursive --init
        mkdir -p build
        cd build
        cmake -DACADOS_WITH_QPOASES=ON ..
        make install
        pip install -e acados/interfaces/acados_template

4. Clone the repository::

    git clone git@github.com:dfki-ric-underactuated-lab/double_pendulum.git

5. Install the double pendulum package::
   
    cd double_pendulum
    make install

Using Conda/Mamba
-----------------

Another option to install and use dependencies would be using conda/`mamba <https://github.com/mamba-org/mamba>`__ environment.

1. Clone the repository::

    git clone git@github.com:dfki-ric-underactuated-lab/double_pendulum.git

2. Create a new conda environment::

    cd double_pendulum
    mamba env create --file=environment.yaml

3. Activate the environment and install the double pendulum package::

    mamba activate double_pendulum
    make install

With this you are done and you can use all functionalities and some controllers in this repository.
If you want to install all requirements (this includes also larger dependencies
such as tensorflow, drake etc.) for all controllers do

4. (Optional) Full installation ::
  
     make pythonfull

Individual Controllers
----------------------

To install dependencies for individual controllers do::

    cd src/python
    pip install .[<name>]

where you replace <name> with the controller name. Have a look in the 
`setup.py <https://github.com/dfki-ric-underactuated-lab/double_pendulum/blob/main/src/python/setup.py>`__
for the controller names.
If you want to install all dependencies for all controllers do::

    cd src/python
    pip install .[all]

Partial Installations
---------------------

The following two bullet points may be useful when only
the C++/Python code is needed, if you work on the code and need to recompile
often, or if errors appear during the installation call in 4.

4. If you want to install only the Python package, you can do::

    make python
    # make pythonfull

    # or

    cd src/python
    pip install .
    # pip install .[all]

5. If you want to build C++ code and install python bindings::

    make cpp

    # or

    cd src/cpp/python
    make
