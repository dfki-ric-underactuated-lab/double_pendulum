Installation
=========================

.. toctree::
   :maxdepth: 3

The code was tested for Python >=3.8 and this is what we recommend. 

Note: If you want to work with a version Python<3.8, that is mostly possible at
the moment. Only the moteus python package needs to be commented in the
setup.py and it will not be possible to operate mjbots hardware.

We recommend using a virtual python environment.

1. Install dependencies::

    # ubuntu20
    sudo apt-get install git unzip libyaml-cpp-dev libeigen3-dev libpython3.8 libx11-6 libsm6 libxt6 libglib2.0-0 python3-pip python3-sphinx python3-numpydoc python3-sphinx-rtd-theme

    # ubuntu22
    sudo apt-get install git unzip libyaml-cpp-dev libeigen3-dev libx11-6 libsm6 libglib2.0-0 libboost-dev python3-pip python3-sphinx python3-numpydoc python3-sphinx-rtd-theme

2. The header only eigen library can be installed via (For more information
   visit `eigen<https://eigen.tuxfamily.org/index.php?title=Main_Page>`__)::

        wget -O Eigen.zip https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
        unzip Eigen.zip
        sudo mv eigen-*/Eigen /usr/local/include/

3. Clone the repository::

    git clone git@github.com:dfki-ric-underactuated-lab/double_pendulum.git

4. Install the double pendulum package::
   
    cd double_pendulum
    make install

With this you are done and you can use all functionalities and some controllers in this repository.
If you want install all requirements (this includes also larger depoendencies
such as tensorflow, drake etc.) for all controllers do

4. (Optional) Full installation ::
  
     make pythonfull

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
