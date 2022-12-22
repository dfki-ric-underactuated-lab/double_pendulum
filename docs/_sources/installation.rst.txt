Installation
=========================

.. toctree::
   :maxdepth: 3

The code was tested for Python >=3.8 and this is what we recommend. 

Note: If you want to work with a version Python<3.8, that is mostly possible at
the moment. Only the moteus python package needs to be commented in the
setup.py and it will not be possible to operate mjbots hardware.

We recommend using a virtual python environment.

1. Install dependencies (only neccessary for C++ code, used for iLQR optimization)::

    # ubuntu20
    sudo apt-get install libyaml-cpp-dev libeigen3-dev libpython3.8 libx11-6 libsm6 libxt6 libglib2.0-0 python3-sphinx python3-numpydoc python3-sphinx-rtd-theme

    # ubuntu22
    sudo apt-get install libyaml-cpp-dev libeigen3-dev libx11-6 libsm6 libglib2.0-0 python3-sphinx python3-numpydoc python3-sphinx-rtd-theme


2. Clone the repository::

    git clone git@git.hb.dfki.de:underactuated-robotics/release_version/caprr-release-version.git


3. Install the double pendulum package::
   
    cd caprr-release-version
    make install


With this you are done. The following two bullet points may be useful when only
the C++/Python code is needed, if you work on the code and need to recompile
often, or if errors appear during the installation call in 3. 

4. If you want to install only the Python package, you can do::

    make python

    # or

    cd src/python
    pip install .

5. If you want to build C++ code and install python bindings::

    make cpp

    # or

    cd src/cpp/python
    make

Potential Issues
----------------

If you have the issue that the eigen library is not correctly installed you can
install the header only library with::

    wget -O Eigen.zip https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
    unzip Eigen.zip
    cp -r eigen-3.4.0/Eigen /usr/local/include

For more information visit `eigen
<https://eigen.tuxfamily.org/index.php?title=Main_Page>`__ .
