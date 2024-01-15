Repository Structure
====================

Programming Language
--------------------

The main programming language of this library is python, but other languages
are also welcome. Besides python, there is a plant and a simulator as well as
an iLQR solver written in C++. The iLQR solver has python bindings so that
it can be used with the other components.

If you want to contribute something please use/create the folder in the ``src``
directory with the name of the used programming language.

Repository Overview
-------------------

The following overview covers the python code and the code which has python
bindings to be used in the same ecosystem. The repository is organized in
modules. The interplay of the modules is visualized in the figure below:

.. image:: ../figures/repository_structure.png
  :width: 100%
  :align: center

There are standardized interfaces and file formats for the communication
between the modules which are explained in the following.

For more details see also:

.. toctree::
   :maxdepth: 1

   software_structure.modelparameters.rst 
   software_structure.trajectorydata.rst 
   software_structure.controller.rst 
