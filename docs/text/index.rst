.. Double Pendulum documentation master file, created by
   sphinx-quickstart on Mon Nov 21 11:08:29 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Double Pendulum's documentation!
===========================================

|pic1| |pic2|

.. |pic1| image:: ../figures/chaotic_freefall_long_exposure_shot.jpg
   :width: 45%

.. |pic2| image:: ../figures/double_pendulum_animation.gif
   :width: 45%

This project offers an open-source and low-cost kit to test control algorithms
for underactuated robots with strongly non-linear dynamics. It implements a
double pendulum platform built using two quasi-direct drive actuators (QDDs).
Due to low friction and high mechanical transparency offered by QDDs, one of
the actuators can be kept passive and be used as an encoder. When the shoulder
motor is passive and elbow motor is active, the system serves as an acrobot and
when the shoulder is active and elbow is passive, the system serves as a
pendubot setup. This project describes the offline and online control methods
which can be studied using this dual purpose kit, lists its components,
discusses best practices for implementation, presents results from experiments
with the simulator and the real system. This repository describes the hardware
(CAD, Bill Of Materials (BOM) etc.) required to build the physical system and
provides the software (URDF models, simulation and controller) to control it.

.. image:: ../figures/Logo_Underactuated_Lab.*
   :width: 100%
   :align: center

.. toctree::
    :maxdepth: 3
    :caption: Table of Contents

    modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
