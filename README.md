<div align="center">

# Dual Purpose Acrobot & Pendubot Platform 
</div>

<div align="center">
<img width="300" src="hardware/images/double_pendulum_CAD.png">
<img width="290" src="docs/figures/double_pendulum_animation.gif" />
<img width="328" src="docs/figures/chaotic_freefall_long_exposure_shot.jpg">
</div>

## Introduction #
This project offers an open-source and low-cost kit to test control algorithms
for underactuated robots with strongly non-linear dynamics. It implements a
**double pendulum** platform built using two quasi-direct drive actuators
(QDDs). Due to low friction and high mechanical transparency offered by QDDs,
one of the actuators can be kept passive and be used as an encoder. When the
shoulder motor is passive and elbow motor is active, the system serves as an
**acrobot** and when the shoulder is active and elbow is passive, the system
serves as a **pendubot** setup. This project describes the _offline_ and
_online_ control methods which can be studied using this dual purpose kit,
lists its components, discusses best practices for implementation, presents
results from experiments with the simulator and the real system. This
repository describes the hardware (CAD, Bill Of Materials (BOM) etc.) required
to build the physical system and provides the software (URDF models, simulation
and controller) to control it.

<!---
## Documentation

The [hardware setup](hardware/testbench_description.md) and the [motor
configuration](hardware/motor_configuration.md) are described in their
respective readme files.  The dynamics of the pendulum are explained
[here](docs/sp_equations.md).

In order to work with this repository you can [get started
here](docs/installation_guide.md) and read the [usage instructions
here](docs/usage_instructions.md) for a description of how to use this
repository on a real system. The instructions for testing the code can be found
[here](docs/code_testing.md).


* [Hardware & Testbench Description](hardware/testbench_description.md)
* [Motor Configuration](hardware/motor_configuration.md)
* [Software Installation Guide](docs/installation_guide.md)
* [Usage Instructions](docs/usage_instructions.md)
* [Code Testing](docs/code_testing.md)
-->

## Installation

Install dependencies:

    sudo apt-get install libyaml-cpp-dev

To install this project, clone this repository and do

    make install

in the main directory. We recommend using a virtual python environment.
For more details, we refer to the [documentation]().

## Repository Structure

The repository is organized in modules. The interplay of the modules is
visualized in the figure below:

<div align="center">
<img width="800" src="docs/figures/repository_structure.png">
</div>

## Benchmark Results

The following plot shows benchmark results obtained in simulation for the
double pendulum design C.0 and model parameters h1.1.  The results show the
robustness of the controllers to

- Inaccuracies in the model parameters
- Noise in the velocity measurements
- Noise in the applied motor torques
- Imperfect step response from the motors
- A time delay in the state measurements

The controllers achieve a score between 0 and 1 in each category which is the
fraction of successful swing-up motions under varying conditions.

<div align="center">
<img width="800" src="docs/figures/benchmark_scores_C.0.h1.1.png">
</div>

## Authors #

* [Shivesh Kumar](https://robotik.dfki-bremen.de/en/about-us/staff/shku02.html) (Project Supervisor)
* [Felix Wiebe](https://robotik.dfki-bremen.de/en/about-us/staff/fewi01.html) (Software Maintainer)
* [Mahdi Javadi](https://robotik.dfki-bremen.de/en/about-us/staff/maja04/) (Hardware Maintainer)
* [Jonathan Babel](https://robotik.dfki-bremen.de/en/about-us/staff/joba02.html) 
* [Lasse Maywald](https://robotik.dfki-bremen.de/en/about-us/staff/lama02/)
* [Heiner Peters](https://robotik.dfki-bremen.de/en/about-us/staff/hepe02.html)
* [Shubham Vyas](https://robotik.dfki-bremen.de/en/about-us/staff/shvy01/)
* [Melya Boukheddimi](https://robotik.dfki-bremen.de/en/about-us/staff/mebo01/)

Feel free to contact us if you have questions about the test bench. Enjoy!

## Contributing

1. Fork it (<https://github.com/yourname/yourproject/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request

See [Contributing](CONTRIBUTING.md) for more details.

## Safety Notes #

When working with a real system be careful and mind the following safety
measures:

* Brushless motors can be very powerful, moving with tremendous force and
  speed. Always limit the range of motion, power, force and speed using
  configurable parameters, current limited supplies, and mechanical design.

* Stay away from the plane in which double pendulum is swinging. It is
  recommended to have a safety cage surrounding the double pendulum in case the
  parts of the pendulum get loose and fly away.

* Make sure you have access to emergency stop while doing experiments. Be extra
  careful while operating in pure torque control loop.

## Acknowledgements #
This work has been performed in the M-RoCK project funded by the German
Aerospace Center (DLR) with federal funds (Grant Number: FKZ 01IW21002) from
the Federal Ministry of Education and Research (BMBF) and is additionally
supported with project funds from the federal state of Bremen for setting up
the Underactuated Robotics Lab (Grant Number: 201-001-10-3/2021-3-2).

## License

This work has been released under the BSD 3-Clause License. Details and terms
of use are specified in the LICENSE file within this repository. Note that we
do not publish third-party software, hence software packages from other
developers are released under their very own terms and conditions. If you
install third-party software packages along with this repo ensure  that you
follow each individual license agreement.   

