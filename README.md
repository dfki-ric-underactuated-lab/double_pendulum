<div align="center">

# Dual Purpose Acrobot & Pendubot Platform
</div>

<div align="center">
<img width="500" src="docs/figures/chaotic_freefall_long_exposure_shot.jpg">
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

<div align="center">
<img width="300" src="hardware/images/double_pendulum_CAD.png">
<img width="290" src="docs/figures/double_pendulum_animation.gif" />
</div>

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

We recommend using a virtual python environment with version >=3.8.10.

Install dependencies:

    # ubuntu20
    sudo apt-get install libyaml-cpp-dev libeigen3-dev libpython3.8 libx11-6 libsm6 libxt6 libglib2.0-0 python3-sphinx python3-numpydoc python3-sphinx-rtd-theme

    # ubuntu22
    sudo apt-get install libyaml-cpp-dev libeigen3-dev libx11-6 libsm6 libglib2.0-0 python3-sphinx python3-numpydoc python3-sphinx-rtd-theme

Update pip:

    pip install -U pip

To install this project, clone this repository and do

    make install

in the main directory.
For more details, we refer to the [documentation]().

## Documentation

The [documentation]() for this project can be found [here]().

The documentation can also be generated locally by doing

    make doc

in the main directory. Afterwards you can open the file
<mark>
    docs/build/_build/html/index.html
</mark>
in your browser.

## Repository Structure

The repository is organized in modules. The interplay of the modules is
visualized in the figure below:

<div align="center">
<img width="800" src="docs/figures/repository_structure.png">
</div>

A minimal example showcasing the interfaces between the plant, a controller and
the simulator is:

    from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
    from double_pendulum.simulation.simulation import Simulator
    from double_pendulum.controller.pid.point_pid_controller import PointPIDController
    from double_pendulum.utils.plotting import plot_timeseries

    plant = SymbolicDoublePendulum(mass=[0.6, 0.5],
                                   length=[0.3, 0.2])

    sim = Simulator(plant=plant)

    controller = PointPIDController(torque_limit=[5.0, 5.0])
    controller.set_parameters(Kp=10, Ki=1, Kd=1)
    controller.init()
    T, X, U = sim.simulate_and_animate(t0=0.0, x0=[2.8, 1.2, 0., 0.],
                                       tf=5., dt=0.01, controller=controller)
    plot_timeseries(T, X, U)

This code snippet uses a plant which is parsed to the simulator. The simulator
is then used to simulate and animate the double pendulum motion by calling the

    plant.rhs(t, x, tau)

with a time t = float, state x = [float, float, float, float] and torque
tau = [float, float] and receiving the change in x in form of its derivative.

The simulation is performed for 5 seconds under the influence of a PID
controller by at every time step calling the

    controller.get_control_output(x, t)

method of the controller with a state x = [float, float, float, float] and a
time t = float and receiving a torque tau.

More examples can be found in the [examples](examples) folder.

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

Contributions to this project, especially in the form of new controllers, are
very welcome!
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

