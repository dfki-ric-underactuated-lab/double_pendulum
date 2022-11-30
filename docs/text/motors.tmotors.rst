T-Motor (AK-80-6)
------------------

The T-Motor double pendulum uses the `AK80-6 Motors
<https://store.tmotor.com/goods.php?id=981>`__ from T-Motor. Here the motors'
physical parameters, the initial setup as well as the usage with the python
driver is documented. The AK80-6 manual can be found `here
<https://store.tmotor.com/images/file/202208/251661393360838805.pdf>`__ .

Physical Parameters of the Actuator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The AK80-6 actuator from T-Motor is a quasi direct drive with a gear
ratio of :math:`\small{6:1}` and a peak torque of
:math:`\small{12\,Nm}` at the output shaft. The motor is equipped with
an absolute :math:`\small{12}` bit rotary encoder and an internal PD
torque control loop. The motor controller is basically the same as the
one used for MIT Mini-Cheetah, which is described in the documentation
from Ben Katz. - `Ben Katz: MIT Mini-Cheetah
Documentation <https://docs.google.com/document/d/1dzNVzblz6mqB3eZVEMyi2MtSngALHdgpTaDJIW_BpS4/edit>`__

.. image:: ../../hardware/images/motor_ak80-6.jpg
   :width: 100%
   :align: center

-  Voltage = :math:`\small{24\,V}`
-  Current = rated :math:`\small{12\,V}`, peak :math:`\small{24\,V}`
-  Torque = rated :math:`\small{6\,Nm}` , peak
   :math:`\small{12\,Nm}` (after the transmission)
-  Transmission :math:`\small{N = 6 : 1}`
-  Weight = :math:`\small{485\,g}`
-  Dimensions = :math:`\small{⌀\,\,98\,mm\times\,38.5\,mm}`.

-  Max. torque to weight ratio = :math:`\small{24\,\frac{Nm}{Kg}}`
   (after the transmission)
-  Max. velocity = :math:`\small{38.2\,\frac{rad}{s}}` =
   :math:`\small{365\,rpm}` (after the transmission)
-  Backlash (accuracy) = :math:`\small{0.15°(degrees)}`


The T-Motor Ak-80-6 has the following motor constants
(before the transmission):

-  Motor constant :math:`\small{k_m = 0.2206 \,\frac{Nm}{\sqrt{W}}}`
-  Electric constant :math:`\small{k_e = 0.009524 \,\frac{V}{rpm}}`
-  Torque constant :math:`\small{k_t = 0.091 \,\frac{Nm}{A}}`
-  Torque = rated :math:`\small{1.092\,Nm}`, peak
   :math:`\small{2.184\,Nm}`
-  Velocity / back-EMF constant
   :math:`\small{k_v = 100 \,\frac{rpm}{V}}`
-  Max. velocity at :math:`\small{24\,V}`\ =
   :math:`\small{251.2 \,\frac{rad}{s}}` =
   :math:`\small{2400 \,\,rpm}`
-  Motor wiring in :math:`\small{\nabla- configuration}`
-  Number of pole pairs = :math:`\small{21}`
-  Resistance phase to phase = :math:`\small{170\pm5\,m\Omega}`
-  Inductance phase to phase = :math:`\small{57\pm10\,m\mu H}`
-  Rotor inertia :math:`\small{Ir = 0.000060719\,Kg.m^2}`

Initial Motor Setup
~~~~~~~~~~~~~~~~~~~

The R-LINK Configuration Tool is used to configure the AK80-6 from
T-Motors. Before starting to use the R-Link device make sure you have
downloaded the ``CP210x Universal Windows Driver`` from silabs. If this
isn't working properly follow the instructions at sparkfun on how to
install ch340 drivers. You have to download the ``CH 341SER (EXE)`` file
from the sparkfun webpage. Notice that you first have to select
uninstall in the CH341 driver menu to uninstall old drivers before you
are able to install the new driver. The configuration tool software for
the R-LINK module can be downloaded on the T-Motors website.

-  **Silabs:** `CP210x Universal Windows
   Driver <https://www.silabs.com/developers/usb-to-uart-bridge-vcp-drivers>`__
-  **CH341:** `Sparkfun - How to install CH340
   drivers <https://learn.sparkfun.com/tutorials/how-to-install-ch340-drivers/all>`__

Tutorials
<<<<<<<<<

-  T-MOTOR: https://www.youtube.com/watch?v=hbqQCgebaF8
-  Skyentific: https://www.youtube.com/watch?v=HzY9vzgPZkA

Instructions: R-Link Config Tool
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

**User manual & configuration tool:**
`store-en.tmotor.com <https://store-en.tmotor.com/goods.php?id=1085>`__

R-LINK is a USB to serial port module, specially designed for CubeMars A
Series of dynamical modular motors. It is possible to calibrate the
encoder in the module, change CAN ID settings, PID settings, as well as
to control position, torque and speed of the motor within the
configuration software tool.

|pic1| |pic2|

.. |pic1| image:: ../../hardware/images/r-link_module.jpg
   :width: 44%

.. |pic2| image:: ../../hardware/images/r-link_wiring.PNG
   :width: 54%

1. Wire the R-LINK module as shown in the figure below. A USB to micro
   USB cable connects a pc with the R-LINK module and the 5pin cable
   goes between the R-LINK module and the Motor.

2. Connect the AK80-6 motor to a power supply (24V, 12A) and do not cut
   off the power before the setting is completed.

3. Start the R-Link Config Tool application (only runs on Windows).

4. Select serial port: USB-Serial\_CH340,wch,cp along with an
   appropriate baud rate (both 921600 and 115200 Bd should work). If
   the serial port option USB-Serial\_CH340,wch,cp does not show up,
   your pc can’t establish a connection to the R-LINK module due to
   remaining driver issues.

5. Choose the desired motor settings on the left side of the config
   tool GUI. Enter the correct CAN ID of the motor under
   ``MotorSelectEnter``. A label on the motor shows the ID.

   -  Velocity: 5 rad/s is a relatively slow speed of revolution, hence
      it offers a good starting point.
   -  Torque: be careful setting a fixed torque, because the friction
      inside the motor decreases with the speed of revolution.
      Therefore a fixed torque commonly leads to either no movement at
      all or accelerates the motor continuously.

6. Start the plotting by ticking the boxes of position, velocity,
   torque and select ``Display``

7. Press ``Run`` to start recording the plots.

8. ``Enter M_Mode`` to control the motor. This is indicated by a color
   change of the plot line, from red to green.

9. In order to push changes in the settings to the motor, press ``Send Once``.

   .. warning:: This button does not work reliably. Usually it has
     to be activated several times before the setting changes
     actually apply on the motor.

10. Stop the motor inside the M-Mode by setting the velocity to 0 and
    pressing ``Send Once`` until the changes apply.

11. ``Exit M_Mode`` to exit the control mode of the motor.

    .. warning:: The next time you start the motor control with
      ``Enter M_Mode`` the motor will restart with the exact same
      settings as you left the control mode with ``Exit M_Mode``. This
      is especially dangerous if a weight is attached to the pendulum
      and the motor control was left with high velocity or torque
      settings.

12. Use ``Stop`` to deactivate the plotting.


Debugging
<<<<<<<<<

Error messages that showed up during the configuration procedure, such
as ``UVLO`` (VM under voltage lockout) and ``OTW`` (Thermal warning and
shutdown), could be interpreted with the help of the data sheet for the
DRV8353M 100-V Three-Phase Smart Gate Driver from Texas Instruments:

| **Datasheet:**
  `DRV8353M <https://www.ti.com/lit/ds/symlink/drv8353m.pdf>`__ (on the
  first Page under: 1. Features)
| 


Communication
~~~~~~~~~~~~~

CAN Bus wiring
<<<<<<<<<<<<<<

Along the CAN bus proper grounding and ideally, isolated ground is
required for improvement of the signal quality. Therefore, the common
shared ground for PC and motors is of great importance in CAN connection
communication. When daisy-chaining multiple actuators, one can use the
Ground form the R-Link connector of the motor, which is connected to the
negative power pin. This can share the common ground from the PC side
and power supply. At the very beginning and end of the CAN chain, there
must be of the termination resistors of :math:`\small{120\,\Omega}`
between CAN-High and CAN-Low, which will be then connected to the
corresponding pins between drivers. These resistors aim to absorb the
signals and prevents the signals from being reflected at the wire ends.
The CAN protocol is differential, hence no additional ground reference
is needed. The diagram below displays the wiring of the CAN bus.

.. image:: ../../hardware/images/can_bus.png
   :width: 100%
   :align: center

**Fig. 2:** main pc = CPU, CAN transceiver = CAN XCVR, actuator = AC

Setting up the CAN interface
<<<<<<<<<<<<<<<<<<<<<<<<<<<<

During regular operation the motors are commanded via CAN interface.
To setup the CAN connection follow these steps:

-  Run this command in the terminal to make sure that ``can0`` 
   (or any other can interface depending on the system)
   shows up as an interface after connecting the USB cable to your PC:
   
.. code:: 

    ip link show

-  Configure the ``can0`` interface to have a 1 Mbaud communication
   frequency: 
   
.. code::

    sudo ip link set can0 type can bitrate 1000000

-  To bring up the ``can0`` interface, run: 
  
.. code:: 

    sudo ip link set up can0

.. note:: Alternatively, one could run the shell script
   ``setup_caninterface.sh`` which will do the job for you.

.. note:: To change motor parameters such as CAN ID or to calibrate the
   encoder, a serial connection is used. The serial terminal GUI used on Linux
   for this purpose is ``cutecom``

Testing Motor Connection
<<<<<<<<<<<<<<<<<<<<<<<<

To test the connection to the motors, you can use the performance profiling
script.  The script will print the communication frequencies to the terminal.

| **Performance Profiler:** Sends and received 1000 zero commands to
  measure the communication frequency with 1/2 motors.
| 

Python Interface
~~~~~~~~~~~~~~~~

The Python - Motor communication is done with the `python driver
<https://github.com/dfki-ric-underactuated-lab/mini-cheetah-tmotor-python-can>`__.
The basic python interface is the following:

Example Motor Initialization (for can interface ``can0`` and ``motor_id`` =1):

.. code:: 

    motor = CanMotorController(can_socket='can0', motor_id=1, socket_timeout=0.5)

Available functions:

.. code:: 

  pos, vel, tau = motor.enable_motor()
  pos, vel, tau = motor.disable_motor()
  pos, vel, tau = motor.set_zero_position()
  pos, vel, tau = motor.send_deg_command(position_in_degrees, velocity_in_degrees, Kp, Kd, tau_ff)
  pos, vel, tau = motor.send_rad_command(position_in_radians, velocity_in_radians, Kp, Kd, tau_ff)

All functions return current position, velocity, torque in SI units
except for ``send_deg_command``, which returns degrees instead of radians.


Internal PD-Controller
~~~~~~~~~~~~~~~~~~~~~~

A proportional-derivative controller, which is based on the MIT
Mini-Cheetah Motor, is implemented on the motor controller board. The
control block diagram of this closed loop controller is shown below. It
can bee seen that the control method is flexible, as pure position,
speed, feed forward torque control or any combination of those is
possible.

.. image:: ../../hardware/images/motor_ak80-6_pdcontroller.png
   :width: 80%
   :align: center

In the `python driver <https://github.com/dfki-ric-underactuated-lab/mini-cheetah-tmotor-python-can>`__ ,
the::

    send_rad_command(Pdes, Pvel, Kp, Kd, tff)

function lets you set desired position (Pdes), velocity (Pvel), Kp, Kd
and feed forward torque (tff) values at every time step.
