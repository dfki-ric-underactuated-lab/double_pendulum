Trajectory data
~~~~~~~~~~~~~~~

Trajectories can be computed with trajectory optimization methods such as
direct collocation and iterative LQR. For the execution zhe trajectories can
later be loaded by a stabilizing controller such as time varying LQR of a model
predictive controller.
The computed trajectories are stored in a csv file. The first line of the csv
file is reserved for the header defining the data type in the corresponding
column. The headers used are:

- time
- pos1,pos2,vel1,vel2,
- acc1,acc2
- pos_meas1,pos_meas2,vel_meas1,vel_meas2
- pos_filt1,pos_filt2,vel_filt1,vel_filt2
- pos_des1,pos_des2,vel_des1,vel_des2
- tau_con1,tau_con2
- tau_fric1,tau_fric2
- tau_meas1,tau_meas2
- tau_des1,tau_des2
- K11,K12,K13,K14,K21,K22,K23,K24
- k1,k2

.. note:: 

    Not all headers/columns have to used.

.. note::

   There should be no space after the separating comma.

The easiest way to stay consistent with this format is to use the functions

.. code::

    save_trajectory(...)
    load_trajectory(...)
    load_trajectory_full(...)

in double_pendulum.utils.csv_trajectory. These functions use the panda library
to save/load the data. Missing header labels are skipped.

