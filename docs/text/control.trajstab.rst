Trajectory Stabilization
========================

When transfering a precomputed trajectory from simulation to reality (or even
another simulator), the trajectory needs to be stabilized, as the mathematical
system dynamics are always only an approximation of the real dynamics.  Because
of this, during the execution small errors will sum up and the system will
eventually deviate from the nominal trajectory. 

.. toctree::
   :maxdepth: 1

   control.trajstab.tvlqr.rst
   control.trajstab.riccati.rst
