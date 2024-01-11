import os
from pathlib import Path
import numpy as np

from pydrake.planning import DirectCollocation
from pydrake.trajectories import PiecewisePolynomial
from pydrake.all import Solve  # pydrake.all should be replaced by real path

from double_pendulum.trajectory_optimization.direct_collocation import dircol_utils
from double_pendulum.utils.urdfs import generate_urdf


class dircol_calculator():
    """dircol_calculator
    Class to calculate a trajectory for the double pendulum, acrobot or
    pendubot with the direct collocation method. Implementation uses drake.

    Parameters
    ----------
    urdf_path : string or path object
        path to urdf file
    robot : string
        robot which is used, Options:
            - "double_pendulum"
            - "acrobot"
            - "pendubot"
    model_pars : model_parameters object
        object of the model_parameters class
    save_dir : string
        path to directory where log data can be stored
        (necessary for temporary generated urdf)
        (Default value=".")
    """
    def __init__(self,
                 urdf_path,
                 robot,
                 model_pars,
                 save_dir="."):
        self.urdf_path = os.path.join(save_dir, robot + ".urdf")
        generate_urdf(urdf_path, self.urdf_path, model_pars=model_pars)
        self.system_name = robot

        meshes_path = os.path.join(Path(urdf_path).parent, "meshes")
        os.system(f"cp -r {meshes_path} {save_dir}")

        self.plant, self.context, self.scene_graph = dircol_utils.create_plant_from_urdf(self.urdf_path)

    def compute_trajectory(self,
                           n,
                           tau_limit,
                           initial_state,
                           final_state,
                           theta_limit,
                           speed_limit,
                           R,
                           time_penalization,
                           init_traj_time_interval,
                           minimum_timestep,
                           maximum_timestep):
        """compute_trajectory.

        Parameters
        ----------
        n : int
            number of knot points for the trajectory
        tau_limit : float
            torque limit, unit=[Nm]
        initial_state : array_like
            shape=(4,)
            initial_state for the trajectory
        final_state : array_like
            shape=(4,)
            final_state for the trajectory
        theta_limit : float
            position limit
        speed_limit : float
            velocity limit
        R : float
            control/motor torque cost
        time_penalization : float
            cost for trajectory length
        init_traj_time_interval : list
            shape=(2,)
            initial time interval for trajectory
        minimum_timestep : float
            minimum timestep size, unit=[s]
        maximum_timestep : float
            maximum timestep size, unit=[s]

        Raises
        ------
        AssertionError
            If the optmization is not successful
        """
        self.dircol = DirectCollocation(
                self.plant,
                self.context,
                num_time_samples=n,
                minimum_timestep=minimum_timestep,
                maximum_timestep=maximum_timestep,
                input_port_index=self.plant.get_actuation_input_port().get_index())

        # Add equal time interval constraint
        self.dircol.AddEqualTimeIntervalsConstraints()

        # Add initial torque condition
        torque_limit = tau_limit  # N*m
        u_init = self.dircol.input(0)
        self.dircol.AddConstraintToAllKnotPoints(u_init[0] == 0)

        # Add torque limit
        u = self.dircol.input()
        # Cost on input "effort"
        if self.plant.num_actuators() > 1:
            self.dircol.AddConstraintToAllKnotPoints(-torque_limit <= u[0])
            self.dircol.AddConstraintToAllKnotPoints(u[0] <= torque_limit)
            self.dircol.AddConstraintToAllKnotPoints(-torque_limit <= u[1])
            self.dircol.AddConstraintToAllKnotPoints(u[1] <= torque_limit)
            self.dircol.AddRunningCost(R * u[0] ** 2)
            self.dircol.AddRunningCost(R * u[1] ** 2)
        else:
            self.dircol.AddRunningCost(R * u[0] ** 2)
            self.dircol.AddConstraintToAllKnotPoints(-torque_limit <= u[0])
            self.dircol.AddConstraintToAllKnotPoints(u[0] <= torque_limit)

        # Initial state constraint
        self.dircol.prog().AddBoundingBoxConstraint(initial_state,
                                        initial_state,
                                        self.dircol.initial_state())

        # Angular velocity constraints
        state = self.dircol.state()
        self.dircol.AddConstraintToAllKnotPoints(state[2] <= speed_limit)
        self.dircol.AddConstraintToAllKnotPoints(-speed_limit <= state[2])
        self.dircol.AddConstraintToAllKnotPoints(state[3] <= speed_limit)
        self.dircol.AddConstraintToAllKnotPoints(-speed_limit <= state[3])

        # Add constraint on elbow position
        self.dircol.AddConstraintToAllKnotPoints(state[1] <= theta_limit)
        self.dircol.AddConstraintToAllKnotPoints(-theta_limit <= state[1])

        # Final state constraint
        self.dircol.prog().AddBoundingBoxConstraint(final_state, final_state, self.dircol.final_state())

        # Add a final cost equal to the total duration.
        self.dircol.AddFinalCost(self.dircol.time() * time_penalization)
        initial_x_trajectory = PiecewisePolynomial.FirstOrderHold(
            init_traj_time_interval,
            np.column_stack((initial_state, final_state)))
        self.dircol.SetInitialTrajectory(PiecewisePolynomial(), initial_x_trajectory)
        self.result = Solve(self.dircol.prog())
        assert self.result.is_success()
        #return result, self.dircol
        (self.x_traj,
         self.acc_traj,
         self.jerk_traj,
         self.u_traj) = dircol_utils.construct_trajectories(self.dircol, self.result)

    def get_trajectory(self, freq):
        """
        Get the trajectory found by the optimization.

        Parameters
        ----------
        freq : float
            frequency with which the trajectory is sampled

        Returns
        -------
        numpy_array
            time points, unit=[s]
            shape=(N,)
        numpy_array
            shape=(N, 4)
            states, units=[rad, rad, rad/s, rad/s]
            order=[angle1, angle2, velocity1, velocity2]
        numpy_array
            shape=(N, 2)
            actuations/motor torques
            order=[u1, u2],
            units=[Nm]
        """
        X, T = dircol_utils.extract_data_from_polynomial(self.x_traj, freq)

        if self.system_name == 'acrobot':
            u2_traj, _ = dircol_utils.extract_data_from_polynomial(self.u_traj, freq)
            u1_traj = np.zeros((u2_traj.size))
        elif self.system_name == 'pendubot':
            u1_traj, _ = dircol_utils.extract_data_from_polynomial(self.u_traj, freq)
            u2_traj = np.zeros((u1_traj.size))
        elif self.system_name == 'double_pendulum':
            torques, _ = dircol_utils.extract_data_from_polynomial(self.u_traj, freq)
            u1_traj = torques[0, :].reshape(T.size).T
            u2_traj = torques[1, :].reshape(T.size).T

        T = np.asarray(T).flatten()
        X = np.asarray(X).T
        U = np.asarray([u1_traj.flatten(), u2_traj.flatten()]).T
        return T, X, U

    def animate_trajectory(self):
        """
        Animate the trajectory, found by the optimization, with the drake
        meshcat viewer in a browser window.
        """
        dircol_utils.animation(self.plant, self.scene_graph, self.x_traj)
