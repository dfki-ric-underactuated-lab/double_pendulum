import numpy as np
from pydrake.systems.trajectory_optimization import DirectCollocation
from pydrake.trajectories import PiecewisePolynomial
from pydrake.all import Solve  # pydrake.all should be replaced by real path

from double_pendulum.trajectory_optimization.direct_collocation import dircol_utils


class dircol_calculator():
    def __init__(self, urdf_path, system_name):
        self.urdf_path = urdf_path
        self.system_name = system_name

        self.plant, self.context, self.scene_graph = dircol_utils.create_plant_from_urdf(self.urdf_path)

    def compute_trajectory(self,
                           n,
                           tau_limit,
                           initial_state,
                           final_state,
                           theta_limit,
                           speed_limit,
                           R,
                           time_panalization,
                           init_traj_time_interval,
                           minimum_timestep,
                           maximum_timestep):
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
            # self.dircol.AddConstraintToAllKnotPoints(-torque_limit <= u[1])
            # self.dircol.AddConstraintToAllKnotPoints(u[1] <= torque_limit)
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
        self.dircol.AddFinalCost(self.dircol.time() * time_panalization)
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
        dircol_utils.animation(self.plant, self.scene_graph, self.x_traj)

