import numpy as np
from meshcat.servers.zmqserver import start_zmq_server_as_subprocess

from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.rendering import MultibodyPositionToGeometryPose
from pydrake.systems.primitives import TrajectorySource
#from pydrake.systems.meshcat_visualizer import ConnectMeshcatVisualizer
from pydrake.geometry import StartMeshcat, MeshcatVisualizer
from pydrake.systems.analysis import Simulator
from pydrake.multibody.plant import MultibodyPlant
from pydrake.geometry import SceneGraph
from pydrake.multibody.parsing import Parser


def create_plant_from_urdf(urdf_path):
    """
    Create a pydrake plant froma  urdf file.

    Parameters
    ----------
    urdf_path : string or path object
        path to urdf file

    Returns
    -------
    pydrake.multibody.plant.MultibodyPlant
        pydrake plant
    pydrake.systems.Context
        pydrake context
    pydrake.geometry.SceneGraph
        pydrake scene graph
    """
    plant = MultibodyPlant(time_step=0.0)
    scene_graph = SceneGraph()
    plant.RegisterAsSourceForSceneGraph(scene_graph)
    parser = Parser(plant)
    parser.AddModels(urdf_path)
    plant.Finalize()
    context = plant.CreateDefaultContext()
    return plant, context, scene_graph


def construct_trajectories(dircol, result):
    """
    Construct trajectories from direct collocation result.

    Parameters
    ----------
    dircol : pydrake.systems.trajectory_optimization.DirectCollocation
        pydrake direct collocation object
    result :

    Returns
    -------
    pydrake.trajectories.PiecewisePolynomial_[float]
        state trajectory
    pydrake.trajectories.PiecewisePolynomial_[float]
        acceleration trajectory
    pydrake.trajectories.PiecewisePolynomial_[float]
        jerk trajectory
    pydrake.trajectories.PiecewisePolynomial_[float]
        torque trajectory
    """
    x_traj = dircol.ReconstructStateTrajectory(result)
    acc_traj = x_traj.derivative(derivative_order=1)
    jerk_traj = x_traj.derivative(derivative_order=2)
    u_traj = dircol.ReconstructInputTrajectory(result)
    return x_traj, acc_traj, jerk_traj, u_traj

def extract_data_from_polynomial(polynomial, frequency):
    """
    Extract data points from pydrake polnomial

    Parameters
    ----------
    polynomial : pydrake.trajectories.PiecewisePolynomial_[float]
        polynomial to be sampled for data points
    frequency : float
        Frequency of the extracted data trajectory

    Returns
    -------
    numpy_array
        shape=(N, 4)
        state trajectory
    numpy_array
        shape=(N,)
        time trajectory
    """
    n_points = int(polynomial.end_time() / (1 / frequency))
    time_traj = np.linspace(polynomial.start_time(),
                            polynomial.end_time(),
                            n_points)
    extracted_time = time_traj.reshape(n_points, 1).T
    extracted_data = np.hstack([polynomial.value(t) for t in
                                np.linspace(polynomial.start_time(),
                                            polynomial.end_time(),
                                            n_points)])
    return extracted_data, extracted_time

def animation(plant, scene_graph, x_trajectory):
    """
    Animate a trajectory in browser window.

    Parameters
    ----------
    plant : pydrake.multibody.plant.MultibodyPlant
        pydrake plant
    scene_graph : pydrake.geometry.SceneGraph
        pydrake scene graph
    x_trajectory : pydrake.trajectories.PiecewisePolynomial_[float]
        state trajectory to be animated
    """
    proc, zmq_url, web_url = start_zmq_server_as_subprocess(server_args=[])
    builder = DiagramBuilder()
    source = builder.AddSystem(TrajectorySource(x_trajectory))
    builder.AddSystem(scene_graph)
    pos_to_pose = builder.AddSystem(
        MultibodyPositionToGeometryPose(plant, input_multibody_state=True))
    builder.Connect(source.get_output_port(0),
                    pos_to_pose.get_input_port())
    builder.Connect(pos_to_pose.get_output_port(),
                    scene_graph.get_source_pose_port(plant.get_source_id()))
    meshcat = StartMeshcat()
    visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)
    simulator.Initialize()
    duration = x_trajectory.end_time()
    visualizer.StartRecording()
    simulator.AdvanceTo(duration)
    visualizer.PublishRecording()

    # meshcat = ConnectMeshcatVisualizer(builder,
    #                                    scene_graph,
    #                                    zmq_url=zmq_url,
    #                                    delete_prefix_on_load=True,
    #                                    open_browser=True)
    # meshcat.load()
    # diagram = builder.Build()
    # simulator = Simulator(diagram)
    # simulator.set_target_realtime_rate(1.0)
    # simulator.Initialize()
    # duration = x_trajectory.end_time()
    # meshcat.start_recording()
    # simulator.AdvanceTo(duration)
    # meshcat.stop_recording()
    # meshcat.publish_recording()
