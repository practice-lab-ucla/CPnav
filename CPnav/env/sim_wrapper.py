import logging


class PolarAction:
    """
    Simple polar action representation:
      - r: forward distance (meters)
      - theta: yaw change (radians), positive is counterclockwise
      - type: optional string label (e.g., 'default', 'stop', 'null')
    """

    # Default action if the VLM response is not parsed
    default = None
    # Special sentinel actions
    stop = None
    null = None

    def __init__(self, r: float, theta: float, type: Optional[str] = None):
        self.r = r
        self.theta = theta
        self.type = type

# Global sentinel instances
STOP_ACTION = PolarAction(0.0, 0.0, "stop")
NULL_ACTION = PolarAction(0.0, 0.0, "null")


class TeleopAction:
    """
    Simple polar action representation:
      - r: forward distance (meters)
      - theta: yaw change (radians), positive is counterclockwise
      - type: optional string label (e.g., 'default', 'stop', 'null')
    """

    def __init__(self, x: float, z: float, yaw: float):
        self.x = x
        self.z = z
        self.yaw = yaw


import habitat_sim
import magnum as mn
import numpy as np
from typing import Optional
from ..utils import local_to_global

from habitat_sim.utils.common import quat_from_angle_axis, quat_to_angle_axis


class SimWrapper:
    """
    A thin wrapper around Habitat-Sim that:
      - Initializes the simulator, agent, and sensors.
      - Applies polar actions to move and rotate the agent.
      - Returns observations with the current agent state.
    """

    def __init__(self, cfg: dict):
        """
        Initialize the simulator with the specified configuration.

        Expected keys in cfg:
          - 'scene_id': short scene identifier (string)
          - 'scene_path': path to the .glb file
          - 'scene_config': path to the scene dataset config
          - 'use_goal_image_agent': bool
          - 'allow_slide': bool
          - 'agent_radius': float
          - 'agent_height': float
          - 'sensor_cfg': dict with:
                'pitch' (float, radians),
                'fov' (float, degrees),
                'height' (float, meters),
                'res_factor' (int)
          - 'goal_image_agent_fov': float (if use_goal_image_agent is True)
        """
        self.scene_id = cfg["scene_id"]
        self.scene_path = cfg["scene_path"]

        self.use_goal_image_agent = cfg["use_goal_image_agent"]
        self.allow_slide = cfg["allow_slide"]

        sensor_cfg = cfg["sensor_cfg"]
        self.sensor_pitch = sensor_cfg["pitch"]
        self.fov = sensor_cfg["fov"]
        self.sensor_height = sensor_cfg["height"]
        self.resolution = (
            sensor_cfg["resolution"][0] // sensor_cfg["res_factor"],
            sensor_cfg["resolution"][1] // sensor_cfg["res_factor"],
        )

        # ------------------------------------------------------------------
        # Simulator configuration
        # ------------------------------------------------------------------
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = cfg["scene_path"]
        backend_cfg.scene_dataset_config_file = cfg["scene_config"]
        backend_cfg.enable_physics = True

        # Agent configuration
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.radius = cfg["agent_radius"]
        agent_cfg.height = cfg["agent_height"]
        agent_cfg.sensor_specifications = self._create_sensor_specs()

        agents = [agent_cfg]

        # Optional goal-image agent
        if self.use_goal_image_agent:
            goal_cfg = self._create_goal_agent_config(
                fov=cfg["goal_image_agent_fov"]
            )
            agents.append(goal_cfg)

        self.sim_cfg = habitat_sim.Configuration(backend_cfg, agents)

        try:
            self.sim = habitat_sim.Simulator(self.sim_cfg)
            logging.info("Simulator initialized successfully.")
        except Exception as e:
            logging.error(f"Error initializing simulator: {e}")
            raise RuntimeError(f"Could not initialize simulator: {e}") from e

    # ----------------------------------------------------------------------
    # Sensor and agent configuration helpers
    # ----------------------------------------------------------------------

    def _create_sensor_specs(self):
        """
        Create sensor specifications for the main agent.

        Returns:
            list[habitat_sim.SensorSpec]: List of sensor specifications.
        """
        sensor_specs = [
            self._create_rgb_sensor_spec(),
            self._create_depth_sensor_spec(),
        ]
        return sensor_specs

    def _create_rgb_sensor_spec(self):
        """
        Create an RGB camera sensor specification.

        Returns:
            habitat_sim.CameraSensorSpec: RGB sensor specification.
        """
        rgb_sensor_spec = habitat_sim.CameraSensorSpec()
        rgb_sensor_spec.uuid = "color_sensor"
        rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor_spec.resolution = self.resolution
        rgb_sensor_spec.hfov = self.fov
        rgb_sensor_spec.position = mn.Vector3([0.0, self.sensor_height, 0.0])
        rgb_sensor_spec.orientation = mn.Vector3([self.sensor_pitch, 0.0, 0.0])
        return rgb_sensor_spec

    def _create_depth_sensor_spec(self):
        """
        Create a depth camera sensor specification.

        Returns:
            habitat_sim.CameraSensorSpec: Depth sensor specification.
        """
        depth_sensor_spec = habitat_sim.CameraSensorSpec()
        depth_sensor_spec.uuid = "depth_sensor"
        depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor_spec.resolution = self.resolution
        depth_sensor_spec.hfov = self.fov
        depth_sensor_spec.position = mn.Vector3([0.0, self.sensor_height, 0.0])
        depth_sensor_spec.orientation = mn.Vector3([self.sensor_pitch, 0.0, 0.0])
        return depth_sensor_spec

    def _create_goal_agent_config(self, fov: float):
        """
        Create a configuration for the goal image capture agent.

        Args:
            fov: Field of view for the goal agent (degrees).

        Returns:
            habitat_sim.agent.AgentConfiguration: Goal agent configuration.
        """
        goal_cfg = habitat_sim.agent.AgentConfiguration()
        goal_sensor_spec = habitat_sim.CameraSensorSpec()
        goal_sensor_spec.uuid = "goal_sensor"
        goal_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        goal_sensor_spec.resolution = self.resolution
        goal_sensor_spec.hfov = fov
        goal_sensor_spec.orientation = mn.Vector3([0.0, 0.0, 0.0])
        goal_sensor_spec.position = mn.Vector3([0.0, 0.0, 0.0])
        goal_cfg.sensor_specifications = [goal_sensor_spec]
        return goal_cfg

    # ----------------------------------------------------------------------
    # Core simulation API
    # ----------------------------------------------------------------------

    def step(self, action: PolarAction) -> dict:
        """
        Apply a polar action to the agent and return new observations.

        Args:
            action: PolarAction describing yaw change and forward motion.

        Returns:
            dict: Sensor observations for agent 0, plus 'agent_state'.
        """
        agent = self.sim.get_agent(0)
        curr_state = agent.get_state()

        if action is PolarAction.null:
            # Do not change the state; just re-sense the environment.
            new_agent_state = curr_state
        else:
            new_agent_state = habitat_sim.AgentState()
            new_agent_state.position = np.copy(curr_state.position)
            new_agent_state.rotation = curr_state.rotation

            # Apply yaw rotation and forward motion
            self._rotate_yaw(new_agent_state, new_agent_state.rotation, action.theta)
            self._move_forward(
                new_agent_state,
                new_agent_state.position,
                new_agent_state.rotation,
                action.r,
            )

            agent.set_state(new_agent_state)

        observation = self.sim.get_sensor_observations(0)
        observation["agent_state"] = agent.get_state()
        return observation

    def _move_forward(
        self,
        agent_state: habitat_sim.AgentState,
        curr_position: np.ndarray,
        curr_rotation,
        magnitude: float,
    ) -> None:
        """
        Move the agent_state forward by a specified magnitude, using the
        pathfinder and optional sliding.

        Args:
            agent_state: The agent state to update.
            curr_position: Current agent position.
            curr_rotation: Current agent rotation (quaternion).
            magnitude: Distance to move forward (meters).
        """
        # Forward in local frame is negative Z
        local_point = np.array([0.0, 0.0, -magnitude], dtype=np.float32)
        global_point = local_to_global(curr_position, curr_rotation, local_point)

        # Move in 10 substeps to respect collisions / navigation mesh
        delta = (global_point - curr_position) / 10.0
        new_position = np.copy(curr_position)

        for _ in range(10):
            if self.allow_slide:
                new_position = self.sim.pathfinder.try_step(
                    new_position, new_position + delta
                )
            else:
                new_position = self.sim.pathfinder.try_step_no_sliding(
                    new_position, new_position + delta
                )

        agent_state.position = new_position

    def _rotate_yaw(
        self,
        agent_state: habitat_sim.AgentState,
        curr_rotation,
        magnitude: float,
    ) -> None:
        """
        Rotate the agent_state around the global +Y axis by the given angle.

        Args:
            agent_state: The agent state to update.
            curr_rotation: Current agent rotation (quaternion).
            magnitude: Angle in radians to rotate counterclockwise.
        """
        theta, axis = quat_to_angle_axis(curr_rotation)

        # Ensure consistent rotation direction (axis.y non-negative)
        if axis[1] < 0:
            theta = 2.0 * np.pi - theta

        new_theta = theta - magnitude
        agent_state.rotation = quat_from_angle_axis(
            new_theta, np.array([0.0, 1.0, 0.0], dtype=np.float32)
        )

    # ----------------------------------------------------------------------
    # Extra utilities
    # ----------------------------------------------------------------------

    def get_goal_image(self, goal_position: np.ndarray, goal_rotation) -> np.ndarray:
        """
        Capture an RGB image from the goal agent's perspective.

        Args:
            goal_position: 3D position for the goal agent.
            goal_rotation: Quaternion rotation for the goal agent.

        Returns:
            np.ndarray: The captured image from the goal sensor.
        """
        assert self.use_goal_image_agent, "Goal image agent is not enabled."

        goal_agent = self.sim.get_agent(1)
        new_agent_state = habitat_sim.AgentState()
        new_agent_state.position = goal_position
        new_agent_state.rotation = goal_rotation
        goal_agent.set_state(new_agent_state)

        observations = self.sim.get_sensor_observations(1)
        return observations["goal_sensor"]

    def set_state(self, pos: np.ndarray, quat) -> None:
        """
        Set the main agent's state.

        Args:
            pos: Position (3D vector).
            quat: Quaternion orientation.
        """
        agent = self.sim.get_agent(0)
        agent_state = habitat_sim.AgentState()
        agent_state.position = pos
        agent_state.rotation = quat
        agent.set_state(agent_state)

    def get_path(self, path: habitat_sim.MultiGoalShortestPath) -> float:
        """
        Compute the geodesic distance for the given path request.

        Args:
            path: A habitat_sim.MultiGoalShortestPath with start and ends set.

        Returns:
            float: Geodesic distance if a path is found; large value otherwise.
        """
        if self.sim.pathfinder.find_path(path):
            return path.geodesic_distance
        else:
            logging.info("NO PATH FOUND")
            return 1000.0

    def close(self) -> None:
        """
        Close the simulator to clean up memory / GPU resources.
        """
        try:
            self.sim.close()
        except Exception:
            # Closing twice or after partial init should not crash the program.
            pass
    
    def teleport_to_xz(self, action: TeleopAction) -> dict:
        """
        Directly teleport the main agent (agent 0) to world-frame (x, z),
        keeping current y. Optionally set yaw (rotation about +Y).

        Args:
            action: TeleopAction(x, z, yaw). If yaw is None, keep current rotation.

        Returns:
            dict: Sensor observations for agent 0, plus 'agent_state'.
        """
        agent = self.sim.get_agent(0)
        curr_state = agent.get_state()

        # Keep current y (Habitat uses y-up)
        y = float(curr_state.position[1])

        new_state = habitat_sim.AgentState()
        new_state.position = np.array([float(action.x), y, float(action.z)], dtype=np.float32)

        # Preserve rotation unless yaw is provided
        if action.yaw is None:
            new_state.rotation = curr_state.rotation
        else:
            new_state.rotation = quat_from_angle_axis(
                float(action.yaw),
                np.array([0.0, 1.0, 0.0], dtype=np.float32),
            )

        agent.set_state(new_state)

        observation = self.sim.get_sensor_observations(0)
        observation["agent_state"] = agent.get_state()
        return observation

