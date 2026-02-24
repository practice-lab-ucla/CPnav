from typing import Optional

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
