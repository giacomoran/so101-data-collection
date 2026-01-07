"""
Hardware setup constants for SO-101 data collection.

Edit these values to match your hardware configuration.
"""

# Robot ports (USB device paths)
ROBOT_PORT: str = "/dev/tty.usbmodem5A460829821"
LEADER_PORT: str = "/dev/tty.usbmodem5A460824651"

# Robot IDs (for calibration files)
ROBOT_ID: str = "arm_follower_0"
LEADER_ID: str = "arm_leader_0"

# Camera indices
WRIST_CAMERA_INDEX: int = 1
TOP_CAMERA_INDEX: int = 0

# Camera resolution
CAMERA_WIDTH: int = 640
CAMERA_HEIGHT: int = 480
