# SO101 Data Collection

Exploring data collection strategies for the SO101 robot arm.

## Latency Measurement Tools

The `src/latency/` directory contains tools for measuring end-to-end latency in the robot control pipeline, following the methodology from the [UMI paper](https://umi-gripper.github.io/).

- **`measure_camera.py`**: Measures camera capture latency using LeRobot's OpenCVCamera wrapper. Displays QR codes encoding timestamps on screen, captures them with the camera, and calculates latency by comparing the decoded timestamp to the receive time.
- **`measure_camera_raw.py`**: Compares multiple OpenCV capture strategies to find minimum achievable camera latency. Tests different approaches (default capture, buffer size=1, frame flushing, MJPG codec) to isolate buffering vs hardware latency.
- **`measure_proprioception.py`**: Measures servo position read round-trip time (RTT) for Feetech STS3215 servos. Timestamps sync_read commands and approximates proprioception latency as RTT/2, assuming symmetric serial delays.
- **`measure_execution.py`**: Measures servo execution latency (command-to-motion delay). Sends sinusoidal position commands while reading actual positions, then uses cross-correlation to find the phase lag between commanded and actual signals.

### Results (Our Setup)

**Hardware:** innomaker U20CAM 1080p camera, Feetech STS3215 servos, macOS with AVFoundation backend

- **Camera (end-to-end)**: ~100ms
- **Proprioception**: ~0.45ms (RTT ~0.9ms)
- **Execution (e2e)**: ~100ms (most joints; shoulder_lift slightly higher at 124ms possibly due to gravity load)

**Note:** Latency compensation is typically needed when data collection embodiment ≠ deployment embodiment (e.g., UMI's case using GoPro for collection but different cameras on the robots). Our single-camera setup maintains consistent latency across training and deployment, so the policy implicitly learns to compensate for consistent latency. Same reasoning applies to proprioception and execution latency.
