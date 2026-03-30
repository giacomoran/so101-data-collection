# Direct Manipulation Setups Benchmark

## Overview

A benchmark experiment comparing two robot data collection setups across three manipulation tasks. The goal is to verify intuitions about the trade-offs between leader-follower teleoperation and direct manipulation. This is for a blog post, not a formal paper.

## Data Collection Setups

### 1. Teleoperation

- Standard approach in LeRobot
- Human operates a "leader" arm that the "follower" mirrors
- **Pros**: Works directly in joint space (no IK)
- **Cons**: No force feedback; introduces lag; makes precise/reactive tasks harder

### 2. Direct Manipulation

- Custom handle + trigger setup to control follower directly
- **Pros**: Works in joint space; provides force feedback; zero lag
- **Cons**:
- Human hand occludes top/front cameras (may be constrained to wrist cam only)
- Follower motors too stiff for direct guidance, so still need leader arm hardware
- Annoying setup: must swap "leader" with "follower" when switching between data collection and inference
- Does not work with rigid grippers, only with soft grippers

---

## Tasks

| Task                                 | Description                     | Success Criteria       |
| ------------------------------------ | ------------------------------- | ---------------------- |
| **Pick and Place Cube**              | Standard manipulation benchmark | TBD                    |
| **Press Up Arrow on GBA**            | Precision task                  | Character moves upward |
| **Throw Ping-Pong Ball into Basket** | Dynamic task requiring timing   | TBD                    |

---

## Camera Setup

- **Teleoperation**: Top camera + Wrist camera
- **Direct-manipulation**: Wrist camera (+ ablation with added top camera, which likely degrades performance given train-inference mismatch)

---

## Experimental Design

### Data Collection

**Time per condition**: 15 minutes (following UMI paper approach)
**Total conditions**: 2 setups × 3 tasks = 6
**Total collection time**: 6 × 15 min = 90 min (1h 30min)

This time-based approach (vs. fixed episode count) naturally captures the efficiency differences between setups. Some setups may yield more episodes in 15 minutes than others.

### Learning Effects Mitigation

**Why randomization isn't feasible**: Switching to/from direct-manipulation setup requires physically swapping the leader and follower arms. Randomizing task/setup order would mean constant hardware reconfiguration.

**Mitigations**:

1. **Prior experience**: Operator already has experience with both setups, reducing "first-time" learning effects

**Acknowledged limitation**: Results may still reflect some ordering effects depending on collection sequence.

### Model Training

**Architecture**: ACTSmooth

| Setup                | Camera Config | Models per Task |
| -------------------- | ------------- | --------------- |
| Teleoperation        | Top + Wrist   | 1               |
| Direct-Manipulation  | Wrist only    | 1               |
| Direct-Manipulation  | Top + Wrist   | 1               |

**Total models**: 3 tasks × (2 + 1 ablation) = **9 models**

### Evaluation

**Episodes per model**: 20
**Total evaluation episodes**: 9 models × 20 episodes = **180 episodes**

---

## Metrics

### Data Collection Metrics

| Metric                      | Description                                     |
| --------------------------- | ----------------------------------------------- |
| **Episodes collected**      | Number of episodes recorded in 15 minutes       |
| **Total dataset length**    | Actual usable data (excludes mistakes/failures) |
| **Total mistakes**          | Count of failed attempts, restarts, errors      |
| **Qualitative impressions** | Subjective notes from the collector             |

### Model Evaluation Metrics

| Metric           | Description                                      |
| ---------------- | ------------------------------------------------ |
| **Success rate** | % of 20 eval episodes that meet success criteria |

---

## Implementation Decisions

**Direct-Manipulation**:

- Uses a single arm (physically the leader arm with trigger handle)
- Arm connected as `SO101Leader` (torque disabled for free movement)
- Same joint positions used for BOTH `observation.state` AND `action` (identical values)
- This matches standard teleop where leader→action, follower→observation, except both come from one arm

**Data Collection**:

- Datasets recorded at 30FPS
- Capture resolution at 640×480 (not native 1080p)
