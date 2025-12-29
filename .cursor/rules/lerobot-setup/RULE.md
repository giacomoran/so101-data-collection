---
alwaysApply: true
---

# LeRobot Setup

This project uses the LeRobot library for robotics data collection with the SO101 robot arm.

## Installation

LeRobot is installed from a **pinned git commit** (not PyPI) because the latest PyPI version (0.4.2) is outdated. See `pyproject.toml` for the exact commit hash.

When you need to understand how LeRobot works internally browse directly `.venv/lib/python3.10/site-packages/lerobot/`.

## Environment Setup

This project uses **nix + uv** (not conda as LeRobot docs suggest).

To enter the dev environment:

```bash
nix develop
uv sync
```
