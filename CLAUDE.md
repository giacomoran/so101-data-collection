# CLAUDE.md

This file provides guidance for Claude Code working in this repository.

## Project Overview

Python 3.12+ project for SO-101 robot arm direct manipulation data collection, training, and evaluation.

**Goal**: Benchmark comparing two data collection strategies (teleoperation, direct-manipulation) across three manipulation tasks (cube, gba, ball). See `plans/main.md` for full experimental design.

### Dependencies

- **LeRobot 0.5.0**: Robotics data collection and training framework
- **lerobot_policy_act_smooth**: ACT policy with prefix conditioning and async inference (editable dep from `../lerobot-policy-act-smooth`)

## Environment Setup

This project uses **nix + uv** (not conda).

To enter the dev environment:

```bash
nix develop
uv sync
```

Remember to source the virtual environment at `.venv` when running Python scripts.

## LeRobot Setup

This project uses the LeRobot library for robotics data collection with the SO101 robot arm.

LeRobot is installed from the **v0.5.0 git tag** (not yet on PyPI). See `pyproject.toml` for the source config.

IMPORTANT: When you need to understand how LeRobot works internally, explore its source files in

```
.venv/lib/python3.12/site-packages/lerobot/
```

## Documentation References

### LeRobot Dataset v3 Format

When working with LeRobot datasets, refer to the official documentation:

- **Dataset Format**: https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3
- **Blog Post**: https://huggingface.co/blog/lerobot-datasets-v3

## Naming Conventions

- **HuggingFace repos/datasets**: Use underscores `_` instead of hyphens `-` in names (e.g., `cube_direct_manipulation` not `cube-direct-manipulation`)
- **Task names**: Use short names: `cube`, `gba`, `ball` (not `pick_place_cube`, `press_gba`, `throw_ball`)

## Linting and formatting

- **Formatting**: Uses ruff for code formatting (auto-runs via Claude Code hook on `.py` file save)
- **Linting**: Uses ruff for linting (manual only)
- **Type Checking**: Disabled (no mypy/pyright/ty)

Run manually if needed:

```bash
uv run ruff format .
uv run ruff check .
```
