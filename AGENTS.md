# AGENTS.md

This file provides guidance for agentic coding agents working in this repository.

## Project Overview

Python 3.10+ monorepo for SO-101 robot arm data collection, training, and evaluation. Uses uv workspace with packages/ structure.
The extended main plan for this project is in @plans/main.md

## Environment Setup

This project uses **nix + uv** (not conda as LeRobot docs suggest).

To enter the dev environment:

```bash
nix develop
uv sync
```

Remember to source the virtual environment at `.venv` when running Python scripts.

## LeRobot Setup

This project uses the LeRobot library for robotics data collection with the SO101 robot arm.

LeRobot is installed from a **pinned git commit** (not PyPI) because the latest PyPI version (0.4.2) is outdated. See `pyproject.toml` for the exact commit hash.

IMPORTANT: When you need to understand how LeRobot works internally, explore it's source files in

```
.venv/lib/python3.10/site-packages/lerobot/
```

## Documentation References

### LeRobot Dataset v3 Format

When working with LeRobot datasets, refer to the official documentation:

- **Dataset Format**: https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3
- **Blog Post**: https://huggingface.co/blog/lerobot-datasets-v3

## Naming Conventions

- **HuggingFace repos/datasets**: Use underscores `_` instead of hyphens `-` in names (e.g., `cube_hand_guided` not `cube-hand-guided`)
- **Task names**: Use short names: `cube`, `gba`, `ball` (not `pick_place_cube`, `press_gba`, `throw_ball`)

## File Organization

- **Plans**: All project plans should be stored in the `plans/` folder.
