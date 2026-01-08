---
description: "General documentation and coding tips for this project"
alwaysApply: true
---

# General Documentation and Coding Tips

## Cursor Rules

- To add or modify cursor rules, refer to the official documentation: https://cursor.com/docs/context/rules

## LeRobot Dataset v3 Format

When working with LeRobot datasets, refer to the official documentation:

- **Dataset Format**: https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3
- **Blog Post**: https://huggingface.co/blog/lerobot-datasets-v3

## Naming Conventions

- **HuggingFace repos/datasets**: Use underscores `_` instead of hyphens `-` in names (e.g., `cube_hand_guided` not `cube-hand-guided`)
- **Task names**: Use short names: `cube`, `gba`, `ball` (not `pick_place_cube`, `press_gba`, `throw_ball`)

## Running Python scripts

Remember that we use Nix, so you need `nix develop`, see flake.nix; and we are using uv, so you need to source the virtual environment at .venv.
