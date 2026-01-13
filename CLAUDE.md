# CLAUDE.md

This file provides guidance for Claude Code working in this repository.

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

### Code Naming

Follow a noun-first approach for variables and constants, for example `idChat` instead of `chatId`. Follow a verb-first approach for functions, for example `computeContentFromTree` or `fetchDataUser`. Notice that we still follow the noun-first approach, it's `fetchDataUser` instead of `fetchUserData`. Plural applies to the noun, for example for multiple user ids use `idsUser` instead of `idsUsers` or `idUsers`.

We use the `bak` prefix/suffix for backup files and folders, ignore those unless explicitly asked. We use the `tmp` prefix/suffix for temporary files and folders, ignore those unless explicitly asked. We use the `zxtra` prefix/suffix for extra files and folders (work-in-progress or uncertain), generally ignore those as well.

### Nominal Consistency

Align the caller's variable names with the callee's parameter names to reduce cognitive load and make data flow explicit. The variable names outside a function must match the internal parameters exactly. The exception is when handling generic sequences or iterative data, where indexed suffixes (like `_1, _2, _3`) are used to represent distinct instances of the same conceptual type.

## File Organization

- **Plans**: All project plans should be stored in the `plans/` folder.

## Consistency

Code within a project should be consistent with the rest of the project, even if this means breaking the rules above. Adhere to shared coding standards, style guides, and best practices across the project. Write code that is easy for any team member to read, understand, and modify, ensuring consistent naming conventions, formatting, and patterns. Prioritize clarity over cleverness, and aim for simplicity and predictability.
