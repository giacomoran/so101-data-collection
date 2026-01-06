"""Evaluation scripts for ACT policies on SO101 robot.

Available scripts:
- eval_sync: Synchronous inference (matches lerobot-record behavior)
- eval_sync_discard: Synchronous inference with action discarding (UMI-style)
- eval_async_discard: Asynchronous inference with action discarding

Usage:
    python -m so101_data_collection.eval.eval_sync --robot.type=so100_follower ...
    python -m so101_data_collection.eval.eval_sync_discard --robot.type=so100_follower ...
    python -m so101_data_collection.eval.eval_async_discard --robot.type=so100_follower ...
"""
