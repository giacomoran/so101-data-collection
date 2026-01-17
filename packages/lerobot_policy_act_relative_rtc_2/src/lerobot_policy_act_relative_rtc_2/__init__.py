"""ACT with Relative Joint Positions policy package for LeRobot (v2).

This package provides a modified ACT (Action Chunking Transformer) policy that:
- Uses relative joint positions as action representation (action - obs.state[t])
- Uses full action prefix conditioning instead of observation deltas (v2 improvement)
- Loads single observation for memory efficiency (v2 improvement)

The policy is compatible with lerobot-train and can be used via:
    lerobot-train --policy.type act_relative_rtc ...

Note: This is v2 which has breaking changes from lerobot_policy_act_relative_rtc (v1).
"""

try:
    import lerobot  # noqa: F401
except ImportError:
    raise ImportError("lerobot is not installed. Please install lerobot to use this policy package.")

from .configuration_act_relative_rtc import ACTRelativeRTCConfig
from .modeling_act_relative_rtc import ACTRelativeRTCPolicy
from .processor_act_relative_rtc import make_act_relative_rtc_2_pre_post_processors
from .relative_stats import compute_relative_stats

__all__ = [
    "ACTRelativeRTCConfig",
    "ACTRelativeRTCPolicy",
    "make_act_relative_rtc_2_pre_post_processors",
    "compute_relative_stats",
]
