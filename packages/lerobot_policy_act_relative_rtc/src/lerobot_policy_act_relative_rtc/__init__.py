"""ACT with Relative Joint Positions policy package for LeRobot.

This package provides a modified ACT (Action Chunking Transformer) policy that:
- Uses relative joint positions as action representation (action - obs.state[t])
- Uses observation deltas (obs.state[t] - obs.state[t-N]) as input

The policy is compatible with lerobot-train and can be used via:
    lerobot-train --policy.type act_relative_rtc ...
"""

try:
    import lerobot  # noqa: F401
except ImportError:
    raise ImportError(
        "lerobot is not installed. Please install lerobot to use this policy package."
    )

from .configuration_act_relative_rtc import ACTRelativeRTCConfig
from .modeling_act_relative_rtc import ACTRelativeRTCPolicy
from .processor_act_relative_rtc import make_act_relative_rtc_pre_post_processors
from .relative_stats import compute_relative_stats

__all__ = [
    "ACTRelativeRTCConfig",
    "ACTRelativeRTCPolicy",
    "make_act_relative_rtc_pre_post_processors",
    "compute_relative_stats",
]

