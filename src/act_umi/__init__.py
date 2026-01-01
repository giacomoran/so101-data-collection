# ACT with Relative Joint Positions (UMI-style)
#
# This module provides an ACT policy that uses relative joint positions
# instead of absolute joint positions, following the UMI paper approach.
#
# Key differences from standard ACT:
# - Actions are represented relative to the current observation state
# - Input observations use deltas (velocity information) instead of absolute positions
# - Information about absolute robot position comes only from images

from .configuration_act_umi import ACTUMIConfig
from .modeling_act_umi import ACTUMIPolicy
from .processor_act_umi import make_act_umi_pre_post_processors

__all__ = [
    "ACTUMIConfig",
    "ACTUMIPolicy",
    "make_act_umi_pre_post_processors",
]

