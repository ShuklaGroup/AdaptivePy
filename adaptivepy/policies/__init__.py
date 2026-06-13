"""Adaptive sampling policies for AdaptivePy."""

from adaptivepy.policies.base import (
    POLICY_REGISTRY,
    Policy,
    get_policy,
    list_policies,
    register_policy,
)

# Import concrete policies so they self-register.
from adaptivepy.policies import fast  # noqa: F401
from adaptivepy.policies import least_counts  # noqa: F401
from adaptivepy.policies import ma_reap  # noqa: F401
from adaptivepy.policies import random  # noqa: F401

__all__ = [
    "POLICY_REGISTRY",
    "Policy",
    "get_policy",
    "list_policies",
    "register_policy",
]
