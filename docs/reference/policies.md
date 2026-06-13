# Policies

::: adaptivepy.policies.base
    options:
      members:
        - Policy
        - register_policy
        - get_policy
        - list_policies
        - POLICY_REGISTRY

::: adaptivepy.policies.least_counts
    options:
      members:
        - LeastCountsPolicy

::: adaptivepy.policies.random
    options:
      members:
        - RandomPolicy

::: adaptivepy.policies.fast
    options:
      members:
        - FastPolicy
        - compute_fast_rewards
        - feature_scale
