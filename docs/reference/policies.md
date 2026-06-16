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

::: adaptivepy.policies.knn_as
    options:
      members:
        - KnnAsPolicy
        - compute_knn_as_scores

::: adaptivepy.policies.ma_reap
    options:
      members:
        - MaReapPolicy
        - aggregate_agent_scores
        - apply_stakes_method
        - compute_agent_scores
