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

::: adaptivepy.policies.maxent_vampnet
    options:
      members:
        - MaxEntVampNetPolicy
        - compute_shannon_entropy
        - rank_frames_by_entropy
        - split_trajectories_from_dataset

::: adaptivepy.policies.ts_dar
    options:
      members:
        - TsDarPolicy
        - compute_ood_scores
        - compute_state_centers
        - rank_frames_by_ood
