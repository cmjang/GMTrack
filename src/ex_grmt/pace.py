"""PACE environment-role split (paper Sec. V-A, Algorithm 1 line 3).

Lives at package root because both sides of the codebase need it and neither should
depend on the other: :class:`~ex_grmt.mdp.commands.MultiMotionCommand` uses it to
decide which motion set an environment samples from, and
:class:`~ex_grmt.rsl_rl.storage.StarRolloutStorage` uses it to decide which rollout
rows feed the PPO surrogate versus the consolidation loss.

The storage infers an environment's role from its flat index alone
(``idx % num_envs < env_split``). If the two sides rounded differently, some
consolidation transitions would silently be optimized with the acquisition objective
-- which is precisely the "Mixed Training" ablation, not the method.
"""

from __future__ import annotations


def pace_env_split(acquisition_fraction: float, num_envs: int) -> int:
  """Number of acquisition environments ``|E_A|``; the remainder are ``E_C``.

  Args:
    acquisition_fraction: ``xi`` in Algorithm 1. Paper Table III uses 0.8.
    num_envs: Total parallel environments.

  Raises:
    ValueError: If the split cannot produce both roles.
  """
  if not 0.0 < acquisition_fraction < 1.0:
    raise ValueError(
      f"acquisition_fraction must be in (0, 1), got {acquisition_fraction}."
    )
  if num_envs < 2:
    raise ValueError(
      f"PACE needs at least 2 environments to form both an acquisition and a "
      f"consolidation group, got num_envs={num_envs}. Use a Stage-I task "
      f"(acquisition_fraction=None) for single-environment debugging."
    )
  return min(max(int(acquisition_fraction * num_envs), 1), num_envs - 1)
