"""Runtime checks against a real environment.

These need CUDA, a compiled MuJoCo scene and a populated motion library, so they are
skipped in a bare checkout. They cover the invariants that a pure-Python unit test
cannot reach -- the ones that depend on what mjlab actually does at runtime rather
than on what its documentation says.

Run with::

    MUJOCO_GL=egl uv run pytest tests/test_integration_env.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytest.importorskip("mjlab")

_MANIFEST = Path(
  "data/current/"
  "stage1_paper_mix_final_backflip_cartwheel_balanced_grounded_no_fall_getup.json"
)

pytestmark = [
  pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
  pytest.mark.skipif(
    not _MANIFEST.exists(), reason="needs a motion library (run prepare_motions)"
  ),
]


@pytest.fixture(scope="module")
def env():
  import mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  import ex_grmt  # noqa: F401

  cfg = load_env_cfg("ExGRMT-Stage1-Flat-Unitree-G1")
  cfg.scene.num_envs = 4
  # Noise off so history frames can be compared exactly.
  for term in cfg.observations["proprio_hist"].terms.values():
    if "enabled" in term.params:
      term.params["enabled"] = False
  e = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  yield e
  e.close()


def _zero_step(env):
  action = torch.zeros(
    env.num_envs, env.action_manager.total_action_dim, device=env.device
  )
  return env.step(action)


def test_table_ii_ground_friction_range_is_physically_effective(env):
  """The terrain floor must not clamp low robot-friction samples via MuJoCo max."""
  import re

  terrain = env.scene["terrain"]
  terrain_ids = terrain.indexing.geom_ids.long()
  terrain_mu = env.sim.model.geom_friction[:, terrain_ids, 0]
  assert torch.allclose(terrain_mu, torch.full_like(terrain_mu, 0.10))

  collision_ids = torch.tensor(
    [
      geom_id
      for geom_id in range(env.sim.mj_model.ngeom)
      if re.match(r".*_collision\d*$", env.sim.mj_model.geom(geom_id).name)
    ],
    device=env.device,
  )
  robot_mu = env.sim.model.geom_friction[:, collision_ids, 0]
  assert bool((robot_mu >= 0.10).all() and (robot_mu <= 1.75).all())
  # shared_random=True gives one coefficient to every robot collision geom in an env.
  torch.testing.assert_close(robot_mu, robot_mu[:, :1].expand_as(robot_mu))
  # MuJoCo's equal-priority contact rule is max(mu_plane, mu_robot), hence the
  # effective coefficient is exactly the sampled robot value over the full range.
  torch.testing.assert_close(torch.maximum(robot_mu, terrain_mu), robot_mu)


def test_proprio_history_is_time_ascending_with_newest_last(env):
  """The single most load-bearing layout assumption in the architecture.

  ``ExGRMTActor`` slices ``o_seq[:, -1]`` as ``o_t`` (Eq. 11) and builds the
  interleaved history so the final token is ``z_t^o`` (Eq. 6). If mjlab buffered
  history newest-first, both would silently use a 10-step-stale observation.
  """
  from mjlab.envs.mdp.observations import joint_pos_rel

  from ex_grmt.envs.env_cfg import HISTORY_LENGTH
  from ex_grmt.rsl_rl.config import ExGRMTActorCfg

  h = HISTORY_LENGTH
  dims = ExGRMTActorCfg().proprio_term_dims

  env.reset()
  for _ in range(h + 3):
    obs, *_ = _zero_step(env)

  flat = obs["proprio_hist"]
  assert flat.shape[-1] == h * sum(dims)

  # Per-term blocks, each viewed as (H, d) -- exactly _proprio_sequence's split.
  parts, off = [], 0
  for d in dims:
    parts.append(flat[:, off : off + h * d].view(-1, h, d))
    off += h * d
  seq = torch.cat(parts, dim=-1)

  joint_pos_block = seq[..., dims[0] + dims[1] : dims[0] + dims[1] + dims[2]]
  current = joint_pos_rel(env, biased=True)

  newest_err = (joint_pos_block[:, -1] - current).abs().max()
  oldest_err = (joint_pos_block[:, 0] - current).abs().max()
  assert newest_err < 1e-6, "history[-1] is not the current frame"
  assert oldest_err > newest_err, "history appears to be newest-first"


def test_command_window_centre_is_the_current_reference(env):
  """Token L (offset 0) must be the reference frame the policy is acting on.

  mjlab's step order is: process_action -> physics -> reward/termination ->
  command_manager.compute() (which advances time_steps) -> observation_manager.
  So the observation returned by ``step()`` already reflects the *next* reference
  index, and the following ``process_action`` reads that same index. Observation and
  action are therefore aligned -- this test pins that alignment.
  """
  from ex_grmt.envs.env_cfg import COMMAND_WINDOW_RADIUS

  cmd = env.command_manager.get_term("motion")
  env.reset()
  obs, *_ = _zero_step(env)

  window = obs["command_window"].view(
    env.num_envs, cmd.num_window_tokens, cmd.command_token_dim
  )
  centre_joint_ref = window[:, COMMAND_WINDOW_RADIUS, 9:]
  # The window group is noisy by config (Table II command perturbation); compare
  # against the clean reference with a tolerance covering the +-0.1 rad joint noise.
  assert torch.allclose(centre_joint_ref, cmd.joint_pos, atol=0.1 + 1e-4), (
    "command-window centre is not the current reference frame"
  )


def test_zero_action_targets_the_reference_pose(env):
  """Paper Eq. (3): ``q_tar = q_ref + a``, so ``a = 0`` must command ``q_ref``.

  This is what distinguishes the implementation from mjlab's default-pose offset,
  and it is invisible to any shape or dtype check.

  The reference must be sampled *before* stepping: ``command_manager.compute()`` runs
  near the end of ``step()``, so by the time ``step()`` returns, ``cmd.joint_pos`` has
  already advanced past the frame the action was built against.
  """
  cmd = env.command_manager.get_term("motion")
  term = env.action_manager.get_term("joint_pos")

  env.reset()
  _zero_step(env)  # settle; reset() itself does not run process_action

  ref_before = cmd.joint_pos[:, term._target_ids].clone()
  _zero_step(env)

  assert torch.allclose(term._processed_actions, ref_before, atol=1e-5), (
    "zero action did not command the reference pose; the action term is offsetting "
    "from somewhere else (mjlab's default standing pose?)"
  )


def test_action_and_observation_share_a_reference_index(env):
  """The window the policy sees must be the window its residual is applied to.

  A one-frame slip here is invisible at 50 Hz -- errors stay small and training still
  converges -- but the policy would be correcting toward the wrong target.
  """
  from ex_grmt.envs.env_cfg import COMMAND_WINDOW_RADIUS

  cmd = env.command_manager.get_term("motion")
  term = env.action_manager.get_term("joint_pos")

  env.reset()
  obs, *_ = _zero_step(env)

  for _ in range(5):
    window = obs["command_window"].view(
      env.num_envs, cmd.num_window_tokens, cmd.command_token_dim
    )
    seen_centre = window[:, COMMAND_WINDOW_RADIUS, 9:]
    obs, *_ = _zero_step(env)
    acted_on = term._processed_actions  # == q_ref used by this step's action
    # Same frame, up to the command-window noise the policy is trained to tolerate.
    assert (seen_centre - acted_on).abs().max() < 0.1 + 1e-3
