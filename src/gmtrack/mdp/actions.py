"""Residual joint-position action on top of the reference pose (paper Eq. 3).

The paper is explicit::

    The action space is a 29-dimensional residual joint-position command a_t.
    The residual is added to the *reference joint pose* to obtain the target
    joint position,        q_t^tar = q_t^ref + a_t.                    (3)

mjlab's ``JointPositionAction(use_default_offset=True)`` instead offsets from the
robot's **default standing pose** ``q_0``, a constant. That is what mjlab's
BeyondMimic port uses, and it defines a materially different learning problem: with a
constant offset the policy has to synthesise the entire reference trajectory in its
output, whereas with the reference offset ``a_t = 0`` already tracks the reference and
the policy only has to supply the correction that physics demands. The FSQ bottleneck
in particular is sized to carry a *correction* signal, not a full joint trajectory.

Timing: mjlab's step order is ``process_action`` -> decimated physics ->
observations/rewards -> ``command_manager.compute()``. The command's ``time_steps``
therefore still points at the current control step ``t`` when the offset is read, so
this really is ``q_t^ref``, not ``q_{t+1}^ref``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

  from gmtrack.mdp.commands import MultiMotionCommand


class ReferenceResidualJointPositionAction(JointPositionAction):
  """``q_tar = q_ref(t) + scale * a_t`` (paper Eq. 3)."""

  cfg: ReferenceResidualJointPositionActionCfg

  def __init__(
    self, cfg: ReferenceResidualJointPositionActionCfg, env: ManagerBasedRlEnv
  ) -> None:
    if cfg.use_default_offset:
      raise ValueError(
        "use_default_offset must be False: this term offsets from the reference "
        "pose, and the constant default-pose offset would be added on top."
      )
    super().__init__(cfg, env)
    self._command: MultiMotionCommand | None = None

  def _reference_joint_pos(self) -> torch.Tensor:
    """``q_t^ref`` for the controlled joints, shape ``(num_envs, action_dim)``."""
    if self._command is None:
      # Resolved lazily: the action manager is built before the command manager, so
      # the term does not exist yet at construction time.
      from gmtrack.mdp.commands import MultiMotionCommand

      self._command = cast(
        MultiMotionCommand,
        self._env.command_manager.get_term(self.cfg.command_name),
      )
      if not isinstance(self._command, MultiMotionCommand):
        raise TypeError(
          f"Command '{self.cfg.command_name}' is a {type(self._command).__name__}; "
          f"the reference-residual action needs a MultiMotionCommand."
        )
    # `command.joint_pos` is the reference for all robot joints in robot joint order;
    # `_target_ids` indexes that same order (the base class uses it on
    # `default_joint_pos`), so the two line up.
    return self._command.joint_pos[:, self._target_ids]

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_actions[:] = actions
    self._processed_actions = (
      self._raw_actions * self._scale + self._reference_joint_pos()
    )
    if self.cfg.clip is not None:
      self._processed_actions = torch.clamp(
        self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
      )


@dataclass(kw_only=True)
class ReferenceResidualJointPositionActionCfg(JointPositionActionCfg):
  """Configuration for :class:`ReferenceResidualJointPositionAction`."""

  command_name: str = "motion"
  """Name of the :class:`~gmtrack.mdp.commands.MultiMotionCommand` term."""
  use_default_offset: bool = False
  """Must stay False; the reference pose is the offset."""

  def build(self, env: ManagerBasedRlEnv) -> ReferenceResidualJointPositionAction:
    return ReferenceResidualJointPositionAction(self, env)
