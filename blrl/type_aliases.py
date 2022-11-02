from typing import NamedTuple
import torch as th


class SubRewardReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    sub_rewards: th.Tensor
