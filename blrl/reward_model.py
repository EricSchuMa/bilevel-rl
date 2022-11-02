import numpy as np
import torch
import torch.nn as nn


class RewardModel(nn.Module):
    def __init__(self, metrics, initial_weights=None, normalize=None, norm_config=None, learnable=None, device='cpu'):
        super().__init__()
        self.metrics = metrics

        # initialize weights with initial weighting parameters or equal weighting
        if learnable:
            if initial_weights is not None:
                self.weights = torch.nn.Parameter(torch.Tensor(initial_weights, device=device).unsqueeze(dim=-1))
            else:
                self.weights = torch.nn.Parameter(
                    1 / len(metrics) * torch.ones(len(metrics), device=device).unsqueeze(dim=-1))
        else:
            self.weights = initial_weights if initial_weights is not None else np.full(len(metrics), 1 / len(metrics))
        self.normalize = normalize
        self.normalization_ranges = get_normalization_ranges(norm_config)

    def compute_sub_rewards(self, traffic_light):
        rewards = []
        for metric, weight in zip(self.metrics, self.weights):
            reward = get_reward_method_by_metric(traffic_light, metric)()
            if self.normalize:
                reward = reward / self.normalization_ranges[metric] * 10
            rewards.append(reward)
        return rewards

    def compute_reward(self, traffic_light):
        return sum([reward * weight for reward, weight in zip(self.compute_sub_rewards(traffic_light), self.weights)])


def generate_reward_model_from_config(norm_config, reward_config, normalize=True, learnable=False):
    return RewardModel(metrics=list(reward_config.keys()),
                       initial_weights=np.array(list(reward_config.values()), dtype=np.float32),
                       normalize=normalize,
                       norm_config=norm_config,
                       learnable=learnable)


def get_reward_fn(config, reward_config):
    if config.get('reward_fn') == 'intrinsic':
        reward_model = generate_reward_model_from_config(config, reward_config)
        return lambda traffic_light: reward_model.compute_reward(traffic_light)
    else:
        return config.get('reward_fn')


def get_reward_method_by_metric(traffic_light, metric):
    return getattr(traffic_light, f"_{metric}_reward")


def get_normalization_ranges(config):
    return {'brake': config.getfloat('brake_range'),
            'emission': config.getfloat('emission_range'),
            'pressure': config.getfloat('pressure_range'),
            'queue': config.getfloat('queue_range'),
            'average_speed': config.getfloat('speed_range'),
            'diff_waiting_time': config.getfloat('diff_wait_range'),
            'wait': config.getfloat('wait_range'),
            }
