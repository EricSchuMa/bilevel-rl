from abc import ABC
from typing import Any, Dict, Tuple, Union

import mlflow
import numpy as np
import pandas as pd
from stable_baselines3.common.logger import KVWriter


class MLflowOutputFormat(KVWriter, ABC):
    """
    Dumps key/value pairs into MLflow's numeric format.
    """

    def write(self,
              key_values: Dict[str, Any],
              key_excluded: Dict[str, Union[str, Tuple[str, ...]]],
              step: int = 0
              ) -> None:
        for (key, value), (_, excluded) in zip(
            sorted(key_values.items()), sorted(key_excluded.items())
        ):
            if excluded is not None and "mlflow" in excluded:
                continue
            if isinstance(value, np.ScalarType):
                if not isinstance(value, str):
                    mlflow.log_metric(key, value, step)


def log_agent_params(config):
    mlflow.log_param('agent', config.get('agent'))
    mlflow.log_param('lr', config.getfloat('learning_rate'))
    mlflow.log_param('max_grad_norm', config.getfloat('max_grad_norm'))
    mlflow.log_param('gamma', config.getfloat('gamma'))

    mlflow.log_param('batch_size', config.getint('batch_size'))
    mlflow.log_param('train_freq', config.getint('train_freq'))
    mlflow.log_param('learning_starts', config.getint('learning_starts'))
    mlflow.log_param('buffer_size', config.getint('buffer_size'))
    mlflow.log_param('exploration_fraction', config.getfloat('exploration_fraction'))

def log_env_params(config):
    mlflow.log_param('delta_time', config.getint('delta_time'))
    mlflow.log_param('yellow_time', config.getint('yellow_time'))
    mlflow.log_param('min_green', config.getint('min_green'))
    mlflow.log_param('observation_c', config.getint('observation_c'))
    mlflow.log_param('observation_fn', config.get('observation_fn'))
    mlflow.log_param('reward_fn', config.get('reward_fn'))
    mlflow.log_param('num_seconds', config.getint('num_seconds'))


def log_meta_params(config):
    mlflow.log_param('meta_lr', config.getfloat('lr'))
    mlflow.log_param('meta_n_steps', config.getfloat('n_steps'))
    mlflow.log_param('meta_batch_size', config.getfloat('batch_size'))


def log_rew_params(config):
    mlflow.log_param('diff_wait', config.getfloat('diff_waiting_time'))
    mlflow.log_param('queue', config.getfloat('queue'))
    mlflow.log_param('brake', config.getfloat('brake'))
    mlflow.log_param('speed', config.getfloat('average_speed'))
    mlflow.log_param('wait', config.getfloat('wait'))


def log_net_params(config):
    mlflow.log_param('h1', config.get('h1'))
    mlflow.log_param('h2', config.get('h2'))


def log_training_params(training_run):
    for k, v in training_run.data.params.items():
        mlflow.log_param("train/" + k, v)


def log_episode_metrics(env, ep_reward, episode):
    df = pd.DataFrame(env.last_trip_info)
    mlflow.log_metric('episode/reward', ep_reward, episode)
    mlflow.log_metric('episode/timeLoss', df['timeLoss_sec'].astype(float).mean(), episode)
