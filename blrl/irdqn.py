from configparser import SectionProxy
from typing import Union, Type, Optional, Tuple, Dict, Any, List, Iterable

import higher
import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import Schedule, GymEnv, MaybeCallback, TrainFreq, RolloutReturn
from stable_baselines3.common.utils import should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.dqn import DQN
from stable_baselines3.dqn.policies import DQNPolicy
from torch import Tensor
from torch.nn import functional as F

from blrl.buffers import SubRewardReplayBuffer
from blrl.reward_model import generate_reward_model_from_config


class IRQDN(DQN):
    """
    Intrinsic Reward Deep Q-Network (DQN)

    Paper: https://arxiv.org/abs/1312.5602, https://www.nature.com/articles/nature14236
    Default hyperparameters are taken from the nature paper,
    except for the optimizer and learning rate that were taken from Stable Baselines defaults.

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1) default 1 for hard update
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param target_update_interval: update the target network every ``target_update_interval``
        environment steps.
    :param exploration_fraction: fraction of entire training period over which the exploration rate is reduced
    :param exploration_initial_eps: initial value of random action probability
    :param exploration_final_eps: final value of random action probability
    :param max_grad_norm: The maximum value for the gradient clipping
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    :reward_model: A parameterized model of the intrinsic reward.

    # (copied from stable_baselines3 DQN class doc)
    """

    def __init__(
            self,
            policy: Union[str, Type[DQNPolicy]],
            env: Union[GymEnv, str],
            norm_config: SectionProxy,
            reward_config: SectionProxy,
            learning_rate: Union[float, Schedule] = 1e-4,
            buffer_size: int = 1000000,
            learning_starts: int = 50000,
            batch_size: Optional[int] = 32,
            tau: float = 1.0,
            gamma: float = 0.99,
            train_freq: Union[int, Tuple[int, str]] = 4,
            gradient_steps: int = 1,
            replay_buffer_class: Type[SubRewardReplayBuffer] = None,
            replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
            optimize_memory_usage: bool = False,
            target_update_interval: int = 10000,
            exploration_fraction: float = 0.1,
            exploration_initial_eps: float = 1.0,
            exploration_final_eps: float = 0.05,
            max_grad_norm: float = 10,
            tensorboard_log: Optional[str] = None,
            create_eval_env: bool = False,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
            meta_lr: float = 1e-4,
            meta_batch_size: int = 64,
            meta_n_steps: int = 20,
    ):
        super(IRQDN, self).__init__(policy,
                                    env,
                                    learning_rate,
                                    buffer_size,
                                    learning_starts,
                                    batch_size,
                                    tau,
                                    gamma,
                                    train_freq,
                                    gradient_steps,
                                    replay_buffer_class,
                                    replay_buffer_kwargs,
                                    optimize_memory_usage,
                                    target_update_interval,
                                    exploration_fraction,
                                    exploration_initial_eps,
                                    exploration_final_eps,
                                    max_grad_norm,
                                    tensorboard_log,
                                    create_eval_env,
                                    policy_kwargs,
                                    verbose,
                                    seed,
                                    device,
                                    _init_setup_model)
        self.reward_model = generate_reward_model_from_config(norm_config, reward_config, learnable=True)
        self.meta_n_steps = meta_n_steps
        self.meta_batch_size = meta_batch_size
        self.meta_opt = th.optim.Adam(self.reward_model.parameters(), lr=meta_lr)
        self.q_net_module = QNet(self.q_net.q_net, self.q_net_target.q_net)

    def train(self, gradient_steps: int, batch_size: int = 100, fnet=None, diffopt=None) -> None:
        # sync 1/2
        fnet.q_net_target.load_state_dict(self.q_net_target.q_net.state_dict())

        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = fnet.forward_target(replay_data.next_observations)
                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=1)
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = (1 - replay_data.dones) * self.gamma * next_q_values

            rewards = th.matmul(replay_data.sub_rewards, self.reward_model.weights)
            target_q_values += rewards

            # Get current Q-values estimates
            current_q_values = fnet(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Optimize the policy
            diffopt.step(loss)

        # sync 2/2
        self.q_net.q_net.load_state_dict(fnet.q_net.state_dict())
        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

    def online_cross_validate(self, fnet, batch_size: int = 100) -> Tensor:
        replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

        with th.no_grad():
            # Compute the next Q-values using the target q-network
            next_q_values = fnet.forward_target(replay_data.next_observations)
            # Follow greedy policy: use the one with the highest value
            next_q_values, _ = next_q_values.max(dim=1)
            # Avoid potential broadcast issue
            next_q_values = next_q_values.reshape(-1, 1)
            # 1-step TD target
            target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

        # Compute the current Q-values using the q-network
        current_q_values = fnet.forward(replay_data.observations)
        current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

        # Compute Huber loss (less sensitive to outliers)
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        return loss

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 4,
            eval_env: Optional[GymEnv] = None,
            eval_freq: int = -1,
            n_eval_episodes: int = 5,
            tb_log_name: str = "run",
            eval_log_path: Optional[str] = None,
            reset_num_timesteps: bool = True,
    ) -> OffPolicyAlgorithm:

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            eval_log_path,
            reset_num_timesteps,
            tb_log_name,
        )

        callback.on_training_start(locals(), globals())

        # outer-loop
        while self.num_timesteps < total_timesteps:
            with higher.innerloop_ctx(self.q_net_module, self.policy.optimizer, device=self.device) as (fnet, diffopt):
                # INNER-LOOP
                for _ in range(self.meta_n_steps):
                    rollout = self.collect_rollouts(
                        self.env,
                        train_freq=self.train_freq,
                        action_noise=self.action_noise,
                        callback=callback,
                        learning_starts=self.learning_starts,
                        replay_buffer=self.replay_buffer,
                        log_interval=log_interval,
                    )

                    if rollout.continue_training is False:
                        break

                    if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:

                        # If no `gradient_steps` is specified,
                        # do as many gradients steps as steps performed during the rollout
                        gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else rollout.episode_timesteps
                        # Special case when the user passes `gradient_steps=0`
                        if gradient_steps > 0:
                            self.train(batch_size=self.batch_size,
                                       gradient_steps=gradient_steps,
                                       fnet=fnet,
                                       diffopt=diffopt)

                # OUTER-LOOP
                ocv_loss = self.online_cross_validate(fnet, batch_size=self.meta_batch_size)
                self.meta_opt.zero_grad()
                ocv_loss.backward()
                self.meta_opt.step()
                self._log_meta_metrics(ocv_loss)

        callback.on_training_end()

        return self

    def collect_rollouts(
            self,
            env: VecEnv,
            callback: BaseCallback,
            train_freq: TrainFreq,
            replay_buffer: SubRewardReplayBuffer,
            action_noise: Optional[ActionNoise] = None,
            learning_starts: int = 0,
            log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        episode_rewards, total_timesteps = [], []
        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert env.num_envs == 1, "OffPolicyAlgorithm only support single environment"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if self.use_sde:
            self.actor.reset_noise()

        callback.on_rollout_start()
        continue_training = True

        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            done = False
            episode_reward, episode_timesteps = 0.0, 0

            while not done:

                if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                    # Sample a new noise matrix
                    self.actor.reset_noise()

                # Select action randomly or according to policy
                action, buffer_action = self._sample_action(learning_starts, action_noise)

                # Rescale and perform action
                new_obs, reward, done, infos = env.step(action)

                # override reward by the agent's intrinsic reward model
                tl = env.get_attr('traffic_signals')[0]['nt1']

                # have to take buffered values of sub_rewards, because VecEnv resets automatically
                sub_rewards = th.Tensor(env.buf_sub_rewards).to(self.device)

                self.num_timesteps += 1
                episode_timesteps += 1
                num_collected_steps += 1

                # Give access to local variables
                callback.update_locals(locals())
                # Only stop training if return value is False, not when it is None.
                if callback.on_step() is False:
                    return RolloutReturn(0.0, num_collected_steps, num_collected_episodes, continue_training=False)

                episode_reward += reward

                # Retrieve reward and episode length if using Monitor wrapper
                self._update_info_buffer(infos, done)

                # Store data in replay buffer (normalized action and unnormalized observation)
                self._store_transition(replay_buffer,
                                       buffer_action,
                                       new_obs,
                                       reward,
                                       done,
                                       infos,
                                       sub_rewards=sub_rewards)

                self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

                # For DQN, check if the target network should be updated
                # and update the exploration schedule
                # For SAC/TD3, the update is done as the same time as the gradient update
                # see https://github.com/hill-a/stable-baselines/issues/900
                self._on_step()

                if not should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
                    break

            if done:
                num_collected_episodes += 1
                self._episode_num += 1
                episode_rewards.append(episode_reward)
                total_timesteps.append(episode_timesteps)

                if action_noise is not None:
                    action_noise.reset()

                # Log training infos
                if log_interval is not None and self._episode_num % log_interval == 0:
                    self._dump_logs()

        mean_reward = np.mean(episode_rewards) if num_collected_episodes > 0 else 0.0

        callback.on_rollout_end()

        return RolloutReturn(mean_reward, num_collected_steps, num_collected_episodes, continue_training)

    def _store_transition(
            self,
            replay_buffer: SubRewardReplayBuffer,
            buffer_action: np.ndarray,
            new_obs: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            infos: List[Dict[str, Any]],
            sub_rewards: Optional[Tensor] = None,
    ) -> None:
        """
        Store transition in the replay buffer.
        We store the normalized action and the unnormalized observation.
        It also handles terminal observations (because VecEnv resets automatically).

        :param replay_buffer: Replay buffer object where to store the transition.
        :param buffer_action: normalized action
        :param new_obs: next observation in the current episode
            or first observation of the episode (when done is True)
        :param reward: reward for the current transition
        :param done: Termination signal
        :param infos: List of additional information about the transition.
            It may contain the terminal observations and information about timeout.
        """
        # Store only the unnormalized version
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        if done and infos[0].get("terminal_observation") is not None:
            next_obs = infos[0]["terminal_observation"]
            # VecNormalize normalizes the terminal observation
            if self._vec_normalize_env is not None:
                next_obs = self._vec_normalize_env.unnormalize_obs(next_obs)
        else:
            next_obs = new_obs_

        replay_buffer.add(
            self._last_original_obs,
            next_obs,
            buffer_action,
            reward_,
            done,
            infos,
            sub_rewards=sub_rewards,
        )

        self._last_obs = new_obs
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_

    def _log_meta_metrics(self, ocv_loss):
        self.logger.record("train/ocv_loss", float(ocv_loss))
        for metric, weight in zip(self.reward_model.metrics, self.reward_model.weights):
            self.logger.record(f"train/weight_{metric}", float(weight))

    def _clip_grad_norm(self, gradients: Iterable[th.Tensor]) -> Iterable[th.Tensor]:
        gradients = list(gradients)
        gs = [g for g in gradients if g is not None]
        max_norm = float(self.max_grad_norm)
        if len(gs) == 0:
            return gradients
        device = gs[0].device
        norm_type = 2.0
        total_norm = th.norm(th.stack([th.norm(g.detach(), norm_type).to(device) for g in gs]), norm_type)
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = th.clamp(clip_coef, max=1.0)
        for g in gradients:
            if g is not None:
                g.detach().mul_(clip_coef_clamped.to(g.device))
        return tuple(gradients)


class QNet(th.nn.Module):
    def __init__(self, q_net, q_net_target):
        super().__init__()
        self.add_module('q_net', q_net)
        self.add_module('q_net_target', q_net_target)

    def forward(self, obs):
        return self.q_net(obs)

    def forward_target(self, obs):
        return self.q_net_target(obs)
