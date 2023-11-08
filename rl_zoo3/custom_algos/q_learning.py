import numpy as np
import sys
import time
import warnings
from copy import deepcopy
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import safe_mean
from typing import Optional, Union

import torch as th
import torch.nn as nn

class QTableWrapper(nn.Module):
    def __init__(self, policy):
        super(QTableWrapper, self).__init__()
        self.policy = nn.Parameter(policy, requires_grad=False)

    def forward(self):
        return self.policy

    def __getitem__(self, idx):
        return self.policy[idx]
    
    def __setitem__(self, idx, value):
        self.policy[idx] = value


class QLearning(BaseAlgorithm):

    # Linear schedule will be defined in `_setup_model()`
    exploration_schedule: Schedule

    def __init__(
        self,
        env: GymEnv,
        learning_rate: Union[float, Schedule] = 1e-4,
        learning_starts: int = 100,
        gamma: float = 0.99,
        policy: BasePolicy = None,  # Set a default value of None
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        device: Union[th.device, str] = "auto",
        support_multi_env: bool = False,
        monitor_wrapper: bool = True,
        seed: Optional[int] = None,
        # is to_mask is True then multi_env is not supported yet
        to_mask= False,
        action_mask = None,
        _init_setup_model = True,
        *args, **kwargs
    ):
        super(QLearning, self).__init__(policy=policy, env=env, learning_rate=learning_rate, stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            support_multi_env=support_multi_env,
            monitor_wrapper=monitor_wrapper,
            seed = seed,
            *args, **kwargs)
        
        self.gamma = gamma
        self.learning_starts = learning_starts

        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        # "epsilon" for the epsilon-greedy exploration
        self.exploration_rate = 0.0

        self.to_mask = to_mask
        self.action_mask = action_mask

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        policy_tensor = th.zeros((self.observation_space.n, self.action_space.n), dtype=th.float32)
        self.policy = QTableWrapper(policy_tensor)

        self.policy = self.policy.to(self.device)

        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.exploration_fraction,
        )

    def predict(self, observation: np.ndarray, deterministic: bool = False, episode_start = False, **kwargs):
        """
        Get the model's action from an observation.

        :param observation: the input observation
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action
        """
        # if self.to_mask and (self.action_mask is None or episode_start):
        #    self.action_mask=self.env.reset_infos[0]['action_mask']

        if not deterministic and np.random.rand() < self.exploration_rate:
            n_batch = observation.shape[0]
            action_mask = None if not self.to_mask else self.action_mask[observation[0]]
            action = np.array([self.action_space.sample(mask = action_mask) for _ in range(n_batch)])
        else:
            action = self.select_max_action(self.policy[observation],observation)
        return action, None

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "Q-Learning",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        if self.to_mask and self.action_mask is None:
            self.action_mask={}

        callback.on_training_start(locals(), globals())

        callback.on_rollout_start()

        # self._last_obs = self.env.reset() this is done in setup learn
        
        self._on_step()

        if self.to_mask and self._last_obs[0] not in self.action_mask.keys():
            # self.action_mask = self.env.reset_infos[0]['action_mask']
            self.action_mask[self._last_obs.item()]= self.env.reset_infos[0]['action_mask']

        while self.num_timesteps < total_timesteps:
               
            # Select action randomly or according to policy
            actions = self._sample_action(self.learning_starts, self.env.num_envs)

            new_obs, rewards, dones, infos = self.env.step(actions)

            if self.to_mask:
                if new_obs[0] not in self.action_mask.keys():
                    if dones[0]:
                        self.action_mask[new_obs.item()] = self.env.reset_infos[0]['action_mask']
                    else:
                        self.action_mask[new_obs.item()] = infos[0]['action_mask']

            self.num_timesteps += self.env.num_envs

            # Give access to local variables
            callback.update_locals(locals())

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)
            self._update_learning_rate()
            
            # Update Q-table
            self.train(actions, new_obs, rewards, dones)

            self._last_obs = new_obs

            callback.on_step()
            self._on_step()


            for idx, done in enumerate(dones):
                if done:
                    self._episode_num += 1

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()
                
        callback.on_rollout_end()

        callback.on_training_end()            
        
        return self
    
    def train(self, actions, new_obs, rewards, dones):
            if dones[0]:
                self.policy[self._last_obs, actions] += self.lr_schedule(self._current_progress_remaining) * (
                    th.tensor(rewards, dtype=th.float32) - self.policy[self._last_obs, actions]
                )

            else: 
                max_action = self.select_max_action(self.policy[new_obs],new_obs)

                self.policy[self._last_obs, actions] += self.lr_schedule(self._current_progress_remaining) * (
                    th.tensor(rewards, dtype=th.float32) + self.gamma * self.policy[new_obs,max_action] - self.policy[self._last_obs, actions]
                )
    
    def select_max_action(self, tensor, obs):
        if self.to_mask:
            # Convert the numpy mask to a torch tensor and make it boolean
            mask_tensor = th.from_numpy(self.action_mask[obs[0]]).bool()
            tensor = th.where(mask_tensor, tensor, th.tensor(-float('inf')))
        max_value = th.max(tensor)
        max_actions = th.nonzero(tensor.flatten() == max_value).flatten()

        # select random argmax
        return np.array(max_actions[th.randint(0, len(max_actions), (1,))])
    
    def _sample_action(
        self,
        learning_starts: int,
        n_envs: int = 1,
    ) -> np.ndarray:
        """
        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)

        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param n_envs:
        :return: action to take in the environment

        """
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts:
            # Warmup phase
            action_mask = None if not self.to_mask else self.action_mask[self._last_obs[0]]
            action = np.array([self.action_space.sample(mask=action_mask) for _ in range(n_envs)])
        else:
            action, _ = self.predict(self._last_obs, deterministic=False)

        return action
    
    def _dump_logs(self) -> None:
        """
        Write log.
        """
        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        self.logger.record("time/episodes", self._episode_num, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        if self.use_sde:
            self.logger.record("train/std", (self.actor.get_std()).mean().item())

        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
        # Pass the number of timesteps for tensorboard
        self.logger.dump(step=self.num_timesteps)



    def _on_step(self) -> None:
        """
        Update the exploration rate 
        This method is called after each step in the environment.
        """

        self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
        self.logger.record("rollout/exploration_rate", self.exploration_rate)

    def _update_learning_rate(self) -> None:
        """
        Update the optimizers learning rate using the current learning rate schedule
        and the current progress remaining (from 1 to 0).

        :param optimizers:
            An optimizer or a list of optimizers.
        """
        # Log the current learning rate
        self.logger.record("train/learning_rate", self.lr_schedule(self._current_progress_remaining))


    # def save(self, save_path):
    #     # Here, you can implement logic to save your model
    #     np.save(save_path, self.policy)

    # def load(self, load_path):
    #     # Here, you can implement logic to load your model
    #     self.policy = np.load(load_path)
