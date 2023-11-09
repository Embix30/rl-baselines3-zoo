import numpy as np
import sys
import time
from copy import deepcopy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_linear_fn
from typing import Optional, Union, List

from rl_zoo3.custom_algos import QLearning, QTableWrapper

import torch as th
import torch.nn as nn


class DoubleQLearning(QLearning):

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        policy_tensorA = th.zeros((self.observation_space.n, self.action_space.n), dtype=th.float32)
        policy_tensorB = th.zeros((self.observation_space.n, self.action_space.n), dtype=th.float32)

        policyA = QTableWrapper(policy_tensorA)
        policyB = QTableWrapper(policy_tensorB)

        self.policy = nn.ModuleDict({
            'A': policyA,
            'B': policyB
        })

        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.exploration_fraction,
        )

    def predict(self, observation: np.ndarray, deterministic: bool = False, **kwargs):
        """
        Get the model's action from an observation.

        :param observation: the input observation
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action
        """
        #if self.to_mask and (self.action_mask is None or episode_start):
        #    self.action_mask=self.env.reset_infos[0]['action_mask']

        if not deterministic and np.random.rand() < self.exploration_rate:
            n_batch = observation.shape[0]
            action_mask = None if not self.to_mask else self.action_mask[observation[0]]
            action = np.array([self.action_space.sample(mask = action_mask) for _ in range(n_batch)])
        else:
            q_values = th.mean(th.concat([self.policy['A'][observation],self.policy['B'][observation]]),dim=0)

            action = self.select_max_action(q_values,observation)
                        
        return action, None

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "Double-Q-Learning",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
    
    def train(self, actions, new_obs, rewards, dones, learning_rate):
            # Update Q-tables using the Double Q-learning update rule
            # Randomly choose which Q-table to update
            if np.random.random() < 0.5:
                primary_policy, secondary_policy = self.policy['A'], self.policy['B']
            else:
                primary_policy, secondary_policy = self.policy['B'], self.policy['A']

            if not dones[0]:
                
                # Get the action that maximizes the Q-value in the second Q-table

                max_action_primary = self.select_max_action(primary_policy[new_obs],new_obs)

                target = th.tensor(rewards, dtype=th.float32) + self.gamma * secondary_policy[new_obs,max_action_primary]

                primary_policy[self._last_obs, actions] += learning_rate * (
                    target - primary_policy[self._last_obs, actions]
                )
            else:
                primary_policy[self._last_obs, actions] += learning_rate * (
                    th.tensor(rewards, dtype=th.float32) - primary_policy[self._last_obs, actions]
                )
