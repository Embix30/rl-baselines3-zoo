import numpy as np
import sys
import time
from copy import deepcopy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_linear_fn
from typing import Optional, Union

from rl_zoo3.custom_algos import DoubleQLearning

import torch as th
import torch.nn as nn


class TruncatedDoubleQ(DoubleQLearning):

    # Linear schedule will be defined in `_setup_model()`
    exploration_schedule: Schedule

    def __init__(
        self,
        C_lower: float = 10,
        C_upper: float = 10,
        *args, **kwargs
    ):
        super().__init__(
            *args, **kwargs
        )
        self.C_lower = C_lower
        self.C_upper = C_upper


    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "Truncated-Double-Q-Learning",
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

                eps = self.gamma* max(-self.C_lower*learning_rate,min(secondary_policy[new_obs,max_action_primary]-primary_policy[new_obs,max_action_primary],self.C_upper*learning_rate))

                target = th.tensor(rewards, dtype=th.float32) + self.gamma * primary_policy[new_obs,max_action_primary] + eps

                primary_policy[self._last_obs, actions] += learning_rate * (
                    target - primary_policy[self._last_obs, actions]
                )
            else:
                primary_policy[self._last_obs, actions] += learning_rate * (
                    th.tensor(rewards, dtype=th.float32) - primary_policy[self._last_obs, actions]
                )
