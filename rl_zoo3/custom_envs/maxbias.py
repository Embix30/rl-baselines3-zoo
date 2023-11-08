""" implementation of grid search environment from lecture notes"""
import random
from typing import Optional, Tuple

import numpy as np
from gymnasium import Env, spaces

START_STATE = 0


class ConstructedMaxBias(Env):
    """grid world environment"""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    def __init__(self, number_arms: int = 20, distribution: str = 'normal', mean: float = -0.1, variance: float = 1, render_mode=None, strict_mask=True) -> None:
        """example from sutton 6.7

        Args:
            number_arms (int): number of possible actions in state `B` aka `1`
        """
        assert isinstance(number_arms, int), f"number_arms has to be an int but is {type(number_arms)}"
        assert number_arms > 2, f"number_arms has to be greater than 2 but is {number_arms}"
        if distribution not in ('normal', 'uniform'):
            raise ValueError("distribution must be 'normal' or 'uniform'")
        
        self.number_arms = number_arms
        self.observation_space = spaces.Discrete(3)
        # the state `0` is equal to `A` and the state `1` is equal to `B` and
        # `2` is equal to terminal state
        self.action_space = spaces.Discrete(self.number_arms)

        self.distribution = distribution
        self.mean = mean
        self.variance = variance

        std_dev = np.sqrt(self.variance)
        # probability distribution according to params
        if self.distribution == 'normal':
            self.reward_func = lambda: np.random.normal(self.mean, std_dev)
        elif self.distribution == 'uniform':
            # Calculate the bounds using the standard deviation for a uniform distribution
            lower_bound = self.mean - np.sqrt(3) * std_dev
            upper_bound = self.mean + np.sqrt(3) * std_dev
            self.reward_func = lambda: np.random.uniform(lower_bound, upper_bound)

        self.render_mode=render_mode

        self.strict_mask= strict_mask

        # initialize information of game
        self.info = {}

        # reset environment
        self.state = None
        self.reset()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """step function of the environment

        Args:
            action (int): action to take

        Raises:
            ValueError: game is in an invalid state

        Returns:
            Tuple[np.ndarray, float, bool, dict]: next state, reward, done, info
        """
        valid_action_space = self.get_valid_actions(self.state)
        done = False
        truncated = False
        # check if action is valid
        if not valid_action_space[action]:
            if self.strict_mask:
                raise ValueError(f'action {action} not allowed in state {self.state} and strict mask is True')
            else:
                return self.state, 0, done, truncated, {}

        # if state is `A`
        if self.state == 0:
            reward = 0.0
            if action == 0:
                next_state = 1
                info = {
                    "next_state": next_state,
                    "reward": reward,
                    "done": done,
                    "action_mask":self.get_valid_actions(next_state),                
                }
            else:
                next_state = 2
                done = True
                info = {
                    "next_state": next_state,
                    "reward": reward,
                    "done": done,
                    "terminal_observation":next_state,                
                }
        # if state is `B`
        elif self.state == 1:
            reward = self.reward_func()
            done = True
            next_state = 2
            info = {
                "next_state": next_state,
                "reward": reward,
                "done": done,
                "terminal_observation":next_state
            }            
        elif self.state == 2:
            info = {"message": "We are already in the terminal state"}
            return 2, 0.0, True, truncated, info
        else:
            raise ValueError("state is not valid")
        self.state = next_state
        self.info = info
        return next_state, reward, done, truncated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Reset the environment and return the initial state and an empty info dict.

        Args:
            seed (Optional[int], optional): Random seed. Defaults to None.
            options (Optional[dict], optional): Options for reset. Defaults to None.

        Returns:
            Tuple[np.ndarray, dict]: Tuple of initial state observation and info dict.
        """
        # Reset the state of the environment to an initial state
        # The 'super().reset()' call is necessary if the superclass's reset method
        # initializes important attributes or has other important side effects.
        # However, if it returns a value that is not used or expected, you can omit
        # assigning its return value to anything.
        super().reset(seed=seed)

        self.state = START_STATE

        # You may want to return additional data in the info dictionary
        info = {'action_mask':self.get_valid_actions(self.state)}

        return self.state, info  # Return the observation and an info dict.

    def get_valid_actions(self, agent_position: int) -> list[int]:
        """get valid actions for a given agent position

        Args:
            agent_position (int): agent position to get the valid actions from

        Raises:
            ValueError: Error is raised if the given position is not valid

        Returns:
            list[int]: list of all possible actions
        """
        if agent_position == 0:
            valid = [0,1]
        elif agent_position in [1, 2]:
            valid = list(range(self.action_space.n))
        else:
            raise ValueError("agent_position is not defined")
        
        mask = np.zeros(self.action_space.n, dtype=np.int8)
        mask[valid] = np.int8(1)
        return mask

    def render(self, mode = None) -> None:
        """render the environment"""

        print(self.info)

    def costum_sample(self) -> np.ndarray:
        """sample a random state from the environment"""
        sample = random.choice([0, 1])
        return sample
    