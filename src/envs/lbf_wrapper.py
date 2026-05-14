from collections.abc import Iterable
import warnings

import numpy as np
from gymnasium.spaces import flatdim

from .multiagentenv import MultiAgentEnv
from .wrappers import FlattenObservation
from .lbf_envs.lbf_env2 import ForagingEnv  


class Lbf2Wrapper(MultiAgentEnv):
    def __init__(
        self,
        key, 
        players,
        max_player_level,
        field_size,
        max_food,
        sight,
        max_episode_steps,
        force_coop,
        normalize_reward,
        grid_observation,
        pretrained_wrapper=None,   
        seed=None,
        common_reward=False,
        reward_scalarisation="sum",
        observe_agent_levels=True,
        penalty=0.0,
        render_mode=None,
        **kwargs,
    ):
        
        if isinstance(field_size, int):
            field_size = (field_size, field_size)

        self._env = ForagingEnv(
            players=players,
            min_player_level=[1] * players,
            max_player_level=[max_player_level] * players,
            min_food_level=[1] * max_food,
            max_food_level=None,  
            field_size=field_size,
            max_num_food=max_food,
            sight=sight,
            max_episode_steps=max_episode_steps,
            force_coop=force_coop,
            normalize_reward=normalize_reward,
            grid_observation=grid_observation,
            observe_agent_levels=observe_agent_levels,
            penalty=penalty,
            render_mode=render_mode,
            **kwargs,
        )

        self._env = FlattenObservation(self._env)

        self.n_agents = self._env.unwrapped.n_agents
        self.episode_limit = max_episode_steps
        self.field_edge = field_size
        self.sight = sight

        self._obs = None
        self._info = {}

        self.longest_action_space = max(self._env.action_space, key=lambda x: x.n)
        self.longest_observation_space = max(
            self._env.observation_space, key=lambda x: x.shape
        )

        self._seed = seed
        try:
            self._env.unwrapped.seed(self._seed)
        except Exception:
            self._env.reset(seed=self._seed)

        self.common_reward = common_reward
        if self.common_reward:
            if reward_scalarisation == "sum":
                self.reward_agg_fn = lambda rewards: float(sum(rewards))
            elif reward_scalarisation == "mean":
                self.reward_agg_fn = lambda rewards: float(sum(rewards) / len(rewards))
            else:
                raise ValueError("reward_scalarisation must be 'sum' or 'mean'.")

    def _pad_observation(self, obs_list):
        target = self.longest_observation_space.shape[0]
        out = []
        for o in obs_list:
            pad = target - len(o)
            if pad > 0:
                o = np.pad(o, (0, pad), mode="constant", constant_values=0)
            out.append(o.astype(np.float32, copy=False))
        return out

    def reset(self, seed=None, options=None):
        obs, info = self._env.reset(seed=seed, options=options)
        self._obs = self._pad_observation(list(obs))
        self._info = info if isinstance(info, dict) else {}
        return self._obs, self._info

    def step(self, actions):
        actions = [int(a) for a in actions]
        obs, reward, done, truncated, info = self._env.step(actions)
        self._obs = self._pad_observation(list(obs))
        self._info = info if isinstance(info, dict) else {}

        if self.common_reward and isinstance(reward, Iterable):
            reward = self.reward_agg_fn(reward)
        elif not self.common_reward and not isinstance(reward, Iterable):
            warnings.warn(
                "common_reward is False but received scalar reward; returning as-is."
            )

        return self._obs, reward, bool(done), bool(truncated), self._info

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    def seed(self, seed=None):
        try:
            return self._env.unwrapped.seed(seed)
        except Exception:
            self._env.reset(seed=seed)
            return seed

    def save_replay(self):
        pass

    def get_stats(self):
        return {}

    def get_obs(self):
        return self._obs

    def get_obs_agent(self, agent_id):
        return self._obs[agent_id]

    def get_obs_size(self):
        return flatdim(self.longest_observation_space)

    def get_state(self):

        if not (isinstance(self._info, dict) and "state" in self._info):
            raise RuntimeError("info['state'] is missing. Check the env implementation.")
        return np.asarray(self._info["state"], dtype=np.float32)

    def get_state_size(self):
        """3 * (max_food + n_agents)"""
        envu = self._env.unwrapped
        return int(3 * (envu.max_num_food + envu.n_agents))

    def get_avail_actions(self):
        total = self.get_total_actions()
        return [[1] * total for _ in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        n = flatdim(self._env.action_space[agent_id])
        return [1] * n + [0] * (self.longest_action_space.n - n)

    def get_total_actions(self):
        return flatdim(self.longest_action_space)
