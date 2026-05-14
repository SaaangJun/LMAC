import os
import sys
from functools import partial

from .multiagentenv import MultiAgentEnv
from .smaclite_wrapper import SMACliteWrapper



if sys.platform == "linux":
    os.environ.setdefault("SC2PATH", "~/StarCraftII")

def __check_and_prepare_smac_kwargs(kwargs):
    assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
    assert kwargs[
        "common_reward"
    ], "SMAC only supports common reward. Please set `common_reward=True` or choose a different environment that supports general sum rewards."
    del kwargs["common_reward"]
    del kwargs["reward_scalarisation"]
    
    if "hallway" or "pursuit" or "disperse" in kwargs.get("key", ""):
        pass
    else:
        assert "map_name" in kwargs, "Please specify the map_name in the env_args"
    return kwargs


def smaclite_fn(**kwargs) -> MultiAgentEnv:
    kwargs = __check_and_prepare_smac_kwargs(kwargs)
    return SMACliteWrapper(**kwargs)

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["smaclite"] = smaclite_fn

def register_grf():
    from .grf_wrapper import GRFWrapper

    def grf_fn(**kwargs) -> MultiAgentEnv:
        kwargs = __check_and_prepare_smac_kwargs(kwargs)
        return GRFWrapper(**kwargs)

    REGISTRY["grf"] = grf_fn


# registering both smac and smacv2 causes a pysc2 error
# --> dynamically register the needed env
def register_smac():
    from .smac_wrapper import SMACWrapper

    def smac_fn(**kwargs) -> MultiAgentEnv:
        kwargs = __check_and_prepare_smac_kwargs(kwargs)
        return SMACWrapper(**kwargs)

    REGISTRY["sc2"] = smac_fn


def register_smacv2():
    from .smacv2_wrapper import SMACv2Wrapper

    def smacv2_fn(**kwargs) -> MultiAgentEnv:
        kwargs = __check_and_prepare_smac_kwargs(kwargs)
        return SMACv2Wrapper(**kwargs)

    REGISTRY["sc2v2"] = smacv2_fn


from envs.lbf_envs.lbf_env import ForagingEnv
import numpy as np
from gym import  spaces
from gym.spaces import flatdim

class LbfWrapper(MultiAgentEnv):
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
        pretrained_wrapper,
        **kwargs,
        ):

        self.episode_limit = max_episode_steps
        self.field_size = field_size
        self.sight = sight
        self._env = ForagingEnv(
            players = players,
            max_player_level = max_player_level,
            field_size = (field_size,field_size),
            max_food = max_food,
            sight = sight,
            max_episode_steps = max_episode_steps,
            force_coop = force_coop,
            normalize_reward = normalize_reward,
            grid_observation = grid_observation,
        )

        # if pretrained_wrapper:
        #     self._env = getattr(pretrained, pretrained_wrapper)(self._env)

        self.n_agents = self._env.n_agents
        self._obs = None

        self.longest_action_space = max(self._env.action_space, key=lambda x: x.n)
        self.longest_observation_space = max(self._env.observation_space, key=lambda x: x.shape)

        self._seed = kwargs["seed"]
        self._env.seed(self._seed)

    def observation(self, observation):
        return tuple(
            [
                spaces.flatten(obs_space, obs)
                for obs_space, obs in zip(self._env.observation_space, observation)
            ]
        )

    def step(self, actions):
        """ Returns reward, terminated, info """
        actions = [int(a) for a in actions]
        self._obs, reward, done, info, self._state= self._env.step(actions)
        self._obs = self.observation(self._obs)
        self._state = self.observation(self._state)
        self._obs = [
            np.pad(
                o,
                (0, self.longest_observation_space.shape[0] - len(o)),
                "constant",
                constant_values=0,
            )
            for o in self._obs
        ]

        return {}, float(sum(reward)), all(done), all(done), {}

    def get_obs(self):
        """ Returns all agent observations in a list """
        return self._obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        raise self._obs[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return flatdim(self.longest_observation_space)

    def get_state(self):
        return self._state

    def get_state_size(self):
        """ Returns the shape of the state"""
        grid_shape_x, grid_shape_y = self.field_size, self.field_size
        grid_shape_x += 2 * self.sight
        grid_shape_y += 2 * self.sight
        state_size = grid_shape_x * grid_shape_y * 3
        return state_size

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        valid = flatdim(self._env.action_space[agent_id]) * [1]
        invalid = [0] * (self.longest_action_space.n - len(valid))
        return valid + invalid

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return flatdim(self.longest_action_space)

    def reset(self):
        """ Returns initial observations and states"""
        self._obs, self._state = self._env.reset()
        self._obs = self.observation(self._obs)
        self._state= self.observation(self._state)

        self._obs = [
            np.pad(
                o,
                (0, self.longest_observation_space.shape[0] - len(o)),
                "constant",
                constant_values=0,
            )
            for o in self._obs
        ]

        return self.get_obs(), self.get_state()

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    def seed(self):
        return self._env.seed

    def save_replay(self):
        pass

    def get_stats(self):
        return {}
    

REGISTRY["lbf"] = partial(env_fn, env=LbfWrapper)    
    