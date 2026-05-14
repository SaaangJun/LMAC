from collections import namedtuple, defaultdict
from enum import Enum
from itertools import product
import logging
from typing import Iterable, Tuple

import gymnasium as gym
from gymnasium.utils import seeding
import numpy as np


class Action(Enum):
    NONE = 0
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4
    LOAD = 5


class CellEntity(Enum):
    OUT_OF_BOUNDS = 0
    EMPTY = 1
    FOOD = 2
    AGENT = 3


class Player:
    def __init__(self):
        self.controller = None
        self.position = None
        self.level = None
        self.field_size = None
        self.score = None
        self.reward = 0
        self.history = None
        self.current_step = None

    def setup(self, position, level, field_size):
        self.history = []
        self.position = position
        self.level = level
        self.field_size = field_size
        self.score = 0

    def set_controller(self, controller):
        self.controller = controller

    def step(self, obs):
        return self.controller._step(obs)

    @property
    def name(self):
        if self.controller:
            return self.controller.name
        else:
            return "Player"


class ForagingEnv(gym.Env):
    """
    Level-Based Foraging with SMAC-style centered coord normalization and zero padding.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 5,
    }

    action_set = [Action.NORTH, Action.SOUTH, Action.WEST, Action.EAST, Action.LOAD]
    Observation = namedtuple(
        "Observation",
        ["field", "actions", "players", "game_over", "sight", "current_step"],
    )
    PlayerObservation = namedtuple(
        "PlayerObservation", ["position", "level", "history", "reward", "is_self"]
    )

    def __init__(
        self,
        players,
        min_player_level,
        max_player_level,
        min_food_level,
        max_food_level,
        field_size,
        max_num_food,
        sight,
        max_episode_steps,
        force_coop,
        normalize_reward=True,
        grid_observation=False,
        observe_agent_levels=True,
        penalty=0.0,
        render_mode=None,
    ):
        self.logger = logging.getLogger(__name__)
        self.render_mode = render_mode
        self.players = [Player() for _ in range(players)]

        self.field = np.zeros(field_size, np.int32)

        self.penalty = penalty

        if isinstance(min_food_level, Iterable):
            assert (
                len(min_food_level) == max_num_food
            ), "min_food_level must be a scalar or a list of length max_num_food"
            self.min_food_level = np.array(min_food_level)
        else:
            self.min_food_level = np.array([min_food_level] * max_num_food)

        if max_food_level is None:
            self.max_food_level = None
        elif isinstance(max_food_level, Iterable):
            assert (
                len(max_food_level) == max_num_food
            ), "max_food_level must be a scalar or a list of length max_num_food"
            self.max_food_level = np.array(max_food_level)
        else:
            self.max_food_level = np.array([max_food_level] * max_num_food)

        if self.max_food_level is not None:
            for min_food_level_i, max_food_level_i in zip(
                self.min_food_level, self.max_food_level
            ):
                assert (
                    min_food_level_i <= max_food_level_i
                ), "min_food_level must be <= max_food_level for each food"

        self.max_num_food = max_num_food
        self._food_spawned = 0.0

        if isinstance(min_player_level, Iterable):
            assert (
                len(min_player_level) == players
            ), "min_player_level must be a scalar or a list of length players"
            self.min_player_level = np.array(min_player_level)
        else:
            self.min_player_level = np.array([min_player_level] * players)

        if isinstance(max_player_level, Iterable):
            assert (
                len(max_player_level) == players
            ), "max_player_level must be a scalar or a list of length players"
            self.max_player_level = np.array(max_player_level)
        else:
            self.max_player_level = np.array([max_player_level] * players)

        if self.max_player_level is not None:
            for i, (min_pl, max_pl) in enumerate(
                zip(self.min_player_level, self.max_player_level)
            ):
                assert (
                    min_pl <= max_pl
                ), f"min_player_level must be <= max_player_level for each player but was {min_pl} > {max_pl} for player {i}"

        self.sight = sight
        self.force_coop = force_coop
        self._game_over = None

        self._rendering_initialized = False
        self._valid_actions = None
        self._max_episode_steps = max_episode_steps

        self._normalize_reward = normalize_reward
        self._grid_observation = grid_observation
        self._observe_agent_levels = observe_agent_levels

        self.action_space = gym.spaces.Tuple(
            tuple([gym.spaces.Discrete(6)] * len(self.players))
        )
        self.observation_space = gym.spaces.Tuple(
            tuple([self._get_observation_space()] * len(self.players))
        )

        self.viewer = None
        self.n_agents = len(self.players)

        self._max_food_level_scalar = 1.0  # updated in reset()

    # ------------------------------
    # normalization helpers (SMAC-style)
    # ------------------------------
    def _center_norm_coord(self, y: float, x: float) -> Tuple[float, float]:
        rows, cols = self.rows, self.cols
        cy = (rows - 1) / 2.0 if rows > 1 else 0.0
        cx = (cols - 1) / 2.0 if cols > 1 else 0.0
        dy = cy if cy > 0 else 1.0
        dx = cx if cx > 0 else 1.0
        ny = (y - cy) / dy
        nx = (x - cx) / dx
        ny = float(np.clip(ny, -1.0, 1.0))
        nx = float(np.clip(nx, -1.0, 1.0))
        return ny, nx

    def _norm_agent_level(self, lvl: float) -> float:
        return float(lvl) / float(max(self.max_player_level)) if lvl > 0 else 0.0

    def _norm_food_level(self, lvl: float) -> float:
        return float(lvl) / float(self._max_food_level_scalar) if lvl > 0 else 0.0

    # ------------------------------
    # gym plumbing
    # ------------------------------
    def seed(self, seed=None):
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)
            self.np_random = self._np_random

    def _get_observation_space(self):
        if not self._grid_observation:
            # vector observation: (y,x,foodlvl)*max_food + (y,x[,lvl])*n_players
            player_obs_len = 3 if self._observe_agent_levels else 2
            dim = 3 * self.max_num_food + player_obs_len * len(self.players)
            low = np.full((dim,), -1.0, dtype=np.float32)
            high = np.ones((dim,), dtype=np.float32)
            # level slots low = 0
            for i in range(self.max_num_food):
                low[3 * i + 2] = 0.0
            if self._observe_agent_levels:
                base = 3 * self.max_num_food
                for i in range(len(self.players)):
                    low[base + 3 * i + 2] = 0.0
            return gym.spaces.Box(low=low, high=high, dtype=np.float32)
        else:
            g = 1 + 2 * self.sight
            low = np.zeros((3, g, g), dtype=np.float32)
            high = np.ones((3, g, g), dtype=np.float32)
            return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    @classmethod
    def from_obs(cls, obs):
        raise NotImplementedError

    @property
    def field_size(self):
        return self.field.shape

    @property
    def rows(self):
        return self.field_size[0]

    @property
    def cols(self):
        return self.field_size[1]

    @property
    def game_over(self):
        return self._game_over

    # ------------------------------
    # env helpers
    # ------------------------------
    def _gen_valid_moves(self):
        self._valid_actions = {
            player: [
                action for action in Action if self._is_valid_action(player, action)
            ]
            for player in self.players
        }

    def neighborhood(self, row, col, distance=1, ignore_diag=False):
        if not ignore_diag:
            return self.field[
                max(row - distance, 0) : min(row + distance + 1, self.rows),
                max(col - distance, 0) : min(col + distance + 1, self.cols),
            ]
        return (
            self.field[
                max(row - distance, 0) : min(row + distance + 1, self.rows), col
            ].sum()
            + self.field[
                row, max(col - distance, 0) : min(col + distance + 1, self.cols)
            ].sum()
        )

    def adjacent_food(self, row, col):
        return (
            self.field[max(row - 1, 0), col]
            + self.field[min(row + 1, self.rows - 1), col]
            + self.field[row, max(col - 1, 0)]
            + self.field[row, min(col + 1, self.cols - 1)]
        )

    def adjacent_food_location(self, row, col):
        if row > 1 and self.field[row - 1, col] > 0:
            return row - 1, col
        elif row < self.rows - 1 and self.field[row + 1, col] > 0:
            return row + 1, col
        elif col > 1 and self.field[row, col - 1] > 0:
            return row, col - 1
        elif col < self.cols - 1 and self.field[row, col + 1] > 0:
            return row, col + 1

    def adjacent_players(self, row, col):
        return [
            player
            for player in self.players
            if (abs(player.position[0] - row) == 1 and player.position[1] == col)
            or (abs(player.position[1] - col) == 1 and player.position[0] == row)
        ]

    def spawn_food(self, max_num_food, min_levels, max_levels):
        food_count = 0
        attempts = 0
        min_levels = max_levels if self.force_coop else min_levels

        perm = self.np_random.permutation(max_num_food)
        min_levels = min_levels[perm]
        max_levels = max_levels[perm]

        while food_count < max_num_food and attempts < 1000:
            attempts += 1
            row = self.np_random.integers(1, self.rows - 1)
            col = self.np_random.integers(1, self.cols - 1)

            if (
                self.neighborhood(row, col).sum() > 0
                or self.neighborhood(row, col, distance=2, ignore_diag=True) > 0
                or not self._is_empty_location(row, col)
            ):
                continue

            self.field[row, col] = (
                min_levels[food_count]
                if min_levels[food_count] == max_levels[food_count]
                else self.np_random.integers(
                    min_levels[food_count], max_levels[food_count] + 1
                )
            )
            food_count += 1
        self._food_spawned = self.field.sum()

    def _is_empty_location(self, row, col):
        if self.field[row, col] != 0:
            return False
        for a in self.players:
            if a.position and row == a.position[0] and col == a.position[1]:
                return False
        return True

    def spawn_players(self, min_player_levels, max_player_levels):
        perm = self.np_random.permutation(len(self.players))
        min_player_levels = min_player_levels[perm]
        max_player_levels = max_player_levels[perm]
        for player, min_pl, max_pl in zip(
            self.players, min_player_levels, max_player_levels
        ):
            attempts = 0
            player.reward = 0
            while attempts < 1000:
                row = self.np_random.integers(0, self.rows)
                col = self.np_random.integers(0, self.cols)
                if self._is_empty_location(row, col):
                    player.setup(
                        (row, col),
                        self.np_random.integers(min_pl, max_pl + 1),
                        self.field_size,
                    )
                    break
                attempts += 1

    def _is_valid_action(self, player, action):
        if action == Action.NONE:
            return True
        elif action == Action.NORTH:
            return (
                player.position[0] > 0
                and self.field[player.position[0] - 1, player.position[1]] == 0
            )
        elif action == Action.SOUTH:
            return (
                player.position[0] < self.rows - 1
                and self.field[player.position[0] + 1, player.position[1]] == 0
            )
        elif action == Action.WEST:
            return (
                player.position[1] > 0
                and self.field[player.position[0], player.position[1] - 1] == 0
            )
        elif action == Action.EAST:
            return (
                player.position[1] < self.cols - 1
                and self.field[player.position[0], player.position[1] + 1] == 0
            )
        elif action == Action.LOAD:
            return self.adjacent_food(*player.position) > 0

        self.logger.error("Undefined action {} from {}".format(action, player.name))
        raise ValueError("Undefined action")

    def _transform_to_neighborhood(self, center, sight, position):
        return (
            position[0] - center[0] + min(sight, center[0]),
            position[1] - center[1] + min(sight, center[1]),
        )

    def get_valid_actions(self) -> list:
        return list(product(*[self._valid_actions[player] for player in self.players]))

    # ------------------------------
    # obs/state construction (SMAC-style fixed slots + zero padding)
    # ------------------------------
    def _visible_to(self, center: Tuple[int, int], target: Tuple[int, int]) -> bool:
        cy, cx = center
        ty, tx = target
        return (abs(ty - cy) <= self.sight) and (abs(tx - cx) <= self.sight)

    def _make_obs(self, player):
        return self.Observation(
            actions=self._valid_actions[player],
            players=[
                self.PlayerObservation(
                    position=self._transform_to_neighborhood(
                        player.position, self.sight, a.position
                    ),
                    level=a.level,
                    is_self=a == player,
                    history=a.history,
                    reward=a.reward if a == player else None,
                )
                for a in self.players
                if (
                    min(
                        self._transform_to_neighborhood(
                            player.position, self.sight, a.position
                        )
                    )
                    >= 0
                )
                and max(
                    self._transform_to_neighborhood(
                        player.position, self.sight, a.position
                    )
                )
                <= 2 * self.sight
            ],
            field=np.copy(self.neighborhood(*player.position, self.sight)),
            game_over=self.game_over,
            sight=self.sight,
            current_step=self.current_step,
        )

    def _make_gym_obs(self):
        def make_obs_array_for_agent(agent_idx: int):
            obs_dim = self.observation_space[0].shape[0]
            out = np.zeros((obs_dim,), dtype=np.float32)

            self_player = self.players[agent_idx]
            cy, cx = self_player.position

            # foods: pick up to K nearest visible, sorted by distance, fill rest zeros
            foods = list(zip(*np.nonzero(self.field)))  # [(y,x), ...]
            visible_foods = []
            for (fy, fx) in foods:
                if self._visible_to((cy, cx), (fy, fx)):
                    d = (fy - cy) ** 2 + (fx - cx) ** 2
                    lvl = float(self.field[fy, fx])
                    ny, nx = self._center_norm_coord(float(fy), float(fx))
                    nl = self._norm_food_level(lvl)
                    visible_foods.append((d, ny, nx, nl))
            visible_foods.sort(key=lambda t: t[0])

            k = min(self.max_num_food, len(visible_foods))
            for i in range(k):
                _, ny, nx, nl = visible_foods[i]
                out[3 * i] = ny
                out[3 * i + 1] = nx
                out[3 * i + 2] = nl
            # remaining foods are already zeros

            # agents: slots = self first, then others in index order
            base = 3 * self.max_num_food
            if self._observe_agent_levels:
                # self
                sny, snx = self._center_norm_coord(float(cy), float(cx))
                sl = self._norm_agent_level(float(self_player.level))
                out[base + 0] = sny
                out[base + 1] = snx
                out[base + 2] = sl
                # others
                slot = 1
                for j, p in enumerate(self.players):
                    if j == agent_idx:
                        continue
                    if self._visible_to((cy, cx), p.position):
                        ny, nx = self._center_norm_coord(float(p.position[0]), float(p.position[1]))
                        nl = self._norm_agent_level(float(p.level))
                        out[base + 3 * slot + 0] = ny
                        out[base + 3 * slot + 1] = nx
                        out[base + 3 * slot + 2] = nl
                    # else keep zeros
                    slot += 1
            else:
                # self
                sny, snx = self._center_norm_coord(float(cy), float(cx))
                out[base + 0] = sny
                out[base + 1] = snx
                # others
                slot = 1
                for j, p in enumerate(self.players):
                    if j == agent_idx:
                        continue
                    if self._visible_to((cy, cx), p.position):
                        ny, nx = self._center_norm_coord(float(p.position[0]), float(p.position[1]))
                        out[base + 2 * slot + 0] = ny
                        out[base + 2 * slot + 1] = nx
                    slot += 1

            return out

        if self._grid_observation:
            # keep original grid mode unchanged
            def make_global_grid_arrays():
                grid_shape_x, grid_shape_y = self.field_size
                grid_shape_x += 2 * self.sight
                grid_shape_y += 2 * self.sight

                agents_layer = np.zeros((grid_shape_x, grid_shape_y), dtype=np.float32)
                for player in self.players:
                    px, py = player.position
                    val = self._norm_agent_level(player.level) if self._observe_agent_levels else 1.0
                    agents_layer[px + self.sight, py + self.sight] = val

                foods_layer = np.zeros((grid_shape_x, grid_shape_y), dtype=np.float32)
                foods_layer[self.sight : -self.sight, self.sight : -self.sight] = (
                    self.field.astype(np.float32) / float(self._max_food_level_scalar)
                )

                access_layer = np.ones((grid_shape_x, grid_shape_y), dtype=np.float32)
                access_layer[: self.sight, :] = 0.0
                access_layer[-self.sight :, :] = 0.0
                access_layer[:, : self.sight] = 0.0
                access_layer[:, -self.sight :] = 0.0
                for player in self.players:
                    px, py = player.position
                    access_layer[px + self.sight, py + self.sight] = 0.0
                foods_x, foods_y = self.field.nonzero()
                for x, y in zip(foods_x, foods_y):
                    access_layer[x + self.sight, y + self.sight] = 0.0

                return np.stack([agents_layer, foods_layer, access_layer])

            def get_agent_grid_bounds(agent_x, agent_y):
                return (
                    agent_x,
                    agent_x + 2 * self.sight + 1,
                    agent_y,
                    agent_y + 2 * self.sight + 1,
                )

            layers = make_global_grid_arrays()
            agents_bounds = [
                get_agent_grid_bounds(*player.position) for player in self.players
            ]
            nobs = tuple(
                [
                    layers[:, sx:ex, sy:ey]
                    for sx, ex, sy, ey in agents_bounds
                ]
            )
        else:
            nobs = tuple([make_obs_array_for_agent(i) for i in range(len(self.players))])

        for i, obs in enumerate(nobs):
            assert self.observation_space[i].contains(obs), \
                f"obs space error: obs: {obs}, obs_space: {self.observation_space[i]}"

        return nobs

    def _make_compact_global_state(self) -> np.ndarray:
        """
        Global state with fixed slots + zero padding:
        foods: take up to max_num_food in stable (y,x) order, rest zeros;
        agents: all agents in index order; coords centered in [-1,1], levels in [0,1].
        """
        state = []
        foods_coords = list(zip(*np.nonzero(self.field)))  # [(y,x), ...]
        foods_coords.sort(key=lambda t: (t[0], t[1]))  # stable order

        # foods
        take = min(self.max_num_food, len(foods_coords))
        for i in range(take):
            y, x = foods_coords[i]
            ny, nx = self._center_norm_coord(float(y), float(x))
            lvl = float(self.field[y, x])
            state.extend([ny, nx, self._norm_food_level(lvl)])
        for _ in range(self.max_num_food - take):
            state.extend([0.0, 0.0, 0.0])

        # agents (all, absolute, index order)
        for p in self.players:
            y, x = p.position
            ny, nx = self._center_norm_coord(float(y), float(x))
            lvl = float(p.level) if self._observe_agent_levels else 1.0
            state.extend([ny, nx, self._norm_agent_level(lvl)])

        return np.asarray(state, dtype=np.float32)

    def _get_info(self):
        return {"state": self._make_compact_global_state()}

    # ------------------------------
    # gym API
    # ------------------------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed, options=options)

        if not hasattr(self, "np_random"):
            self.np_random, _ = seeding.np_random(seed)

        self.field = np.zeros(self.field_size, np.int32)
        self.spawn_players(self.min_player_level, self.max_player_level)
        player_levels = sorted([player.level for player in self.players])

        if self.max_food_level is not None:
            max_food_scalar = int(max(self.max_food_level))
        else:
            max_food_scalar = int(sum(player_levels[:3]))
        self._max_food_level_scalar = max(1, max_food_scalar)

        self.spawn_food(
            self.max_num_food,
            min_levels=self.min_food_level,
            max_levels=self.max_food_level
            if self.max_food_level is not None
            else np.array([self._max_food_level_scalar] * self.max_num_food),
        )
        self.current_step = 0
        self._game_over = False
        self._gen_valid_moves()

        nobs = self._make_gym_obs()
        return nobs, self._get_info()

    def step(self, actions):
        self.current_step += 1

        for p in self.players:
            p.reward = 0

        actions = [
            Action(a) if Action(a) in self._valid_actions[p] else Action.NONE
            for p, a in zip(self.players, actions)
        ]

        for i, (player, action) in enumerate(zip(self.players, actions)):
            if action not in self._valid_actions[player]:
                self.logger.info(
                    "{}{} attempted invalid action {}.".format(
                        player.name, player.position, action
                    )
                )
                actions[i] = Action.NONE

        loading_players = set()

        # move & collisions
        collisions = defaultdict(list)
        for player, action in zip(self.players, actions):
            if action == Action.NONE:
                collisions[player.position].append(player)
            elif action == Action.NORTH:
                collisions[(player.position[0] - 1, player.position[1])].append(player)
            elif action == Action.SOUTH:
                collisions[(player.position[0] + 1, player.position[1])].append(player)
            elif action == Action.WEST:
                collisions[(player.position[0], player.position[1] - 1)].append(player)
            elif action == Action.EAST:
                collisions[(player.position[0], player.position[1] + 1)].append(player)
            elif action == Action.LOAD:
                collisions[player.position].append(player)
                loading_players.add(player)

        for k, v in collisions.items():
            if len(v) > 1:
                continue
            v[0].position = k

        # process LOAD
        while loading_players:
            player = loading_players.pop()
            loc = self.adjacent_food_location(*player.position)
            if loc is None:
                continue
            frow, fcol = loc
            food = self.field[frow, fcol]

            adj_players = self.adjacent_players(frow, fcol)
            adj_players = [p for p in adj_players if p in loading_players or p is player]

            adj_player_level = sum([a.level for a in adj_players])
            loading_players = loading_players - set(adj_players)

            if adj_player_level < food:
                for a in adj_players:
                    a.reward -= self.penalty
                continue

            for a in adj_players:
                a.reward = float(a.level * food)
                if self._normalize_reward:
                    a.reward = a.reward / float(adj_player_level * self._food_spawned)
            self.field[frow, fcol] = 0  # consumed

        self._game_over = (
            self.field.sum() == 0 or self._max_episode_steps <= self.current_step
        )
        self._gen_valid_moves()

        for p in self.players:
            p.score += p.reward

        rewards = [p.reward for p in self.players]
        done = self._game_over
        truncated = False
        info = self._get_info()
        return self._make_gym_obs(), rewards, done, truncated, info

    # ------------------------------
    # rendering
    # ------------------------------
    def _init_render(self):
        from .rendering import Viewer
        self.viewer = Viewer((self.rows, self.cols))
        self._rendering_initialized = True

    def render(self):
        if not self._rendering_initialized:
            self._init_render()
        return self.viewer.render(self, return_rgb_array=self.render_mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
