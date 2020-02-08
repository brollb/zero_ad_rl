from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
from gym.spaces import Discrete, Box
import numpy as np
import zero_ad
from zero_ad import MapType

class BaseZeroADEnv(gym.Env):
    def __init__(self):
        self.step_count = 8
        self.game = zero_ad.ZeroAD('0.0.0.0:50050')
        self.prev_state = None
        self.state = None

    def reset(self):
        self.prev_state = self.game.reset(self.scenario_config())
        self.state = self.game.step([zero_ad.actions.reveal_map()])
        return self.observation(self.state)

    def step(self, action_index):
        action = self.resolve_action(action_index)
        self.prev_state = self.state
        self.state = self.game.step([action])
        for _ in range(self.step_count - 1):
            self.state = self.game.step()

        player_states = [player['state'] for player in self.state.data['players']]
        players_finished = [state != 'active' for state in player_states]
        done = any(players_finished)
        reward = self.reward(self.prev_state, self.state)
        if done:
            print('episode complete. reward:', reward)
        return self.observation(self.state), reward, done, {}

    def observation(self, state):
        pass

    def scenario_config(self):
        pass

    def reward(self, prev_state, state):
        pass

    def resolve_action(self, action_index):
        pass

class CavalryVsInfantryEnv(BaseZeroADEnv):
    def __init__(self, config):
        super().__init__()
        self.action_space = Discrete(2)
        self.observation_space = Box(0.0, 1.0, shape=(1, ), dtype=np.float32)

    def resolve_action(self, action_index):
        return self.retreat() if action_index == 0 else self.attack()

    def retreat(self):
        units = self.state.units(owner=1)
        center = self.center(units)
        offset = self.enemy_offset(self.state)
        rel_position = 20 * (offset / np.linalg.norm(offset, ord=2))
        position = list(center - rel_position)
        return zero_ad.actions.walk(units, *position)

    def attack(self):
        units = self.state.units(owner=1)
        center = self.center(units)

        enemy_units = self.state.units(owner=2)
        enemy_positions = np.array([unit.position() for unit in enemy_units])
        dists = np.linalg.norm(enemy_positions - center, ord=2, axis=1)
        closest_index = np.argmin(dists)
        closest_enemy = enemy_units[closest_index]

        return zero_ad.actions.attack(units, closest_enemy)

    def scenario_config(self):
        config = zero_ad.ScenarioConfig('CavalryVsInfantry', type=MapType.SCENARIO)
        config.set_victory_conditions(zero_ad.VictoryConditions.CONQUEST_UNITS)
        config.add_player('Player 1', civ='spart', team=1)
        config.add_player('Player 2', civ='spart', team=2)
        return config

    def get_player_state(self, state, index):
        return state.data['players'][index]['state']

    def observation(self, state):
        dist = np.linalg.norm(self.enemy_offset(state))
        max_dist = 80
        normalized_dist = dist/max_dist if not np.isnan(dist/max_dist) else 1.
        return np.array([min(normalized_dist, 1.)])

    def enemy_offset(self, state):
        player_units = state.units(owner=1)
        enemy_units = state.units(owner=2)
        return self.center(enemy_units) - self.center(player_units)

    def center(self, units):
        positions = np.array([unit.position() for unit in units])
        return np.mean(positions, axis=0)

    def reward(self, prev_state, state):
        if self.get_player_state(state, 1) == 'defeated':
            return -1
        elif self.get_player_state(state, 2) == 'defeated':
            return 1
        else:
            return 0

class SimpleMinimapCavVsInfEnv(CavalryVsInfantryEnv):
    def __init__(self, config):
        super().__init__(config)
        self.observation_space = Box(0.0, 1.0, shape=(84, 84, 3), dtype=np.float32)

    def observation(self, state):
        obs = np.zeros((84, 84, 3))
        my_units = state.units(owner=1)
        center = self.center(my_units)
        if len(my_units) > 0:
            min_x = center[0] - 42
            max_x = center[0] + 42
            min_z = center[1] - 42
            max_z = center[1] + 42
            for unit in state.units():
                pos = unit.position()
                if min_x < pos[0] < max_x and min_z < pos[1] < max_z:
                    x = int(pos[0] - min_x)
                    z = int(pos[1] - min_z)
                    obs[x][z][int(unit.owner())] = 1.

        return obs

