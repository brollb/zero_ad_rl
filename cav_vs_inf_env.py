from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image
import gym
import math
from gym.spaces import Discrete, Box
import numpy as np
import zero_ad
from os import path
import json

class StateBuilder():
    def __init__(self, space):
        self.space = space

    def from_json(self, state):
        pass

    def to_image(self, state):
        return Image.fromarray(self.from_json(state))

class ActionBuilder():
    def __init__(self, space):
        self.space = space

    def to_json(self, action, state):
        pass

    def to_image(self, action):
        pass

class BaseZeroADEnv(gym.Env):
    def __init__(self, config, action_builder, state_builder):
        self.actions = action_builder
        self.states = state_builder
        self.action_space = self.actions.space
        self.observation_space = self.states.space
        self.config = config
        self.step_count = 8
        server_address = self.address(config.worker_index)
        self.game = zero_ad.ZeroAD(server_address)
        self.prev_state = None
        self.state = None
        self.cum_reward = 0

    def address(self, worker_index):
        port = 6000 + worker_index
        return f'http://127.0.0.1:{port}'

    def reset(self):
        self.prev_state = self.game.reset(self.scenario_config())
        self.state = self.game.step([zero_ad.actions.reveal_map()])
        return self.observation(self.state)

    def step(self, action_index):
        action = self.actions.to_json(action_index, self.state)
        self.prev_state = self.state
        self.state = self.game.step([action])
        for _ in range(self.step_count - 1):
            self.state = self.game.step()

        player_states = [player['state'] for player in self.state.data['players']]
        players_finished = [state != 'active' for state in player_states]
        done = any(players_finished)
        reward = self.reward(self.prev_state, self.state)
        self.cum_reward += reward
        if done:
            stats = self.episode_complete_stats(self.state)
            stats_str = ' '
            for (k, v) in stats.items():
                stats_str += k + ': ' + str(v) + '; '

            print(f'episode complete.{stats_str}')
            self.cum_reward = 0

        return self.observation(self.state), reward, done, {}

    def episode_complete_stats(self, state):
        stats = {}
        stats['reward'] = self.cum_reward
        stats['win'] = self.get_player_state(state, 2) == 'defeated'
        return stats

    def get_player_state(self, state, index):
        return state.data['players'][index]['state']

    def reward(self, prev_state, state):
        if self.get_player_state(state, 1) == 'defeated':
            return -1
        elif self.get_player_state(state, 2) == 'defeated':
            return 1
        else:
            return 0

    def observation(self, state):
        return self.states.from_json(state)

    def scenario_config(self):
        pass

def center(units):
    positions = np.array([unit.position() for unit in units])
    return np.mean(positions, axis=0)

def enemy_offset(state):
    player_units = state.units(owner=1)
    enemy_units = state.units(owner=2)
    return center(enemy_units) - center(player_units)

class EnemyDistance(StateBuilder):
    def __init__(self):
        space = Box(0.0, 1.0, shape=(1, ), dtype=np.float32)
        super().__init__(space)

    def from_json(self, state):
        dist = np.linalg.norm(enemy_offset(state))
        max_dist = 80
        normalized_dist = dist/max_dist if not np.isnan(dist/max_dist) else 1.
        return np.array([min(normalized_dist, 1.)])

class AttackRetreat(ActionBuilder):
    def __init__(self):
        super().__init__(Discrete(2))

    def to_json(self, action_index, state):
        return self.retreat(state) if action_index == 0 else self.attack(state)

    def retreat(self, state):
        units = state.units(owner=1)
        center_pt = center(units)
        offset = enemy_offset(state)
        rel_position = 20 * (offset / np.linalg.norm(offset, ord=2))
        position = list(center_pt - rel_position)
        return zero_ad.actions.walk(units, *position)

    def attack(self, state):
        units = state.units(owner=1)
        center_pt = center(units)

        enemy_units = state.units(owner=2)
        enemy_positions = np.array([unit.position() for unit in enemy_units])
        dists = np.linalg.norm(enemy_positions - center_pt, ord=2, axis=1)
        closest_index = np.argmin(dists)
        closest_enemy = enemy_units[closest_index]

        return zero_ad.actions.attack(units, closest_enemy)

class CavalryVsInfantryEnv(BaseZeroADEnv):
    def __init__(self, config):
        super().__init__(config, AttackRetreat(), EnemyDistance())

    def scenario_config_file(self):
        return 'CavalryVsInfantry.json'

    def scenario_config(self):
        configs_dir = path.join(path.dirname(path.realpath(__file__)), 'scenario-configs')
        filename = self.scenario_config_file()
        config_path = path.join(configs_dir, filename)
        with open(config_path) as f:
            config = f.read()
        return config

class Minimap(StateBuilder):
    def __init__(self):
        space = Box(0.0, 1.0, shape=(84, 84, 3), dtype=np.float32)
        super.__init__(space)

    def from_json(self, state):
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

class SimpleMinimapCavVsInfEnv(BaseZeroADEnv):
    def __init__(self, config):
        super().__init__(config, AttackRetreat(), Minimap())

class AttackAndMove(AttackRetreat):
    def __init__(self):
        space = Discrete(9)
        super.__init__(space)

    def to_json(self, action_index, state):
        if action_index == 8:
            return self.attack(state)
        else:
            return self.move(state, 2 * math.pi * action_index/8)

    def move(self, state, angle, distance=15):
        units = state.units(owner=1)
        center_pt = center(units)

        offset = distance * np.array([math.cos(angle), math.sin(angle)])
        position = list(center_pt + offset)

        return zero_ad.actions.walk(units, *position)


class MinimapCavVsInfEnv(BaseZeroADEnv):
    def __init__(self, config):
        super().__init__(config, AttackAndMove(), Minimap())
        self.level = config.get('level', 1)
        self.caution_factor = 10

    def on_train_result(self, mean_reward):
        max_reward = self.max_reward()
        min_reward = self.min_reward()
        percent_to_advance = 0.85
        reward_to_advance = min_reward + percent_to_advance * (max_reward - min_reward)
        if mean_reward > reward_to_advance:
            self.level += 1
            print('advancing to level', self.level)
            if self.level > 5:
                self.caution_factor = 5

    def scenario_config_file(self):
        if self.level < 7:
            return 'CavalryVsInfantryL' + str(self.level)+ '.json'
        else:
            return 'CavalryVsInfantry.json'

    def player_unit_health(self, state, owner=1):
        return sum(( unit.health(True) for unit in state.units(owner=owner)))

    def reward(self, prev_state, state):
        return self.damage_diff(prev_state, state) - 0.0001

    def max_reward(self):
        enemy_units = min(self.level, 7)
        return enemy_units

    def min_reward(self):
        player_units = 5
        return -self.caution_factor * player_units

    def damage_diff(self, prev_state, state):
        prev_enemy_health = self.player_unit_health(prev_state, 2)
        enemy_health = self.player_unit_health(state, 2)
        enemy_damage = prev_enemy_health - enemy_health
        assert enemy_damage >= 0, f'Enemy damage is negative: {enemy_damage}'

        prev_player_health = self.player_unit_health(prev_state)
        player_health = self.player_unit_health(state)
        player_damage = prev_player_health - player_health
        assert player_damage >= 0, f'Player damage is negative: {player_damage}'
        return enemy_damage - self.caution_factor * player_damage

    def episode_complete_stats(self, state):
        stats = super().episode_complete_stats(state)
        stats['reward_ratio'] = (self.cum_reward - self.min_reward())/(self.max_reward() - self.min_reward())

        if stats['reward_ratio'] > 1:
            print('---------- Reward is above max expected value -----------')
            print(self.cum_reward, 'vs', self.max_reward())
            prev_enemy_health = self.player_unit_health(state, 2)
            print('enemy health:', prev_enemy_health)

        stats['level'] = self.level
        return stats
