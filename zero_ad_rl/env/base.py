from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image
import gym
import zero_ad

class StateBuilder():
    def __init__(self, space):
        self.space = space

    def from_json(self, state):
        pass

    def to_image(self, state):
        arry = self.from_json(state)
        return Image.fromarray(arry)

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


