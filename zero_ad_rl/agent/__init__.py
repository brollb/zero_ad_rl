from ray.rllib.agents import registry
from ray.rllib.policy.policy import Policy
from ray.rllib.agents.trainer_template import build_trainer

class KitingPolicy(Policy):
    def __init__(self, obs_space, action_space, config):
        Policy.__init__(self, obs_space, action_space, config)
        self.distance = 0.25

    def compute_actions(self,
                    obs_batch,
                    state_batches,
                    prev_action_batch=None,
                    prev_reward_batch=None,
                    info_batch=None,
                    episodes=None,
                    **kwargs):
        return [0 if dist < self.distance else 1 for dist in obs_batch], [], {}
    
    def learn_on_batch(self, samples):
        return {}

    def get_weights(self):
        return {'distance': self.distance}

    def set_weights(self, weights):
        if 'distance' in weights:
            self.distance = weights['distance']

KitingAgent = build_trainer(
    name='Kiting',
    default_policy=KitingPolicy,
)
zero_ad_agents = {
    'Kiting': KitingAgent,
}

def get_agent_class(name):
    return zero_ad_agents[name] if name in zero_ad_agents else registry.get_agent_class(name)
