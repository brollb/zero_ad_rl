import ray
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('states', help='states.jsonl file to generate demonstration from')
    # TODO: should we use actions?
    parser.add_argument('actions', help='actions.jsonl file to generate demonstration from')
    parser.add_argument('--run')
    parser.add_argument('--env')
    parser.add_argument('--url', default='http://127.0.0.1:6000',
            help='0 AD game server URL (running with --rlinterface flag)')
    parser.add_argument('--out', default='demonstrations')
    parser.add_argument('--target-env')

    args = parser.parse_args()

    ray.init()
    cls = get_agent_class(args.run)
    agent = cls(env=args.env)
    agent.restore(args.checkpoint)

    # TODO:
    env = agent.workers.local_worker().env

    # TODO: Read in states
    # TODO: Read in the states or the demonstration file?
    prev_obs = None
    prev_action = None
    batch_builder = SampleBatchBuilder()
    writer = JsonWriter(args.out)
    for states_path in args.states:
        with open(states_path, 'r') as states_file:
            states = (GameState(json.loads(line), env.game) for line in states_file)
            trajectory = []
            for (t, state) in enumerate(states):
                if is_game_over(state):
                    continue

                env.prev_state = env.state
                env.state = state
                obs = env.observation(state)
                action = agent.compute_action(obs)
                command = env.resolve_action(action)
                if target_env:
                    target_env.prev_state = target_env.state
                    target_env.state = state
                    obs = target_env.observation(state)
                    action = closest_action(target_env, command)

                if prev_obs is not None:
                    trajectory.append([prev_obs, prev_action, obs])
                    batch_builder.add_values(
                        t=t,
                        eps_id=0,
                        agent_index=0,
                        obs=prev_obs,
                        actions=action,
                        action_prob=1.0,  # put the true action probability here
                        rewards=0,
                        prev_actions=prev_action,
                        prev_rewards=0,
                        dones=False,  # TODO
                        infos=None,
                        new_obs=obs)

                prev_obs = obs
                prev_action = action
        writer.write(batch_builder.build_and_reset())
