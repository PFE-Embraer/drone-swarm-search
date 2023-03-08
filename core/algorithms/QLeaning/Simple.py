from pettingzoo.mpe import simple_v2
import sys

env = simple_v2.env(max_cycles=50, continuous_actions=False, render_mode='human')

def policy(observation, agent): 
    print(round(observation[0]*100))
    print(observation)
    return env.action_space(agent).sample()

env.reset()
termination = False

while not termination:
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        action = policy(observation, agent)
        if not truncation:
            env.step(action)
        else:
            sys.exit(1)