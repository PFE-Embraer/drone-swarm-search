from pettingzoo.mpe import simple_v2
from QLearningBox import QLearningBox

env = simple_v2.env(max_cycles=10, continuous_actions=False)
env.reset()

for agent in env.agent_iter():
    env.reset()
    qlearn = QLearningBox(
        env,
        agent=agent,
        alpha=0.1,
        gamma=0.6,
        epsilon=0.7,
        epsilon_min=0.05,
        epsilon_dec=0.99,
        episodes=10000,
    )
    train_dir = "core/algorithms/QLeaning/"
    q_table = qlearn.train(
        filename=f"{train_dir}/qtable_{agent}.csv",
        plotFile=f"{train_dir}/qtable_{agent}",
    )

    break


#
# teste:
#
