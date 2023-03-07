import numpy as np
from numpy import savetxt
import matplotlib.pyplot as plt


class QLearningBox:
    def __init__(
        self, env, agent, alpha, gamma, epsilon, epsilon_min, epsilon_dec, episodes
    ):
        self.env = env
        self.agent = agent
        # discretizando o espaco de estados
        # self.num_states = np.array([1000, 1000])
        # self.num_states = np.round(self.num_states, 0).astype(int) + 1
        self.action_spaces = env.action_space(agent)
        self.observation_spaces = env.observation_space(agent)

        # inicializando uma q-table com 3 dimensoes: x, velocidade, acao
        self.Q = np.zeros([1000 + 1, 1000 + 1, self.action_spaces.n])

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.episodes = episodes

    def select_action(self, state_adj):
        if np.random.random() < 1 - self.epsilon:
            return np.argmax(self.Q[state_adj[0], state_adj[1]])
        return np.random.randint(0, self.action_spaces.n)

    def transform_state(self, state):
        state_adj = [1000, 1000]
        return np.round(state_adj, 0).astype(int)

    def train(self, filename, plotFile=None):
        # Initialize variables to track rewards
        reward_list = []
        ave_reward_list = []
        actions_per_episode = []

        # Run Q-learning algorithm
        for i in range(self.episodes):
            # Initialize parameters
            done = False
            tot_reward, reward = 0, 0
            state = self.env.reset()

            # discretizando o estado
            state_adj = self.transform_state(state)

            qtd_actions = 0
            while done != True:
                action = self.select_action(state_adj)
                self.env.step(action)

                state2 = self.env.state()
                reward = self.env.rewards[self.agent]
                done = self.env.truncations[self.agent]

                # Discretize state2
                state2_adj = self.transform_state(state2)

                delta = self.alpha * (
                    reward
                    + self.gamma * np.max(self.Q[state2_adj[0], state2_adj[1]])
                    - self.Q[state_adj[0], state_adj[1], action]
                )
                self.Q[state_adj[0], state_adj[1], action] += delta

                # Update variables
                tot_reward += reward
                state_adj = state2_adj
                qtd_actions = qtd_actions + 1

            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon = self.epsilon * self.epsilon_dec

            # Track rewards
            reward_list.append(tot_reward)

            if (i + 1) % 100 == 0:
                ave_reward = np.mean(reward_list)
                ave_reward_list.append(ave_reward)
                actions_per_episode.append(qtd_actions)
                reward_list = []

            if (i + 1) % 100 == 0:
                print(
                    "Episode {} Average Reward: {}  Actions in this episode {} ".format(
                        i + 1,
                        ave_reward,
                        actions_per_episode[len(actions_per_episode) - 1],
                    )
                )

        # TODO Q eh um vetor tridimensional. savetxt nao trabalha com arrays com mais de 2
        # dimensoes. Eh necessario fazer ajustes para armazenar a Q-table neste caso.
        # savetxt(filename, self.Q, delimiter=',')
        if plotFile is not None:
            self.plotactions(plotFile, ave_reward_list)
        return self.Q

    def plotactions(self, plotFile, rewards):
        plt.plot(100 * (np.arange(len(rewards)) + 1), rewards)
        plt.xlabel("Episodes")
        plt.ylabel("Average Reward")
        plt.title("Average Reward vs Episodes")
        plt.savefig(plotFile + ".jpg")
        plt.close()
