import numpy as np
import matplotlib.pyplot as plt

AGENT_ALPHA = 0.1
N_ARMS = 2
MEAN_PARAMETER = [0.1, 0.7] # probability parameter of bernoulli bandit problem
MAX_STEPS = 100  # maximal gamestep
NUM_GAMES = 5  # games played of the greedy algorithm


def train_gradientbandit(agent, env, num_games, printed):

    chosen_arms = np.zeros(shape=(num_games, env.max_steps))
    rewards = np.zeros(shape=(num_games, env.max_steps))
    regrets = np.zeros(shape=(num_games, env.max_steps))
    optimalities = np.zeros(shape=(num_games, env.max_steps))
    optimalities_percentage = np.zeros(shape=(num_games, env.max_steps))

    for game in range(num_games):
        # playing the algo for `num_games` rounds
        agent.reset()
        env.reset()
        done = False
        while (not done):
            # playing the game until it is done
            action = agent.select_arm()
            _new_state, reward, done, _info = env.step(action)
            rewards[game, (env.count-1)] = reward
            chosen_arms[game, (env.count-1)] = action
            regrets[game, (env.count-1)] = env.regret
            optimalities[game, (env.count-1)] = env.played_optimal
            optimalities_percentage[game,(env.count-1)] = agent.get_prob(env.optimal[1])

            agent.update(action, reward)

    if printed:
        plt.subplot(5, 1, 1)
        for i, reward in enumerate(rewards):
            plt.plot(range(env.max_steps), reward,
                     label=f"reward {i}, alpha {agent.alpha}")
            plt.legend()

        plt.subplot(5, 1, 2)
        for i, arm in enumerate(chosen_arms):
            plt.plot(range(env.max_steps), arm,
                     label=f"action sequence {i}, alpha {agent.alpha}")
            plt.legend()

        plt.subplot(5, 1, 3)
        for i, regret in enumerate(regrets):
            plt.plot(range(env.max_steps), regret,
                     label=f"regrets {i}, alpha {agent.alpha}")
            plt.legend()

        plt.subplot(5, 1, 4)
        for i, optim in enumerate(optimalities):
            plt.plot(range(env.max_steps), optim,
                     label=f"optimalities {i}, alpha {agent.alpha}")
            plt.legend()

        plt.subplot(5, 1, 5)
        for i, optim_percent in enumerate(optimalities_percentage):
            plt.plot(range(env.max_steps), optim_percent,
                     label=f"optimality prob {i}, alpha {agent.alpha}")
            plt.legend()

        plt.show()

    return rewards, chosen_arms, regrets, optimalities, optimalities_percentage
