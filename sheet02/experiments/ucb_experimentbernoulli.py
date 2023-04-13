import matplotlib.pyplot as plt
import numpy as np

from sheet01.environments.multiarmed_bandits import BernoulliBanditEnv
from sheet02.experiments.trainmultiarmed import train_multiarmed
from sheet02.models.mutliarmedmodels import UCB

MAX_STEPS = 1000
N_ARMS = 10
USED_DELTA = 0.000001
NUM_GAMES = 3000
USED_PREFACTORS = [0.1, 0.5, 0.7]


def ucb_exp(max_steps, n_arms, used_delta, used_prefactors, num_games, printed):
    statistics_mean = {}
    statistics_cumsum = {}
    statistics_regrets = {}
    statistics_optimalities = {}

    for used_prefactor in used_prefactors:
        agent = UCB(delta=used_delta, n_arms=n_arms, prefactor=used_prefactor)
        rewards = np.zeros(shape=(num_games, max_steps))
        regrets = np.zeros(shape=(num_games, max_steps))
        optimalities = np.zeros(shape=(num_games, max_steps))
        for game in range(num_games):
            # mean_parameter = np.random.normal(
            #     loc=0.0, scale=1.0, size=n_arms).tolist()
            mean_parameter = np.random.uniform(
                low=0.0, high=1.0, size=n_arms).tolist()
            env = BernoulliBanditEnv(
                mean_parameter=mean_parameter, max_steps=max_steps)
            agent.reset()
            reward, _chosen_arms, regret, optimality = train_multiarmed(
                agent=agent, env=env, num_games=1, parameter="delta", printed=False)
            rewards[game,] = reward
            regrets[game,] = regret
            optimalities[game,] = optimality

        mean_rewards = np.mean(rewards, axis=0)
        mean_cum_rewards = np.cumsum(mean_rewards)
        mean_regrets = np.mean(regrets, axis=0)
        mean_optimalities = np.mean(optimalities, axis=0)
        index_array = np.arange(len(mean_optimalities))
        mean_optimalities = mean_optimalities / (index_array + 1)

        statistics_mean[str(used_prefactor)] = mean_rewards
        statistics_cumsum[str(used_prefactor)] = mean_cum_rewards
        statistics_regrets[str(used_prefactor)] = mean_regrets
        statistics_optimalities[str(used_prefactor)] = mean_optimalities

        # print statistics in console
        print(50 * "*")
        print(
            f"total mean reward with prefactor= {used_prefactor} is {mean_cum_rewards[-1]}")
        print(
            f"total regret with prefactor= {used_prefactor} is {mean_regrets[-1]}")
        print(
            f"total optimality with prefactor= {used_prefactor} is {mean_optimalities[-1]}")
        print(50 * "*")

    if printed:
        plt.subplot(4, 1, 1)
        for used_prefactor, traj in statistics_mean.items():
            plt.plot(range(len(traj)), traj,
                     label=f"mean reward, prefactor {used_prefactor}")
            plt.legend()
        plt.subplot(4, 1, 2)
        for used_prefactor, traj in statistics_cumsum.items():
            plt.plot(range(len(traj)), traj,
                     label=f"cumsum reward, prefactor {used_prefactor}")
            plt.legend()
        plt.subplot(4, 1, 3)
        for used_prefactor, traj in statistics_regrets.items():
            plt.plot(range(len(traj)), traj,
                     label=f"regrets, prefactor {used_prefactor}")
            plt.legend()
        plt.subplot(4, 1, 4)
        for used_prefactor, traj in statistics_optimalities.items():
            plt.plot(range(len(traj)), traj,
                     label=f"optimalities, prefactor {used_prefactor}")
            plt.legend()
        plt.show()

    return statistics_mean, statistics_cumsum, statistics_regrets, statistics_optimalities


if __name__ == "__main__":
    ucb_exp(max_steps=MAX_STEPS, n_arms=N_ARMS,
            used_delta=USED_DELTA, used_prefactors=USED_PREFACTORS, num_games=NUM_GAMES, printed=True)
