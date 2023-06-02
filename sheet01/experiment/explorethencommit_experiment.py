""" Run experiments with expore then commit algorithm for different explore parameters
"""
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from sheet01.environments.multiarmed_bandits import GaussianBanditEnv
from sheet01.models.explorethencommit import ExploreThenCommit
from sheet01.experiment.run_explorethencommit import train_exandcommit
from sheet01.experiment.trainingsutils import plot_statistics
MAX_STEPS = 1000
N_ARMS = 10
USED_EXPLORES = [5, 10, 15]
NUM_GAMES = 1000


def explorethencommit_exp(
    max_steps: int, n_arms: int, used_explores: List[int], num_games: int, printed: bool
):
    statistics_mean = {}
    statistics_cumsum = {}
    statistics_regrets = {}
    statistics_optimalities = {}

    for explore in used_explores:
        agent = ExploreThenCommit(explore=explore, n_arms=n_arms)
        rewards = np.zeros(shape=(num_games, max_steps))
        regrets = np.zeros(shape=(num_games, max_steps))
        optimalities = np.zeros(shape=(num_games, max_steps))
        for game in range(num_games):
            mean_parameter = np.random.normal(loc=0.0, scale=1.0, size=n_arms).tolist()
            env = GaussianBanditEnv(mean_parameter=mean_parameter, max_steps=max_steps)
            agent.reset()
            reward, _chosen_arms, regret, optimality = train_exandcommit(
                agent=agent, env=env, num_games=1, parameter="explore", printed=False
            )
            rewards[game,] = reward
            regrets[game,] = regret
            optimalities[game,] = optimality

        mean_rewards = np.mean(rewards, axis=0)
        mean_cum_rewards = np.cumsum(mean_rewards)
        mean_regrets = np.mean(regrets, axis=0)
        mean_optimalities = np.mean(optimalities, axis=0)
        index_array = np.arange(len(mean_optimalities))
        mean_optimalities = mean_optimalities / (index_array + 1)

        statistics_mean[str(explore)] = mean_rewards
        statistics_cumsum[str(explore)] = mean_cum_rewards
        statistics_regrets[str(explore)] = mean_regrets
        statistics_optimalities[str(explore)] = mean_optimalities

        # print statistics in console
        print(50 * "*")
        print(f"total mean reward with explore= {explore} is {mean_cum_rewards[-1]}")
        print(f"total regret with explore= {explore} is {mean_regrets[-1]}")
        print(f"total optimality with explore= {explore} is {mean_optimalities[-1]}")
        print(50 * "*")

    if printed:
        plt.subplot(4, 1, 1)
        for exploration, traj in statistics_mean.items():
            plt.plot(
                range(len(traj)),
                traj,
                label=f"mean reward, explore_steps {exploration}",
            )
            plt.legend()
        plt.subplot(4, 1, 2)
        for exploration, traj in statistics_cumsum.items():
            plt.plot(
                range(len(traj)),
                traj,
                label=f"cumsum reward, explore_steps {exploration}",
            )
            plt.legend()
        plt.subplot(4, 1, 3)
        for exploration, traj in statistics_regrets.items():
            plt.plot(
                range(len(traj)), traj, label=f"regrets, explore_steps {exploration}"
            )
            plt.legend()
        plt.subplot(4, 1, 4)
        for exploration, traj in statistics_optimalities.items():
            plt.plot(
                range(len(traj)),
                traj,
                label=f"optimalities, explore_steps {exploration}",
            )
            plt.legend()
        plt.show()

    return (
        statistics_mean,
        statistics_cumsum,
        statistics_regrets,
        statistics_optimalities,
    )


if __name__ == "__main__":
    explorethencommit_exp(
        max_steps=MAX_STEPS,
        n_arms=N_ARMS,
        used_explores=USED_EXPLORES,
        num_games=NUM_GAMES,
        printed=True,
    )
