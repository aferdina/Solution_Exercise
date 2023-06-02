""" run explore then commit algorithm with the optimum from upper bound
"""
import math
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from sheet01.environments.multiarmed_bandits import GaussianBanditEnv
from sheet01.models.explorethencommit import ExploreThenCommit
from sheet01.experiment.run_explorethencommit import train_exandcommit
from sheet01.experiment.trainingsutils import bound_function


def explorethencommit_optim(
    max_steps: int, n_arms: int, num_games: int, printed: bool
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """calculate the optimal explore parameter from the upper bound from the lecture

    Args:
        max_steps (int): number of steps to run the bandit
        n_arms (int): number of arms in the bandit model
        num_games (int): number of games to run the bandit model
        printed (bool): true, if information should be printed to screen

    Returns:
        Tuple: mean_rewards, mean_cum_rewards, mean_regrets, mean_optimalities
    """
    rewards = np.zeros(shape=(num_games, max_steps))
    regrets = np.zeros(shape=(num_games, max_steps))
    optimalities = np.zeros(shape=(num_games, max_steps))
    for game in range(num_games):
        mean_parameter = np.random.normal(loc=0.0, scale=1.0, size=n_arms).tolist()
        env = GaussianBanditEnv(mean_parameter=mean_parameter, max_steps=max_steps)
        calc_deltas = np.max(mean_parameter) - mean_parameter
        upperbound = max_steps / n_arms
        bounds = [1, upperbound]
        res = minimize_scalar(
            lambda x_explore: bound_function(
                explore=x_explore, deltas=calc_deltas, number_of_games=max_steps
            ),
            bounds=bounds,
        )
        if res.success:
            explore = math.ceil(res.x)
            # print("Optimal value: ", explore)
        else:
            # print("Optimization failed: ", res.message)
            raise KeyError
        agent = ExploreThenCommit(explore=explore, n_arms=n_arms)
        reward, _chosen_arms, regret, optimality = train_exandcommit(
            agent=agent, env=env, num_games=1, parameter="explore", printed=False
        )
        rewards[game,] = reward
        regrets[game,] = regret
        optimalities[game,] = optimality

    mean_rewards = np.mean(rewards, axis=0)
    mean_cum_rewards = np.cumsum(mean_rewards, axis=0)
    mean_regrets = np.mean(regrets, axis=0)
    mean_optimalities = np.mean(optimalities, axis=0)
    index_array = np.arange(len(mean_optimalities))
    mean_optimalities = mean_optimalities / (index_array + 1)

    print(f"total mean reward of optim is {mean_cum_rewards[-1]}")
    print(f"total mean regret of optim is {mean_regrets[-1]}")
    print(f"total number of optimal action of optim is {mean_optimalities[-1]}")
    if printed:
        plt.subplot(4, 1, 1)
        plt.plot(range(len(mean_rewards)), mean_rewards, label="mean reward optim m")
        plt.legend()
        plt.subplot(4, 1, 2)
        plt.plot(
            range(len(mean_cum_rewards)),
            mean_cum_rewards,
            label="cumsum reward optim m",
        )
        plt.legend()
        plt.subplot(4, 1, 3)
        plt.plot(range(len(mean_regrets)), mean_regrets, label="regrets optim m")
        plt.legend()
        plt.subplot(4, 1, 4)
        plt.plot(
            range(len(mean_optimalities)),
            mean_optimalities,
            label="optimalities optim m",
        )
        plt.legend()
        plt.show()

    return mean_rewards, mean_cum_rewards, mean_regrets, mean_optimalities


if __name__ == "__main__":
    NUM_GAMES = 5000
    MAX_STEPS = 10000
    N_ARMS = 10
    explorethencommit_optim(
        max_steps=MAX_STEPS, n_arms=N_ARMS, num_games=NUM_GAMES, printed=True
    )
