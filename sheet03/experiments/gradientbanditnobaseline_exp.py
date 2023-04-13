import json

import matplotlib.pyplot as plt
import numpy as np

from sheet01.environments.multiarmed_bandits import GaussianBanditEnv
from sheet03.experiments.traingradientbandits import train_gradientbandit
from sheet03.models.multiarmedmodels import GradientBanditnobaseline

MAX_STEPS = 1000
N_ARMS = 10
# USED_ALPHA = np.arange(start=0.05, stop=0.5, step=0.05).tolist()
USED_ALPHA = [0.1, 0.2, 0.3]
NUM_GAMES = 3000


def gradientbanditnobaseline_exp(max_steps, n_arms, used_alpha, num_games, printed):
    statistics_mean = {}
    statistics_cumsum = {}
    statistics_regrets = {}
    statistics_optimalities = {}
    statistic_optimalities_percentage = {}

    for alpha in used_alpha:
        # IMPORTANT: boltzmanmConstant and gumbel lead to the same results
        agent = GradientBanditnobaseline(
            alpha=alpha, n_arms=n_arms)
        rewards = np.zeros(shape=(num_games, max_steps))
        regrets = np.zeros(shape=(num_games, max_steps))
        optimalities = np.zeros(shape=(num_games, max_steps))
        optimalities_percentage = np.zeros(shape=(num_games, max_steps))
        for game in range(num_games):
            mean_parameter = np.random.normal(
                loc=0.0, scale=1.0, size=n_arms).tolist()
            env = GaussianBanditEnv(
                mean_parameter=mean_parameter, max_steps=max_steps)
            agent.reset()
            reward, _chosen_arms, regret, optimality, optimality_percentage = train_gradientbandit(
                agent=agent, env=env, num_games=1, printed=False)
            rewards[game,] = reward
            regrets[game,] = regret
            optimalities[game,] = optimality
            optimalities_percentage[game,] = optimality_percentage

        mean_rewards = np.mean(rewards, axis=0)
        mean_cum_rewards = np.cumsum(mean_rewards)
        mean_regrets = np.mean(regrets, axis=0)
        mean_optimalities = np.mean(optimalities, axis=0)
        index_array = np.arange(len(mean_optimalities))
        mean_optimalities = mean_optimalities / (index_array + 1)
        mean_optimalities_percentage = np.mean(optimalities_percentage, axis=0)

        statistics_mean[str(alpha)] = mean_rewards.tolist()
        statistics_cumsum[str(alpha)] = mean_cum_rewards.tolist()
        statistics_regrets[str(alpha)] = mean_regrets.tolist()
        statistics_optimalities[str(alpha)] = mean_optimalities.tolist()
        statistic_optimalities_percentage[str(alpha)] = mean_optimalities_percentage.tolist()
        # print statistics in console
        print(50 * "*")
        print(
            f"total mean reward with alpha= {alpha} is {mean_cum_rewards[-1]}")
        print(
            f"total regret with alpha= {alpha} is {mean_regrets[-1]}")
        print(
            f"total optimality with alpha= {alpha} is {mean_optimalities[-1]}")
        print(
            f"total optimality action percentage with alpha= {alpha} is {mean_optimalities_percentage[-1]}")
        print(50 * "*")

    if printed:
        plt.subplot(5, 1, 1)
        for used_alpha, traj in statistics_mean.items():
            plt.plot(range(len(traj)), traj,
                     label=f"mean reward, alpha {used_alpha}")
            plt.legend()
        plt.subplot(5, 1, 2)
        for used_alpha, traj in statistics_cumsum.items():
            plt.plot(range(len(traj)), traj,
                     label=f"cumsum reward, alpha {used_alpha}")
            plt.legend()
        plt.subplot(5, 1, 3)
        for used_alpha, traj in statistics_regrets.items():
            plt.plot(range(len(traj)), traj,
                     label=f"regrets, alpha {used_alpha}")
            plt.legend()
        plt.subplot(5, 1, 4)
        for used_alpha, traj in statistics_optimalities.items():
            plt.plot(range(len(traj)), traj,
                     label=f"optimalities, alpha {used_alpha}")
            plt.legend()
        plt.subplot(5, 1, 5)
        for used_alpha, traj in statistic_optimalities_percentage.items():
            plt.plot(range(len(traj)), traj,
                     label=f"optimalities, alpha {used_alpha}")
            plt.legend()
        plt.show()

    return statistics_mean, statistics_cumsum, statistics_regrets, statistics_optimalities


if __name__ == "__main__":
    statistic_mean, statistics_cumsum, statistics_regrets, statistics_optimalities = gradientbanditnobaseline_exp(
        max_steps=MAX_STEPS, n_arms=N_ARMS, used_alpha=USED_ALPHA, num_games=NUM_GAMES, printed=False)

    save_results = {"mean_rewards": statistic_mean, "total_rewards": statistics_cumsum,
                    "regrets": statistics_regrets, "optimalities": statistics_optimalities}

    with open("results/gradient_banditnobaseline.json", "w") as file:
        json.dump(save_results, file)
