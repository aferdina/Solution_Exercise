from sheet01.environments.multiarmed_bandits import GaussianBanditEnv
from sheet03.models.multiarmedmodels import BoltzmannGumbelRightWay
from sheet02.experiments.trainmultiarmed import train_multiarmed
import numpy as np
import matplotlib.pyplot as plt

MAX_STEPS = 1000
N_ARMS = 10
NUM_GAMES = 3000


def bolzmann_exp(max_steps, n_arms, num_games, printed):

    statistics_mean = {}
    statistics_cumsum = {}
    statistics_regrets = {}
    statistics_optimalities = {}

    # IMPORTANT: boltzmanmConstant and gumbel lead to the same results
    rewards = np.zeros(shape=(num_games, max_steps))
    regrets = np.zeros(shape=(num_games, max_steps))
    optimalities = np.zeros(shape=(num_games, max_steps))
    for game in range(num_games):
        mean_parameter = np.random.uniform(
            low=0.0, high=1.0, size=n_arms).tolist()
        env = GaussianBanditEnv(
            mean_parameter=mean_parameter, max_steps=max_steps)
        # mean_parameter = np.random.normal(
        #    loc=0.0, scale=1.0, size=n_arms).tolist()
        # env = GaussianBanditEnv(
        #    mean_parameter=mean_parameter, max_steps=max_steps)
        # from paper optimal constant
        some_constant = 1.0
        agent = BoltzmannGumbelRightWay(
            some_constant=some_constant, n_arms=n_arms)
        reward, _chosen_arms, regret, optimality = train_multiarmed(
            agent=agent, env=env, num_games=1, parameter="some_constant", printed=False)
        rewards[game,] = reward
        regrets[game,] = regret
        optimalities[game,] = optimality

    mean_rewards = np.mean(rewards, axis=0)
    mean_cum_rewards = np.cumsum(mean_rewards)
    mean_regrets = np.mean(regrets, axis=0)
    mean_optimalities = np.mean(optimalities, axis=0)
    index_array = np.arange(len(mean_optimalities))
    mean_optimalities = mean_optimalities / (index_array + 1)

    statistics_mean["optimal_constant"] = mean_rewards
    statistics_cumsum["optimal_constant"] = mean_cum_rewards
    statistics_regrets["optimal_constant"] = mean_regrets
    statistics_optimalities["optimal_constant"] = mean_optimalities

    # print statistics in console
    print(50*"*")
    print(
        f"total mean reward with optimal constant is {mean_cum_rewards[-1]}")
    print(
        f"total regret with optimal constant is {mean_regrets[-1]}")
    print(
        f"total optimality with optimal constant is {mean_optimalities[-1]}")
    print(50*"*")

    if printed:
        plt.subplot(4, 1, 1)
        for some_constant, traj in statistics_mean.items():
            plt.plot(range(len(traj)), traj,
                     label=f"mean reward, some_constant {some_constant}")
            plt.legend()
        plt.subplot(4, 1, 2)
        for some_constant, traj in statistics_cumsum.items():
            plt.plot(range(len(traj)), traj,
                     label=f"cumsum reward, some_constant {some_constant}")
            plt.legend()
        plt.subplot(4, 1, 3)
        for some_constant, traj in statistics_regrets.items():
            plt.plot(range(len(traj)), traj,
                     label=f"regrets, some_constant {some_constant}")
            plt.legend()
        plt.subplot(4, 1, 4)
        for some_constant, traj in statistics_optimalities.items():
            plt.plot(range(len(traj)), traj,
                     label=f"optimalities, some_constant {some_constant}")
            plt.legend()
        plt.show()

    return statistics_mean, statistics_cumsum, statistics_regrets, statistics_optimalities


if __name__ == "__main__":
    bolzmann_exp(max_steps=MAX_STEPS, n_arms=N_ARMS,
                 num_games=NUM_GAMES, printed=True)
