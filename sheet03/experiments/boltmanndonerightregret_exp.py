from sheet01.environments.multiarmed_bandits import GaussianBanditEnv
from sheet03.models.multiarmedmodels import BoltzmannGumbelRightWay
from sheet02.experiments.trainmultiarmed import train_multiarmed
import numpy as np
import matplotlib.pyplot as plt
import json

MAX_STEPS = 1000
N_ARMS = 10
USED_SOMECONSTANT = np.arange(start=0.05, stop=10.0, step=0.05).tolist()
NUM_GAMES = 3000


def bolzmann_exp(max_steps, n_arms, used_someconstant, num_games, printed):

    statistics_cumsum = []
    statistics_regrets = []
    statistics_optimalities = []

    for some_constant in used_someconstant:
        # IMPORTANT: boltzmanmConstant and gumbel lead to the same results
        agent = BoltzmannGumbelRightWay(
            some_constant=some_constant, n_arms=n_arms)
        rewards = np.zeros(shape=(num_games, max_steps))
        regrets = np.zeros(shape=(num_games, max_steps))
        optimalities = np.zeros(shape=(num_games, max_steps))
        for game in range(num_games):
            mean_parameter = np.random.normal(
                loc=0.0, scale=1.0, size=n_arms).tolist()
            env = GaussianBanditEnv(
                mean_parameter=mean_parameter, max_steps=max_steps)
            agent.reset()
            reward, _chosen_arms, regret, optimality = train_multiarmed(
                agent=agent, env=env, num_games=1,parameter="some_constant", printed=False)
            rewards[game,] = reward
            regrets[game,] = regret
            optimalities[game,] = optimality

        mean_rewards = np.mean(rewards, axis=0)
        mean_cum_rewards = np.cumsum(mean_rewards)
        mean_regrets = np.mean(regrets, axis=0)
        mean_optimalities = np.mean(optimalities, axis=0)
        index_array = np.arange(len(mean_optimalities))
        mean_optimalities = mean_optimalities / (index_array + 1)

        statistics_cumsum.append(mean_cum_rewards[-1])
        statistics_regrets.append(mean_regrets[-1])
        statistics_optimalities.append(mean_optimalities[-1])
        # print statistics in console
        print(50*"*")
        print(
            f"total mean reward with some_constant= {some_constant} is {mean_cum_rewards[-1]}")
        print(
            f"total regret with some_constant= {some_constant} is {mean_regrets[-1]}")
        print(
            f"total optimality with some_constant= {some_constant} is {mean_optimalities[-1]}")
        print(50*"*")

    if printed:

        # plot total rewards
        plt.subplot(3, 1, 1)
        plt.plot(used_someconstant, statistics_cumsum,
                 label="mean reward for boltzman gumbel right way")
        plt.legend()

        # plot regrets
        plt.subplot(3, 1, 2)
        plt.plot(used_someconstant, statistics_regrets,
                 label="regrets for boltzman gumbel right way")
        plt.legend()

        # plot total optimalities
        plt.subplot(3, 1, 3)
        plt.plot(used_someconstant, statistics_optimalities,
                 label="total optimalities for boltzman gumbel right way")
        plt.legend()

        plt.show()

    return statistics_cumsum, statistics_regrets, statistics_optimalities


if __name__ == "__main__":
    statistics_cumsum, statistics_regrets, statistics_optimalities = bolzmann_exp(max_steps=MAX_STEPS, n_arms=N_ARMS,
                                                                                  used_someconstant=USED_SOMECONSTANT, num_games=NUM_GAMES, printed=True)

    save_results = {"total_rewards": statistics_cumsum,
                    "regrets": statistics_regrets, "optimalities": statistics_optimalities}

    with open("results/gumbel_experiment.json", "w") as file:
        json.dump(save_results, file)