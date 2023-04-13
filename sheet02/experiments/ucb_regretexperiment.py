import matplotlib.pyplot as plt
import numpy as np

from sheet01.environments.multiarmed_bandits import GaussianBanditEnv
from sheet02.experiments.trainmultiarmed import train_multiarmed
from sheet02.models.mutliarmedmodels import UCB

MAX_STEPS = 1000
USED_MEAN_PARAMETERS = [0.1, 0.4, 0.3]
NUM_GAMES = 50


def calc_ucb_regret_bound(max_steps, used_mean_parameters):
    """ Calculate the regret bound for the ucb algorithm from lecture

    Args:
        max_steps (int): number of maximal steps in the game
        used_mean_parameters (list): list including all mean parameters of arms
    """
    used_parameters = np.asarray(used_mean_parameters)
    optim_gaps = np.max(used_parameters) - used_parameters
    optim_gaps_withoutzero = optim_gaps[np.nonzero(optim_gaps)[0]]
    return 3 * np.sum(optim_gaps_withoutzero) + np.int32(16) * np.log(max_steps) * np.sum(1 / optim_gaps_withoutzero)


def ucb_regret_exp(max_steps, used_mean_parameters, num_games, printed):
    n_arms = len(used_mean_parameters)
    regret_list = []
    for game_length in range(1, max_steps + 1):
        print(f"game length is {game_length}")
        # delta from theorem
        delta = 1 / (float(game_length + 1) ** 2)

        # initialize agent
        agent = UCB(delta=delta, n_arms=n_arms)
        env = GaussianBanditEnv(
            mean_parameter=used_mean_parameters, max_steps=game_length + 1)

        _rewards, _chosen_arms, regrets, _optimalities = train_multiarmed(
            agent=agent, env=env, num_games=num_games, parameter="delta", printed=False)

        mean_regrets = np.mean(regrets, axis=0)
        regret_list.append(float(mean_regrets[-1]))

    if printed:
        # for _no_traj,traj in enumerate(regrets):
        #     plt.plot(range(max_steps),traj)
        plt.plot(range(max_steps), regret_list,
                 label=f"mean regret, delta {delta}")
        plt.plot(range(max_steps), calc_ucb_regret_bound(range(max_steps),
                                                         used_mean_parameters=used_mean_parameters), label="regret bound")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    ucb_regret_exp(max_steps=MAX_STEPS,
                   used_mean_parameters=USED_MEAN_PARAMETERS, num_games=NUM_GAMES, printed=True)
