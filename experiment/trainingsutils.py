import matplotlib.pyplot as plt



def plot_statistics(prin_rewards, prin_chosen_arms, prin_regrets, prin_optimalities, parameter, name):

    plt.subplot(4, 1, 1)
    for i, reward in enumerate(prin_rewards):
        plt.plot(range(len(reward)), reward,
                    label=f"reward {i}, {name} {parameter}")
        plt.legend()

    plt.subplot(4, 1, 2)
    for i, action in enumerate(prin_chosen_arms):
        plt.plot(range(len(action)), action,
                    label=f"action sequence {i}, {name} {parameter}")
        plt.legend()

    plt.subplot(4, 1, 3)
    for i, action in enumerate(prin_regrets):
        plt.plot(range(len(action)), action,
                    label=f"regrets {i}, {name} {parameter}")
        plt.legend()

    plt.subplot(4, 1, 4)
    for i, action in enumerate(prin_optimalities):
        plt.plot(range(len(action)), action,
                    label=f"optimalities {i}, {name} {parameter}")
        plt.legend()

    plt.show()

import numpy as np

def bound_function(explore, deltas, number_of_games):
    """ bound function for sub gaussian bandits

    Args:
        explore (_type_): _description_
        deltas (_type_): _description_
        number_of_games (_type_): _description_

    Returns:
        _type_: _description_
    """
    numb_arms = deltas.shape[0]
    assert (1 <= explore) and (explore <= number_of_games /
                               numb_arms), f"explore must be greater than 1 and less then {number_of_games/numb_arms}"
    output = explore * np.sum(deltas) + (number_of_games-explore*numb_arms) * \
        np.sum(deltas*np.exp(-explore/4*deltas**2))
    return output

