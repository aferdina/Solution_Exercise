from sheet01.tests.utils import is_float_between_0_and_1, is_positive_integer
from sheet01.models.explorethencommit import BaseModel
import random
import math


class EpsilonGreedy(BaseModel):
    """class for epsilon greedy algorithm"""

    def __init__(self, epsilon, n_arms):
        """initialize epsilon greedy algorithm

        Args:
            epsilon (float): epsilon parameter for the epsilon greedy algorithm
            n_arms (int): number of possible arms
        """
        super().__init__(n_arms=n_arms)
        assert is_float_between_0_and_1(
            epsilon
        ), f"{epsilon} should be a float between 0 and 1"
        self.epsilon = epsilon

    def select_arm(self):
        """select the best arm given the estimators of the values

        Returns:
            int: best action based on the estimators of the values
        """
        if random.random() > self.epsilon:
            max_value = max(self.values)
            best_action = (self.values).index(max_value)
            return best_action
        return random.randrange(self.n_arms)

    def update(self, chosen_arm, reward):
        """update the value estimators and counts based on the new observed
          reward and played action

        Args:
            chosen_arm (int): action which was played
            reward (float): reward of the multiarmed bandit, based on playing action `chosen_arm`
        """
        # increment the chosen arm
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        times_played_chosen_arm = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        # update via memory trick
        new_value = (
            (times_played_chosen_arm - 1) / float(times_played_chosen_arm)
        ) * value + (1 / float(times_played_chosen_arm)) * reward
        self.values[chosen_arm] = new_value


class UCB(BaseModel):
    """class for ucb algorithm"""

    def __init__(self, delta, n_arms, prefactor=2):
        """initialize upper confidence bound algorithm

        Args:
            n_arms (int): number of arms in the multiarmed bandit model
            delta (float): delta parameter of ucb algorithm
        """
        super().__init__(n_arms=n_arms)
        assert is_float_between_0_and_1(
            delta
        ), f"{delta} should be a float between 0 and 1"
        self.delta = delta
        self.ucb_values = [float("inf") for _ in range(self.n_arms)]
        self.prefactor = prefactor

    def select_arm(self):
        """select the best arm given the value estimators and the ucb bound
        Returns:
            int: best action based on upper confidence bound
        """
        max_value = max(self.ucb_values)
        return (self.ucb_values).index(max_value)

    def update(self, chosen_arm, reward):
        """update the ucb bound of the ucb algorithm

        Args:
            chosen_arm (int): action which was played an should be updated
            reward (float): reward of the multiarmed bandit, based on playing action `chosen_arm`
        """
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        times_played_chosen_arm = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        new_value = (
            (times_played_chosen_arm - 1) / float(times_played_chosen_arm)
        ) * value + (1 / float(times_played_chosen_arm)) * reward
        self.values[chosen_arm] = new_value
        # update all arms which are played at least one time
        # # pylint: disable=C0301
        for arm in [
            arm_index
            for arm_index, already_played in enumerate(self.counts)
            if already_played != 0
        ]:
            bonus = math.sqrt(
                (self.prefactor * math.log(1 / self.delta)) / float(self.counts[arm])
            )
            self.ucb_values[arm] = self.values[arm] + bonus

    def reset(self):
        """reset agent by resetting all required statistics"""
        self.counts = [0 for _ in range(self.n_arms)]
        self.values = [0.0 for _ in range(self.n_arms)]
        self.ucb_values = [float("inf") for _ in range(self.n_arms)]
