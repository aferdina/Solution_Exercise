from abc import ABC, abstractmethod
from typing import List
from sheet01.tests.utils import is_positive_integer


class BaseModel(ABC):
    """create a basemodel class for multiarmed bandit models"""

    def __init__(self, n_arms: int):
        """initialize epsilon greedy algorithm

        Args:
            epsilon (float): epsilon parameter for the epsilon greedy algorithm
            n_arms (int): number of possible arms
        """
        assert is_positive_integer(n_arms), f"{n_arms} should be a positive integer"
        self.n_arms = n_arms
        self.counts: List[int] = [0 for _ in range(self.n_arms)]
        self.values: List[float] = [0.0 for _ in range(self.n_arms)]

    @abstractmethod
    def select_arm(self, *args) -> int:
        """select arm given a specific policy"""

    @abstractmethod
    def update(self, *args) -> None:
        """update algorithm given a specific arm and a specific reward"""

    def reset(self) -> None:
        """reset agent by resetting all required statistics"""
        self.counts = [0 for _ in range(self.n_arms)]
        self.values = [0.0 for _ in range(self.n_arms)]


class ExploreThenCommit(BaseModel):
    """explore then commit algorithm"""

    def __init__(self, explore: int, n_arms: int) -> None:
        """initialize explore then commit algorithm

        Args:
            explore (int): number of steps to explore each arm
            n_arms (int): number of arms in the multi arm bandit
        """
        super().__init__(n_arms=n_arms)
        self.explore = explore
        self.counts_actions = [0 for _ in range(n_arms)]

    def select_arm(self, count: int) -> None:
        """select the best arm given the estimators of the values

        Args:
            count (int): step in the game

        Returns:
            int: best action based on the estimators of the values
        """
        if self.explore * self.n_arms < count:
            max_value = max(self.values)
            best_action = (self.values).index(max_value)
            return best_action
        return count % self.n_arms

    def update(self, chosen_arm: int, reward: float):
        """update the value estimators and counts based on the new observed
         reward and played action

        Args:
            chosen_arm (int): action which was played
            reward (float): reward of the multiarmed bandit, based on playing action `chosen_arm`
        """
        # increment the chosen arm
        self.counts_actions[chosen_arm] = self.counts_actions[chosen_arm] + 1
        times_played_chosen_arm = self.counts_actions[chosen_arm]
        value = self.values[chosen_arm]
        # update via memory trick
        new_value = (
            (times_played_chosen_arm - 1) / float(times_played_chosen_arm)
        ) * value + (1 / float(times_played_chosen_arm)) * reward
        self.values[chosen_arm] = new_value

