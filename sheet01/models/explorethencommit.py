class ExploreThenCommit:
    """ explore then commit algorithm
    """

    def __init__(self, explore, n_arms):
        """initialize explore then commit algorithm

        Args:
            explore (int): number of steps to explore each arm
            n_arms (int): number of arms in the multi arm bandit
        """
        self.explore = explore
        self.counts_actions = [0 for _ in range(n_arms)]
        self.values = [0.0 for _ in range(n_arms)]
        self.n_arms = n_arms

    def select_arm(self, count):
        """ select the best arm given the estimators of the values

        Args:
            count (int): step in the game

        Returns:
            int: best action based on the estimators of the values
        """
        if self.explore * self.n_arms < count:
            max_value = max(self.values)
            best_action = self.values.index(max_value)
            return best_action
        return count % self.n_arms

    def update(self, chosen_arm, reward):
        """ update the value estimators and counts based on the new observed
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
        new_value = ((times_played_chosen_arm - 1) / float(times_played_chosen_arm)
                     ) * value + (1 / float(times_played_chosen_arm)) * reward
        self.values[chosen_arm] = new_value

    def reset(self):
        """ reset agent by resetting all required statistics
        """
        self.counts_actions = [0 for _ in range(self.n_arms)]
        self.values = [0.0 for _ in range(self.n_arms)]
