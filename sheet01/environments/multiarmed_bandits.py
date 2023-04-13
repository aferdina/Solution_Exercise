""" Include all game environments for multi armed bandits
"""
import numpy as np

from sheet01.tests.utils import is_list_of_floats, is_positive_integer, check_floats_between_zero_and_one


class BaseBanditEnv:
    """ class for a basic multiarmed bandit model
    """

    def __init__(self, mean_parameter, max_steps):
        """create a multiarm bandit with `len(p_parameter)` arms

        Args:
            mean_parameter (list): list containing mean parameter of guassian bandit arms 
            max_steps (int): number of total steps for the bandit problem
        """
        assert is_list_of_floats(
            mean_parameter), "The means of a multiarmed bandit model should be a list of floats."
        assert isinstance(
            max_steps, int), "The number of maximal steps should be an int."
        assert is_positive_integer(max_steps), "The number of steps should be a positive integer"
        self.n_arms = len(mean_parameter)
        self.max_steps = max_steps
        self.count = 0

        # to save regret statistics
        self.optimal = [max(mean_parameter),
                        mean_parameter.index(max(mean_parameter))]
        self.played_optimal = 0
        self.regret = 0.0

    def step(self, action):
        pass

    def reset(self):
        """ reset all statistics to run a new game
        """
        self.count = 0
        self.played_optimal = 0
        self.regret = 0.0


class GaussianBanditEnv(BaseBanditEnv):
    """ class for creating gaussian bandit
    """

    def __init__(self, mean_parameter, max_steps, var_scale=1.0):
        """create a multiarm bandit with `len(p_parameter)` arms

        Args:
            mean_parameter (list): list containing mean parameter of guassian bandit arms
            max_steps (int): number of total steps for the bandit problem
        """
        super().__init__(mean_parameter=mean_parameter, max_steps=max_steps)

        self.p_parameter = mean_parameter
        self.var_scale = var_scale

    def step(self, action):
        """ play an action in the gaussian bandit modell

        Args:
            action (int): choosen arm

        Returns:
            list: new state, reward, done (bool if game is finished), info
        """
        assert action in range(
            self.n_arms), f"the action {action} is not valid"
        reward = np.random.normal(
            loc=self.p_parameter[action], scale=self.var_scale, size=None)
        self.count += 1

        # check if best action was played
        if action == self.optimal[1]:
            self.played_optimal += 1

        # update the regret in the game
        self.regret += (self.optimal[0] - reward)

        # if game is finished `done=True`
        done = bool(self.count >= self.max_steps)

        return 0, reward, done, {}


class BernoulliBanditEnv(BaseBanditEnv):
    """ Bernoulli game environment from lecture
    """

    def __init__(self, mean_parameter, max_steps):
        super().__init__(mean_parameter=mean_parameter, max_steps=max_steps)
        assert check_floats_between_zero_and_one(
            mean_parameter), f"mean parameter has to be a list of floats between zero and one"
        self.p_parameter = mean_parameter

    def step(self, action):
        """play a step of the multiarmed bandit, given an action

        Args:
            action (int): chosen arm 

        Returns:
            list: next state, reward, info if done, game info
        """
        # check, if the action is valid
        assert action in range(
            self.n_arms), f"the action {action} is not valid"

        # sample the reward, depending on the chosen arm
        if np.random.uniform() < self.p_parameter[action]:
            reward = 1
        else:
            reward = 0

        # set counter +1
        self.count += 1

        # check if best action was played
        if action == self.optimal[1]:
            self.played_optimal += 1

        # update the regret in the game
        self.regret += (self.optimal[0] - reward)

        # if game is finished `done=True`
        done = bool(self.count >= self.max_steps)

        return 0, reward, done, {}


if __name__ == "__main__":
    # play Bernoulli Bandit
    print(50 * "-")
    print("play bernoulli bandit")
    arm_probabilities_bernoulli = [0.1, 0.9, 0.1, 0.1]
    VARMAXSTEPS = len(arm_probabilities_bernoulli)
    bandit_env = BernoulliBanditEnv(
        mean_parameter=arm_probabilities_bernoulli, max_steps=VARMAXSTEPS)
    for play_action in range(len(arm_probabilities_bernoulli)):
        _, get_reward, _, _ = bandit_env.step(play_action)
        print("Arm", play_action, "gave a reward of:", get_reward)
        print(f"optimal action was {bandit_env.played_optimal} times played")
        print(f"the new regret is {bandit_env.regret}")

    print(50 * "-")
    print("play gaussian bandit")
    # play Gaussian Bandit
    arm_means_gaussian = [1.0, 2.0, 5.0, 2.0]
    VARMAXSTEPS = len(arm_means_gaussian)
    bandit_env = GaussianBanditEnv(
        mean_parameter=arm_means_gaussian, max_steps=VARMAXSTEPS)
    for play_action in range(len(arm_means_gaussian)):
        _, get_reward, _, _ = bandit_env.step(play_action)
        print("Arm", play_action, "gave a reward of:", get_reward)
        print(f"optimal action was {bandit_env.played_optimal} times played")
        print(f"the new regret is {bandit_env.regret}")
