from abc import ABC, abstractmethod
import numpy as np

from sheet01.tests.utils import is_float_between_0_and_1, is_positive_integer, is_positive_float


class BaseModel(ABC):
    """ create a basemodel class for multiarmed bandit models
    """

    def __init__(self, n_arms):
        """ initialize epsilon greedy algorithm

        Args:
            epsilon (float): epsilon parameter for the epsilon greedy algorithm
            n_arms (int): number of possible arms
        """
        assert is_positive_integer(
            n_arms), f"{n_arms} should be a positive integer"
        self.n_arms = n_arms
        self.counts = np.zeros(self.n_arms, dtype=np.float32)
        self.values = np.zeros(self.n_arms, dtype=np.float32)

    @abstractmethod
    def select_arm(self, *args, **kwargs):
        """ select arm given a specific policy
        """
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        """ update algorithm given a specific arm and a specific reward
        """
        pass

    def reset(self):
        """ reset agent by resetting all required statistics
        """
        self.counts = np.zeros(self.n_arms, dtype=np.float32)
        self.values = np.zeros(self.n_arms, dtype=np.float32)

class GradientBandit(BaseModel):
    """ gradient bandit algorithm
    """

    def __init__(self, alpha, n_arms):
        """initialize gradient bandit with learning rate `alpha` and `n_arms`

        Args:
            alpha (float): float describing learning rate
            n_arms (int): number of used arms
        """
        super().__init__(n_arms=n_arms)
        # init tests
        assert is_positive_float(
            alpha), "Learning rate has to be a positive float"

        self.alpha = alpha
        self.count = 0
        self.mean_reward = 0.0

    def get_prob(self, action):
        """ get probability from an action
        """
        input_vector = np.exp(self.values)
        return float(input_vector[action]/np.sum(input_vector))

    def select_arm(self):
        """ choose arm in the gradient bandit algorithmus

        Returns:
            int: sampled action
        """
        input_vector = np.exp(self.values)
        input_vector = input_vector / np.sum(input_vector)
        return np.random.choice(self.n_arms, p=input_vector)

    def update(self, chosen_arm, reward):
        """ update the value estimators and counts based on the new observed
         reward and played action

        Args:
            chosen_arm (int): action which was played
            reward (float): reward of the multiarmed bandit, based on playing action `chosen_arm`
        """
        action_prob = self.get_prob(chosen_arm)
        # increment the chosen arm
        action_prob_vec = np.array([-1 * action_prob for _ in range(self.n_arms)])
        action_prob_vec[chosen_arm] = 1 - action_prob
        # update via memory trick
        gradients = (self.alpha * (reward -self.mean_reward)) * action_prob_vec

        # update values
        self.values = self.values + gradients
        self.count += 1
        # update mean reward
        self.mean_reward = ((self.count - 1) / float(self.count)
                            ) * self.mean_reward + (1 / float(self.count)) * reward

    def reset(self):
        """ reset agent by resetting all required statistics
        """
        self.count = 0
        self.values = np.zeros(self.n_arms, dtype=np.float32)
        self.mean_reward = 0.0


class GradientBanditnobaseline(GradientBandit):
    """ gradient bandit algorithm
    """

    def __init__(self, alpha, n_arms):
        """initialize gradient bandit with learning rate `alpha` and `n_arms`

        Args:
            alpha (float): float describing learning rate
            n_arms (int): number of used arms
        """
        super().__init__(alpha=alpha, n_arms=n_arms)

    def update(self, chosen_arm, reward):
        """ update the value estimators and counts based on the new observed
         reward and played action

        Args:
            chosen_arm (int): action which was played
            reward (float): reward of the multiarmed bandit, based on playing action `chosen_arm`
        """
        action_prob = self.get_prob(chosen_arm)
        # increment the chosen arm
        action_prob_vec = np.array([-1 * action_prob for _ in range(self.n_arms)])
        action_prob_vec[chosen_arm] = 1 - action_prob
        # update via memory trick
        gradients = (self.alpha * (reward)) * action_prob_vec

        # update values
        self.values = self.values + gradients

class BoltzmannConstant(BaseModel):
    """ boltzmann exploration algorithm also known as softmax bandit
    """

    def __init__(self, temperature, n_arms):
        """initialize boltzmann algorithm with constant temperature

        Args:
            temperature (float): float describing learning rate
            n_arms (int): number of used arms
        """
        super().__init__(n_arms=n_arms)
        # init tests
        assert is_positive_float(
            temperature), "The temperature  has to be a positive float"
        self.temperature = temperature

    def select_arm(self):
        """choose an arm from the boltzmann distribution

        Returns:
            int: simulated action
        """
        canonical_parameter = self.temperature * self.values
        input_vector = np.exp(canonical_parameter)
        probs = (input_vector / np.sum(input_vector)).tolist()
        x = np.random.rand()
        cum = 0
        for i,p in enumerate(probs):
            cum += p
            if x < cum:
                break
        return i
        #return np.random.choice(self.n_arms, p=input_vector)

    def update(self, chosen_arm, reward):
        """ update the value estimators and counts based on the new observed
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
        new_value = ((times_played_chosen_arm - 1) / times_played_chosen_arm) * \
            value + (1 / times_played_chosen_arm) * reward
        self.values[chosen_arm] = new_value


class BoltzmannGumbel(BoltzmannConstant):
    """ boltzmann exploration algorithm also known as softmax bandit
    """

    def __init__(self, temperature, n_arms):
        """initialize boltzmann algorithm with constant temperature

        Args:
            temperature (float): float describing learning rate
            n_arms (int): number of used arms
        """
        super().__init__(temperature=temperature, n_arms=n_arms)

    def select_arm(self):
        """ select action with respect to gumbel trick

        Returns:
            int: returned action
        """
        _parameter = self.temperature * self.values
        gumbel_rvs = np.random.gumbel(loc=0,scale=1,size=self.n_arms)
        return np.argmax(_parameter + gumbel_rvs)


class BoltzmannGumbelRightWay(BaseModel):
    """ boltzmann exploration algorithm also known as softmax bandit
    """

    def __init__(self, some_constant, n_arms):
        """initialize boltzmann algorithm with constant temperature

        Args:
            temperature (float): float describing learning rate
            n_arms (int): number of used arms
        """
        super().__init__(n_arms=n_arms)
        # init tests
        assert is_positive_float(
            some_constant), "The some_constant  has to be a positive float"

        self.some_constant = some_constant

    def select_arm(self):
        """ get action from boltzmann gumbel paper
        """

        gumbel_rvs = np.random.gumbel(loc=0.0,scale=1.0,size=self.n_arms)
        betas = self.some_constant * np.sqrt(1/self.counts)
        used_parameter = self.values + betas*gumbel_rvs
        return np.argmax(used_parameter)

    def update(self, chosen_arm, reward):
        """ update the value estimators and counts based on the new observed
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
        new_value = ((times_played_chosen_arm - 1) / times_played_chosen_arm) * \
            value + (1 / times_played_chosen_arm) * reward
        self.values[chosen_arm] = new_value
