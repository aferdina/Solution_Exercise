import numpy as np
from sheet01.environments.multiarmed_bandits import GaussianBanditEnv
from sheet01.models.explorethencommit import ExploreThenCommit
from sheet01.experiment.trainingsutils import plot_statistics


AGENT_EXPLORE = 2
# probability parameter of bernoulli bandit problem
MEAN_PARAMETER = [0.6, 0.7, 0.5]
N_ARMS = len(MEAN_PARAMETER)
MAX_STEPS = 10 # maximal gamestep
NUM_GAMES = 5  # games played of the greedy algorithm

agent = ExploreThenCommit(explore=AGENT_EXPLORE, n_arms=N_ARMS)
gauss_multi_arm = GaussianBanditEnv(
    mean_parameter=MEAN_PARAMETER, max_steps=MAX_STEPS)


def train_exandcommit(agent, env, num_games, parameter, printed):

    chosen_arms = np.zeros(shape=(num_games, env.max_steps))
    rewards = np.zeros(shape=(num_games, env.max_steps))
    regrets = np.zeros(shape=(num_games, env.max_steps))
    optimalities = np.zeros(shape=(num_games, env.max_steps))

    for game in range(num_games):
        # playing the algo for `num_games` rounds
        agent.reset()
        env.reset()
        done = False
        while not done:
            action = agent.select_arm(env.count)
            _new_state, reward, done, _info = env.step(action)
            rewards[game, (env.count-1)] = reward
            chosen_arms[game, (env.count-1)] = action
            regrets[game, (env.count-1)] = env.regret
            optimalities[game, (env.count-1)] = env.played_optimal
            if env.count <= agent.explore*agent.n_arms:
                # only update in exploration phase
                agent.update(action, reward)

    if printed:
       plot_statistics(prin_rewards=rewards, prin_chosen_arms=chosen_arms, prin_regrets=regrets,
                        prin_optimalities=optimalities, parameter=getattr(
                            agent, parameter), name=parameter)

    return rewards, chosen_arms, regrets, optimalities


if __name__ == "__main__":

    train_exandcommit(agent=agent, env=gauss_multi_arm,
                      num_games=NUM_GAMES, parameter="explore", printed=True)
