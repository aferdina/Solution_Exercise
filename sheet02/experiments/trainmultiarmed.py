import numpy as np

from sheet01.experiment.trainingsutils import plot_statistics


def train_multiarmed(agent, env, num_games, parameter, printed):
    """ train/run multiarmed model on a game environment

    Args:
        agent (obj): agent, multiarmed model
        env (obj): game environment
        num_games (int): number of games to play
        parameter (str): string with name of releveant parameter
        printed (bool): bool if metrics should be printed

    Returns:
        list: list including all relevant metrics
    """

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
            # playing the game until it is done
            action = agent.select_arm()
            _new_state, reward, done, _info = env.step(action)
            rewards[game, (env.count - 1)] = reward
            chosen_arms[game, (env.count - 1)] = action
            regrets[game, (env.count - 1)] = env.regret
            optimalities[game, (env.count - 1)] = env.played_optimal

            agent.update(action, reward)

    if printed:
        plot_statistics(prin_rewards=rewards, prin_chosen_arms=chosen_arms, prin_regrets=regrets,
                        prin_optimalities=optimalities, parameter=getattr(
                agent, parameter), name=parameter)

    return rewards, chosen_arms, regrets, optimalities
