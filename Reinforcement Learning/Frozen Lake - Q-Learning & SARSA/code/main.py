# -*- coding: utf-8 -*-

from World import World
import numpy as np
import datetime as dt


def hyperparameter_tuning_Sarsa():
    # Taking start time timestamp
    start = dt.datetime.now()

    # Execute Sarsa algorithm and extracting stats info                     Episodes,alpha,gamma,eps,Decay
    QMatrixSarsa, episodes, timesteps, rewards, L1_Norms = env.optimize_Sarsa(56176, 0.007, 0.9, 0.9, 0.001)

    # Calculating Execution duration in minutes
    duration = (dt.datetime.now() - start).total_seconds() / 60

    # Best state action values derived in the 1st MDP assignment, denoted as V-mdp
    V_opt = np.array([0., 0.28592528, 0.07677605, 0.00829162, 0.74737133, 0.57641035, 0., -0.08591965, 0.92809858,
                      0.58410791, 0.1885577, 0.08029601, 0., 0., 0., -0.08591964])

    # Best state action values derived in the Q-Learning algorithm, denoted as V-Sarsa
    v_max = np.array(QMatrixSarsa.max(axis=1))

    print(
        "Episodes: {0}, "
        "Averaged Reward: {1}, "
        "L1-Norm: {2}, "
        "Execution Time: {3} Minutes".format(str(len(episodes)), str(sum(rewards) / (len(episodes))),
                                             np.linalg.norm(v_max - V_opt, ord=1), duration))
    # Plotting state action values based on the Q-function
    env.plot_actionValues(QMatrixSarsa)
    # Plotting state values based on the Q-function
    env.plot_state_values(QMatrixSarsa)
    # Plotting rewards per episode with a smoothed averaged reward with a window size of 100
    env.plot_episodes_rewards(episodes, rewards, 0.007, 0.9, 0.999, 100)
    # Plotting time steps per episode
    env.plot_timesteps_episodes(episodes, timesteps, 0.007, 0.9, 0.999)
    # Plotting L1-Norm per episode
    env.plot_L1_episodes(episodes, L1_Norms, 0.007, 0.9, 0.999)


def hyperparameter_tuning_QL():

    # Taking start time timestamp
    start = dt.datetime.now()

    # Execute Q-Learning algorithm and extracting stats info                 Episodes,alpha,gamma,eps,Decay
    QMatrixQL, episodes, timesteps, rewards, L1_Norms = env.optimize_QLearning(66058, 0.001, 0.9, 1, 0.001)

    # Calculating Execution duration in minutes
    duration = (dt.datetime.now() - start).total_seconds() / 60

    print(
        "Episodes: {0}, "
        "Averaged Reward: {1}, "
        "L1-Norm: {2}, "
        "Execution Time: {3} Minutes".format(str(len(episodes)), str(sum(rewards) / (len(episodes))),
                                             L1_Norms[-1], duration))

    # Plotting state action values based on the Q-function
    env.plot_actionValues(QMatrixQL)
    # Plotting state values based on the Q-function
    env.plot_state_values(QMatrixQL)
    # Plotting rewards per episode with a smoothed averaged reward with a window size of 100
    env.plot_episodes_rewards(episodes, rewards, 0.001, 1, 0.001, 100)
    # Plotting time steps per episode
    env.plot_timesteps_episodes(episodes, timesteps, 0.001, 1, 0.001)
    # Plotting L1-Norm per episode
    env.plot_L1_episodes(episodes, L1_Norms, 0.001, 0.9, 0.001)

if __name__ == "__main__":

    # Environment instantiation
    env = World()

    # Initialize environment
    env.reset()

    # Executing Sarsa algorithm with with the best found hyper-parameters
    QMatrix_SARSA = env.SARSA(56176, 0.007, 0.9, 0.9, 0.001)
    env.plot_actionValues(QMatrix_SARSA)

    # Executing Q-Learning algorithm with the best found hyper-parameters
    QMatrix_QL = env.QLearning(66058, 0.001, 0.9, 1, 0.001)
    env.plot_actionValues(QMatrix_QL)

