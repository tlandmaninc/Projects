import numpy as np
import matplotlib.pyplot as plt
from numpy.random import choice
import pandas as pd
import datetime as dt


class World:

    def __init__(self):

        self.nRows = 4
        self.nCols = 4
        self.stateHoles = [1, 7, 14, 15]
        self.stateGoal = [13]
        self.nStates = 16
        self.States = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        self.nActions = 4
        self.rewards = np.array([-1] + [-0.04] * 5 + [-1] + [-0.04] * 5 + [1, -1, -1] + [-0.04])
        self.stateInitial = [4]
        self.observation = []

    def _plot_world(self):

        nRows = self.nRows
        nCols = self.nCols
        stateHoles = self.stateHoles
        stateGoal = self.stateGoal
        coord = [[0, 0], [nCols, 0], [nCols, nRows], [0, nRows], [0, 0]]
        xs, ys = zip(*coord)
        plt.plot(xs, ys, "black")
        for i in stateHoles:
            (I, J) = np.unravel_index(i, shape=(nRows, nCols), order='F')
            coord = [[J, nRows - I],
                     [J + 1, nRows - I],
                     [J + 1, nRows - I + 1],
                     [J, nRows - I + 1],
                     [J, nRows - I]]
            xs, ys = zip(*coord)
            plt.fill(xs, ys, "0.5")
            plt.plot(xs, ys, "black")
        for ind, i in enumerate([stateGoal]):
            (I, J) = np.unravel_index(i, shape=(nRows, nCols), order='F')
            coord = [[J, nRows - I],
                     [J + 1, nRows - I],
                     [J + 1, nRows - I + 1],
                     [J, nRows - I + 1],
                     [J, nRows - I]]
            xs, ys = zip(*coord)
            plt.fill(xs, ys, "0.8")
            plt.plot(xs, ys, "black")
        plt.plot(xs, ys, "black")
        X, Y = np.meshgrid(range(nCols + 1), range(nRows + 1))
        plt.plot(X, Y, 'k-')
        plt.plot(X.transpose(), Y.transpose(), 'k-')

    @staticmethod
    def _truncate(n, decimals=0):
        multiplier = 10 ** decimals
        return int(n * multiplier) / multiplier

    def plot(self):
        """
        plot function
        :return: None
        """
        nStates = self.nStates
        nRows = self.nRows
        nCols = self.nCols

        self._plot_world()
        states = range(1, nStates + 1)
        k = 0
        for i in range(nCols):
            for j in range(nRows, 0, -1):
                plt.text(i + 0.5, j - 0.5, str(states[k]), fontsize=26, horizontalalignment='center',
                         verticalalignment='center')
                k += 1
        plt.title('MDP gridworld', size=16)
        plt.axis("equal")
        plt.axis("off")
        # plt.show(block=False)
        plt.show()

    def plot_value(self, valueFunction):

        nRows = self.nRows
        nCols = self.nCols
        stateHoles = self.stateHoles
        stateGoal = self.stateGoal

        fig = plt.plot(1)
        self._plot_world()
        k = 0
        for i in range(nCols):
            for j in range(nRows, 0, -1):
                if k + 1 not in stateHoles + stateGoal:
                    plt.text(i + 0.5, j - 0.5, str(self._truncate(valueFunction[k], 3)), fontsize=12,
                             horizontalalignment='center', verticalalignment='center')
                k += 1
        plt.title('MDP gridworld', size=16)
        plt.axis("equal")
        plt.axis("off")
        plt.show()

    def plot_policy(self, policy):

        nStates = self.nStates
        nActions = self.nActions
        nRows = self.nRows
        nCols = self.nCols
        stateHoles = self.stateHoles
        stateGoal = self.stateGoal
        policy = policy.reshape(nRows, nCols, order="F").reshape(-1, 1)
        X, Y = np.meshgrid(range(nCols + 1), range(nRows + 1))
        X1 = X[:-1, :-1]
        Y1 = Y[:-1, :-1]
        X2 = X1.reshape(-1, 1) + 0.5
        Y2 = np.flip(Y1.reshape(-1, 1)) + 0.5
        X2 = np.kron(np.ones((1, nActions)), X2)
        Y2 = np.kron(np.ones((1, nActions)), Y2)
        mat = np.cumsum(np.ones((nStates, nActions)), axis=1).astype("int64")
        if policy.shape[1] == 1:
            policy = (np.kron(np.ones((1, nActions)), policy) == mat)
        index_no_policy = stateHoles + stateGoal
        index_policy = [item - 1 for item in range(1, nStates + 1) if item not in index_no_policy]
        mask = policy.astype("int64") * mat
        mask = mask.reshape(nRows, nCols, nCols)
        X3 = X2.reshape(nRows, nCols, nActions)
        Y3 = Y2.reshape(nRows, nCols, nActions)
        alpha = np.pi - np.pi / 2 * mask
        self._plot_world()
        for ii in index_policy:
            ax = plt.gca()
            j = int(ii / nRows)
            i = (ii + 1 - j * nRows) % nCols - 1
            index = np.where(mask[i, j] > 0)[0]
            h = ax.quiver(X3[i, j, index], Y3[i, j, index], np.cos(alpha[i, j, index]), np.sin(alpha[i, j, index]), 0.3)
        states = range(1, nStates + 1)
        k = 0
        for i in range(nCols):
            for j in range(nRows, 0, -1):
                plt.text(i + 0.25, j - 0.25, str(states[k]), fontsize=16, horizontalalignment='right',
                         verticalalignment='bottom')
                k += 1
        plt.axis("equal")
        plt.axis("off")
        plt.show()

    def get_nrows(self):

        return self.nRows

    def get_ncols(self):

        return self.nCols

    def get_stateHoles(self):

        return self.stateHoles

    def get_stateGoal(self):

        return self.stateGoal

    def get_nstates(self):

        return self.nStates

    def get_nactions(self):

        return self.nActions

    def get_transition_model(self, p=0.8):
        nstates = self.nStates
        nrows = self.nRows
        holes_index = self.stateHoles
        goal_index = self.stateGoal
        terminal_index = holes_index + goal_index
        # actions = ["1", "2", "3", "4"]
        actions = [1, 2, 3, 4]  # I changed str to int
        transition_models = {}
        for action in actions:
            transition_model = np.zeros((nstates, nstates))
            for i in range(1, nstates + 1):
                if i not in terminal_index:
                    if action == 1:
                        if i + nrows <= nstates:
                            transition_model[i - 1][i + nrows - 1] += (1 - p) / 2
                        else:
                            transition_model[i - 1][i - 1] += (1 - p) / 2
                        if 0 < i - nrows <= nstates:
                            transition_model[i - 1][i - nrows - 1] += (1 - p) / 2
                        else:
                            transition_model[i - 1][i - 1] += (1 - p) / 2
                        if (i - 1) % nrows > 0:
                            transition_model[i - 1][i - 1 - 1] += p
                        else:
                            transition_model[i - 1][i - 1] += p
                    if action == 2:
                        if i + nrows <= nstates:
                            transition_model[i - 1][i + nrows - 1] += p
                        else:
                            transition_model[i - 1][i - 1] += p
                        if 0 < i % nrows and (i + 1) <= nstates:
                            transition_model[i - 1][i + 1 - 1] += (1 - p) / 2
                        else:
                            transition_model[i - 1][i - 1] += (1 - p) / 2
                        if (i - 1) % nrows > 0:
                            transition_model[i - 1][i - 1 - 1] += (1 - p) / 2
                        else:
                            transition_model[i - 1][i - 1] += (1 - p) / 2
                    if action == 3:
                        if i + nrows <= nstates:
                            transition_model[i - 1][i + nrows - 1] += (1 - p) / 2
                        else:
                            transition_model[i - 1][i - 1] += (1 - p) / 2
                        if 0 < i - nrows <= nstates:
                            transition_model[i - 1][i - nrows - 1] += (1 - p) / 2
                        else:
                            transition_model[i - 1][i - 1] += (1 - p) / 2
                        if 0 < i % nrows and (i + 1):
                            transition_model[i - 1][i + 1 - 1] += p
                        else:
                            transition_model[i - 1][i - 1] += p
                    if action == 4:
                        if 0 < i - nrows <= nstates:
                            transition_model[i - 1][i - nrows - 1] += p
                        else:
                            transition_model[i - 1][i - 1] += p
                        if 0 < i % nrows and (i + 1) <= nstates:
                            transition_model[i - 1][i + 1 - 1] += (1 - p) / 2
                        else:
                            transition_model[i - 1][i - 1] += (1 - p) / 2
                        if (i - 1) % nrows > 0:
                            transition_model[i - 1][i - 1 - 1] += (1 - p) / 2
                        else:
                            transition_model[i - 1][i - 1] += (1 - p) / 2
                elif i in terminal_index:
                    transition_model[i - 1][i - 1] = 1

            transition_models[action] = pd.DataFrame(transition_model, index=range(1, nstates + 1),
                                                     columns=range(1, nstates + 1))
        return transition_models

    def step(self, action):
        observation = self.observation
        state = observation[0]
        prob = {}
        done = False
        transition_models = self.get_transition_model(0.8)
        # print('inside')
        # print(state)
        # print(action)
        # print("ACTION: {0}".format(action))
        # print("STATE: {0}".format(state))
        prob = transition_models[action].loc[state, :]
        # print(transition_models[action].loc[state, :])
        s = choice(self.States, 1, p=prob)
        next_state = s[0]
        reward = self.rewards[next_state - 1]

        if next_state in self.stateGoal + self.stateHoles:
            done = True
        self.observation = [next_state]
        return next_state, reward, done

    def reset(self, *args):
        # def reset(self):
        if not args:
            observation = self.stateInitial
        else:
            observation = []
            while not (observation):
                observation = np.setdiff1d(choice(self.States), self.stateHoles + self.stateGoal)
        self.observation = observation
        return observation

    def render(self):

        nStates = self.nStates
        nActions = self.nActions
        nRows = self.nRows
        nCols = self.nCols
        stateHoles = self.stateHoles
        stateGoal = self.stateGoal

        observation = self.observation  # observation
        state = observation[0]

        # state = 3

        J = nRows - (state - 1) % nRows - 1
        I = int((state - 1) / nCols)

        circle = plt.Circle((I + 0.5, J + 0.5), 0.28, color='black')
        fig = plt.gcf()
        ax = fig.gca()
        ax.add_artist(circle)

        self.plot()

        # plt.ion()
        # plt.show()
        # plt.draw()
        # plt.pause(0.5)
        # plt.ion()
        # plt.show(block=False)
        # time.sleep(1)
        # nRows = self.nRows
        # nCols = self.nCols
        # stateHoles = self.stateHoles
        # stateGoal = self.stateGoal

        # print(state)

        # circle = plt.Circle((0.5, 0.5), 0.1, color='black')
        # fig, ax = plt.subplots()
        # ax.add_artist(circle)

        # k = 0
        # for i in range(nCols):
        #     for j in range(nRows, 0, -1):
        #         if k + 1 not in stateHoles + stateGoal:
        #             plt.text(i + 0.5, j - 0.5, str(self._truncate(valueFunction[k], 3)), fontsize=12,
        #                      horizontalalignment='center', verticalalignment='center')
        #         k += 1

    def close(self):
        plt.pause(0.5)
        plt.close()

    def show(self):
        plt.ion()
        plt.show()

    def getEpsilonGreedyAction(self, state, Q, epsilon):
        """

        :param state: Agent's current State S
        :param Q: Q function (State-Action Matrix)
        :param epsilon: Threshold for exploitation-exploitation ratio
        :return: Chosen action a
        """

        # Generate random number uniformly and compare to epsilon threshold - epsilon greedy
        if np.random.uniform(0, 1) < epsilon:

            # Choose action randomly - e.g. Exploration
            action = np.random.randint(1, self.get_nactions() + 1)
        else:

            # Choose action greedely - e.g. Exploitation
            action = Q.loc[state, :].idxmax(axis=1)

        return action

    def getGLIE(self, numOfEpisodes, epsilon, decayFactor=0.001, linear=True):
        """

        :param numOfEpisodes: Number of episodes
        :param epsilon: Threshold for exploitation-exploitation ratio
        :param decayFactor: Decay factor
        :param linear: True for linear decay, else exponential decay
        :return: GLIE's parameters (epsilons)
        """

        # Minimal exploitation-exploitation ratio of 0.1%
        minEpsilon = 0.001

        # Calculate GLIE's epsilons using linear decay
        if linear:

            GLIE = [max(minEpsilon, epsilon - (decayFactor * i)) for i in range(numOfEpisodes)]

        else:

            # Calculate GLIE's epsilons using exponential decay
            GLIE = [max(minEpsilon, epsilon * ((1-decayFactor) ** i)) for i in range(numOfEpisodes)]

        return GLIE

    def SARSA(self, numOfEpisodes, alpha, gamma, epsilon, decayFactor):
        """
            Sarsa Algorithm

        :param numOfEpisodes: Number of episodes
        :param alpha: Learning rate (step size)
        :param gamma: Discount factor (gamma), Default is 0.9
        :param epsilon: Initial threshold for exploitation-exploitation ratio
        :param decayFactor: GLIE's Decay factor
        :return:  Q function (State-Action Matrix); averageReward - Total reward divided by number of episodes
        """
        print("#############################################################################################")
        print("Executing Sarsa with the following hyperparameters:\n\u03B1 = {0}, \u03B3 = 0.9, \u03B5 = {1}, "
              "Exponential Decay Factor = {2}, Number of Episodes: {3}".format(alpha, epsilon, decayFactor,numOfEpisodes))
        print("#############################################################################################")


        # Sets random seed to get reproducible results
        np.random.seed(468)

        # Get GLIE's epsilons
        GLIE = self.getGLIE(numOfEpisodes, epsilon, decayFactor, False)

        # Q-function Dimensions
        shape = (self.get_nstates(), self.get_nactions())

        # Initialize Q-function arbitrarily with zeros
        Q = pd.DataFrame(np.zeros(shape, dtype=float), index=range(1, self.get_nstates() + 1),
                         columns=range(1, self.get_nactions() + 1))

        # Initialize list of rewards per episode
        totalRewardsPerEpisode = []

        # Iterate episodes
        for episode in range(numOfEpisodes):

            # Initialize total reward summarizer between each episode
            cumulativeReward = 0

            # Extract current episode epsilon
            eps = GLIE[episode]

            # Get agent's initial state (S0) using exploring starts
            self.reset(True)
            state = self.observation[0]

            # Initialize episode's stopping criteria
            done = False

            # Select action a based on initial state S0 using epsilon-greedy
            action = self.getEpsilonGreedyAction(state, Q, eps)

            # Iterate episode until the agent arrives terminal state
            while not done:
                # Retrieve new state S' based on current state S and selected action a
                new_state, reward, done = self.step(action)

                # Select new action by the given new state using epsilon greedy
                new_action = self.getEpsilonGreedyAction(new_state, Q, eps)

                # Sarsa's update rule
                Q.loc[state, action] = Q.loc[state, action] + alpha * (
                        reward + gamma * Q.loc[new_state, new_action] - Q.loc[state, action])

                # Update episode cumulative reward
                cumulativeReward += reward
                # Update current state S to new state S'
                state = new_state
                # Update current action a to new state a'
                action = new_action

            # Add cumulative reward to episodes reward list
            totalRewardsPerEpisode.append(cumulativeReward)

            # Prints progress ever thousand episodes
            if (episode + 1) % 1000 == 0:
                print("Episode: {0} , Average Reward: {1}".format(str(episode+1),
                                                                 str(sum(totalRewardsPerEpisode) / (episode + 1))))

        # Calculate average reward
        averageReward = (sum(totalRewardsPerEpisode) / numOfEpisodes)

        # Prints Average reward
        print("Final Episode: {0} , Average Reward: {1}".format(str(numOfEpisodes), str(averageReward)))

        self.close()

        return Q

    def QLearning(self, numOfEpisodes, alpha, gamma, epsilon, decayFactor):
        """
         Q-Learning Algorithm

        :param numOfEpisodes: Number of episodes
        :param alpha: Learning rate (step size)
        :param gamma: Discount factor (gamma), Default is 0.9
        :param epsilon: Initial threshold for exploitation-exploitation ratio
        :param decayFactor: GLIE's Decay factor
        :return:  Q function (State-Action Matrix); averageReward - Total reward divided by number of episodes
        """
        print("#############################################################################################")
        print("Executing Q-Learning with the following hyperparameters:\n\u03B1 = {0}, \u03B3 = 0.9, \u03B5 = {1}, "
              "Exponential Decay Factor = {2}, Number of Episodes: {3}".format(alpha, epsilon, decayFactor,numOfEpisodes))
        print("#############################################################################################")

        # Sets random seed to get reproducible results
        np.random.seed(468)

        # Get GLIE's epsilons
        GLIE = self.getGLIE(numOfEpisodes, epsilon, decayFactor)

        # Initialize list of rewards per episode
        totalRewardsPerEpisode = []

        # Q-function Dimensions
        shape = (self.get_nstates(), self.get_nactions())

        # Initialize Q-function arbitrarily with zeros
        Q = pd.DataFrame(np.zeros(shape, dtype=float), index=range(1, self.get_nstates() + 1),
                         columns=range(1, self.get_nactions() + 1))

        # Iterate episodes
        for episode in range(numOfEpisodes):

            # Extract current episode epsilon
            eps = GLIE[episode]

            # Get agent's initial state (S0) using exploring starts
            self.reset(True)
            state = self.observation[0]

            # Initialize total reward summarizer between each episode
            cumulativeReward = 0

            # Initialize episode's stopping criteria
            done = False

            # Iterate episode until the agent arrives terminal state
            while not done:
                # Select action a based on initial state S0 using epsilon-greedy
                action = self.getEpsilonGreedyAction(state, Q, eps)

                # Retrieve new state S' based on current state S and selected action a
                new_state, reward, done = self.step(action)

                # Sarsa's update rule
                Q.loc[state, action] = Q.loc[state, action] + alpha * (
                        reward + gamma * Q.loc[new_state, :].max() - Q.loc[state, action])

                # Update episode cumulative reward
                cumulativeReward += reward

                # Update current state S to new state S'
                state = new_state

            # Add cumulative reward to episodes reward list
            totalRewardsPerEpisode.append(cumulativeReward)

            # Prints progress ever thousand episodes
            if (episode + 1) % 1000 == 0:
                print("Episodes: {0}, Success rate: {1}".format(str(episode + 1),
                                                                str(sum(totalRewardsPerEpisode) / (episode + 1))))

        # Calculate average reward
        averageReward = (sum(totalRewardsPerEpisode) / numOfEpisodes)

        # Prints Average reward
        print("Final Episode: {0} , Average Reward: {1}".format(str(numOfEpisodes), str(averageReward)))

        self.close()

        return Q

    def optimize_Sarsa(self, numOfEpisodes, alpha, gamma, epsilon, decayFactor):
        """
        Optimizing Sarsa and storing evidence for future plots

        :param numOfEpisodes: Number of episodes
        :param alpha: Learning rate (step size)
        :param gamma: Discount factor (gamma), Default is 0.9
        :param epsilon: Initial threshold for exploitation-exploitation ratio
        :param decayFactor: GLIE's Decay factor
        :return: Q function (State-Action Matrix);
                episodes vector; time steps vector, total rewards vector; L1 norm vector
        """
        # Sets random seed to get reproducible results
        np.random.seed(468)
        # Get GLIE's epsilons
        GLIE = self.getGLIE(numOfEpisodes, epsilon, decayFactor, linear=False)
        # Initialize list of rewards per episode
        totalRewardsPerEpisode = []
        # Q-function Dimensions
        shape = (self.get_nstates(), self.get_nactions())
        # Initialize Q-function arbitrarily with zeros
        Q = pd.DataFrame(np.zeros(shape, dtype=float), index=range(1, self.get_nstates() + 1),
                         columns=range(1, self.get_nactions() + 1))

        # Initialize lists for storing evidence
        timesteps = []
        episodes = []
        L1_Norms = []

        # Iterate episodes
        for episode in range(numOfEpisodes):

            # Initialize time step counter between each episode
            t = 0
            # Extract current episode epsilon
            eps = GLIE[episode]
            # Get agent's initial state (S0) using exploring starts
            self.reset(True)
            state = self.observation[0]
            # Initialize total reward summarizer between each episode
            cumulativeReward = 0
            # Initialize episode's stopping criteria
            done = False
            # Select action a based on initial state S0 using epsilon-greedy
            action = self.getEpsilonGreedyAction(state, Q, eps)

            # Iterate episode until the agent arrives terminal state
            while not done:
                # Increse time steps counter
                t += 1
                # Retrieve new state S' based on current state S and selected action a
                new_state, reward, done = self.step(action)
                # Select new action by the given new state using epsilon greedy
                new_action = self.getEpsilonGreedyAction(new_state, Q, eps)
                # Sarsa's update rule
                Q.loc[state, action] = Q.loc[state, action] + alpha * (
                        reward + gamma * Q.loc[new_state, new_action] - Q.loc[state, action])
                # Update episode cumulative reward
                cumulativeReward += reward
                # Update current state S to new state S'
                state = new_state
                # Update current action a to new action a'
                action = new_action

            # Best state action values derived in the 1st MDP assignment, denoted as V-mdp
            V_opt = np.array(
                [0., 0.28592528, 0.07677605, 0.00829162, 0.74737133, 0.57641035, 0., -0.08591965, 0.92809858,
                 0.58410791, 0.1885577, 0.08029601, 0., 0., 0., -0.08591964])

            # Best state action values derived in the Q-Learning algorithm, denoted as V-Sarsa
            v_max = np.array(Q.max(axis=1))

            # Appending current episode measurements
            episodes.append(episode + 1)
            timesteps.append(t)
            totalRewardsPerEpisode.append(cumulativeReward)
            L1_Norms.append(np.linalg.norm(v_max - V_opt, ord=1))

            if (episode + 1) % 1000 == 0:
                print("Episodes: {0}, Success rate: {1}".format(str(episode + 1),
                                                                str(sum(totalRewardsPerEpisode) / (episode + 1))))
        # Calculate average reward
        averageReward = (sum(totalRewardsPerEpisode) / numOfEpisodes)

        # Print Summary
        print("Final Episode: {0}, Averaged Reward: {1}, L1-Norm: {2}".format(str(len(episodes)),
                                                                               str(averageReward),
                                                                               L1_Norms[-1]))
        self.close()

        return Q, episodes, timesteps, totalRewardsPerEpisode, L1_Norms

    def optimize_QLearning(self, numOfEpisodes, alpha, gamma, epsilon, decayFactor):
        """
        Optimizing Q-Learning and storing evidence for future plots

        :param numOfEpisodes: Number of episodes
        :param alpha: Learning rate (step size)
        :param gamma: Discount factor (gamma), Default is 0.9
        :param epsilon: Initial threshold for exploitation-exploitation ratio
        :param decayFactor: GLIE's Decay factor
        :return: Q function (State-Action Matrix);
                episodes vector; time steps vector, total rewards vector; L1 norm vector
        """
        # Sets random seed to get reproducible results
        np.random.seed(468)
        # Get GLIE's epsilons
        GLIE = self.getGLIE(numOfEpisodes, epsilon, decayFactor, linear=False)
        # Initialize list of rewards per episode
        totalRewardsPerEpisode = []
        # Q-function Dimensions
        shape = (self.get_nstates(), self.get_nactions())
        # Initialize Q-function arbitrarily with zeros
        Q = pd.DataFrame(np.zeros(shape, dtype=float), index=range(1, self.get_nstates() + 1),
                         columns=range(1, self.get_nactions() + 1))

        # Initialize lists for storing evidence
        timesteps = []
        episodes = []
        L1_Norms = []

        # Iterate episodes
        for episode in range(numOfEpisodes):

            # Initialize time step counter between each episode
            t = 0
            # Extract current episode epsilon
            eps = GLIE[episode]
            # Get agent's initial state (S0) using exploring starts
            self.reset(True)
            state = self.observation[0]
            # Initialize total reward summarizer between each episode
            cumulativeReward = 0
            # Initialize episode's stopping criteria
            done = False

            # Iterate episode until the agent arrives terminal state
            while not done:
                # Increse time steps counter
                t += 1
                # Select action a based on initial state S0 using epsilon-greedy
                action = self.getEpsilonGreedyAction(state, Q, eps)
                # Retrieve new state S' based on current state S and selected action a
                new_state, reward, done = self.step(action)

                # Q-Learning's update rule
                Q.loc[state, action] = Q.loc[state, action] + alpha * (
                        reward + gamma * Q.loc[new_state, :].max() - Q.loc[state, action])
                # Update episode cumulative reward
                cumulativeReward += reward
                # Update current state S to new state S'
                state = new_state

            # Best state action values derived in the 1st MDP assignment, denoted as V-mdp
            V_opt = np.array(
                [0., 0.28592528, 0.07677605, 0.00829162, 0.74737133, 0.57641035, 0., -0.08591965, 0.92809858,
                 0.58410791, 0.1885577, 0.08029601, 0., 0., 0., -0.08591964])

            # Best state action values derived in the Q-Learning algorithm, denoted as V-Sarsa
            v_max = np.array(Q.max(axis=1))

            # Appending current episode measurements
            episodes.append(episode + 1)
            timesteps.append(t)
            totalRewardsPerEpisode.append(cumulativeReward)
            L1_Norms.append(np.linalg.norm(v_max - V_opt, ord=1))

            if (episode + 1) % 1000 == 0:
                print("Episodes: {0}, Success rate: {1}".format(str(episode + 1),
                                                                str(sum(totalRewardsPerEpisode) / (episode + 1))))
        # Calculate average reward
        averageReward = (sum(totalRewardsPerEpisode) / numOfEpisodes)

        # Print Summary
        print("Final Episode: {0}, Averaged Reward: {1}, L1-Norm: {2}".format(str(len(episodes)),
                                                                              str(averageReward),
                                                                              L1_Norms[-1]))
        self.close()

        return Q, episodes, timesteps, totalRewardsPerEpisode, L1_Norms



    def plot_timesteps_episodes(self, episodes, timesteps, alpha, epsilon, decayFactor):
        """
         Plot Time steps per episode

        :param episodes: Vector of episodes
        :param timesteps: Vector of time steps per each episode
        :param alpha: Learning rate alpha
        :param epsilon: Initial GLIE's epsilon
        :param decayFactor: GLIE's decay factor
        :return:
        """
        # Figure Dimension
        plt.figure(figsize=(10, 5))

        # Plot Time steps per Episode
        plt.plot(episodes, timesteps)

        # Add Figure's Title and Axis Labels
        plt.title('Convergence Analysis\n\u03B1 = {0}, \u03B3 = 0.9, \u03B5 = {1}, Decay = {2}'.format(alpha, epsilon,
                                                                                                       decayFactor))
        plt.xlabel('Episodes')
        plt.ylabel('Timesteps')

        # timestamp = str(dt.datetime.now().strftime("%d.%m.%Y_%H.%M"))
        # plt.savefig("Episodes Timesteps Chart - ep={0}_a={1}_df={2}_{3}.png".format(episodes[-1], alpha, decayFactor,
        #                                                                             timestamp))

        plt.show()

        return plt

    def plot_episodes_rewards(self, episodes, rewards, alpha, epsilon, decayFactor, window_size=250):

        # Figure Dimension
        plt.figure(figsize=(15, 5))

        # Plot Reward per Episodes
        plt.plot(episodes, rewards, label="Reward")

        # Calculate smoothed mean reward using moving average based on a given window size (default=250)
        smoothedMeanRewards = [np.mean(rewards[n - window_size:n]) if n > window_size else np.mean(rewards[:n])
                               for n in range(1, len(rewards))]

        # Add smoothed mean reward to plot
        plt.plot(smoothedMeanRewards, label="Smoothed Average Reward (Window: {0})".format(window_size))

        # Add Figure's Title and Axis Labels
        plt.title('Reward per Episode\n\u03B1 = {0}, \u03B3 = 0.9, \u03B5 = {1}, Decay = {2}'.format(alpha, epsilon,
                                                                                                     decayFactor))
        plt.xlabel('Episodes')
        plt.ylabel('Reward Per Episode')
        plt.legend(loc="lower right")

        # Annotate maximal smoothed average reward position
        for i, j in zip(episodes, smoothedMeanRewards):
            if (j == max(smoothedMeanRewards)):
                plt.annotate("({:.0f},{:.2f})".format(i, j), xy=(i, j))

        # timestamp = str(dt.datetime.now().strftime("%d.%m.%Y_%H.%M"))
        # plt.savefig(
        #     'Reward per Episode - ep={0}_a={1}_df={2}_{3}.png'.format(episodes[-1], alpha, decayFactor, timestamp))

        plt.show()

    def plot_L1_episodes(self, episodes, L1_Norms, alpha, epsilon, decayFactor):

        # Figure Dimension
        plt.figure(figsize=(15, 5))

        # Plot L1 norm per Episodes
        plt.plot(episodes, L1_Norms, label="L1 norm")

        # Add Figure's Title and Axis Labels
        plt.title(
            'L1 norm per Episode\n\u03B1 = {0}, \u03B3 = 0.9, \u03B5 = {1}, Decay = {2}'.format(alpha, epsilon,
                                                                                                decayFactor))
        plt.xlabel('Episodes')
        plt.ylabel('L1 norm')
        plt.legend(loc="lower right")

        # Annotate minimal L1 norm position
        for i, j in zip(episodes, L1_Norms):
            if j == min(L1_Norms):
                plt.annotate("({:.0f},{:.2f})".format(i, j), xy=(i, j + 0.5))

        timestamp = str(dt.datetime.now().strftime("%d.%m.%Y_%H.%M"))
        plt.savefig(
            'L1 norm per Episode - ep={0}_a={1}_df={2}_{3}.png'.format(episodes[-1], alpha, decayFactor, timestamp))

        plt.show()

        return plt

    def plot_actionValues(self, QMatrix):
        """
        Plots state-action values including the highest state values and agent's policy

        :param QMatrix: Q function (State-Action Matrix)
        :return:
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import warnings
        warnings.filterwarnings("ignore")

        # Calculate the optimal policy based on the given Q-function
        optimalPolicy = np.array(QMatrix.idxmax(axis=1))

        # Terminal states list
        terminatedStates = np.array(self.stateHoles + self.stateGoal) - 1
        np.put(optimalPolicy, terminatedStates, [i * 0 for i in range(len(terminatedStates))])

        # Sets Figure's Dimensions
        QPlot = plt.figure(constrained_layout=False, figsize=(10, 10))
        gridWorld = gridspec.GridSpec(4, 4)
        gridWorld.update(wspace=0, hspace=0)

        # Triangle Positions
        trianglePositions = [dict(xy=(0.1, 0.9), xycoords='axes fraction', va='center', ha='center'),
                     dict(xy=(0.5, 0.8), xycoords='axes fraction', va='center', ha='center'),
                     dict(xy=(0.75, 0.5), xycoords='axes fraction', va='center', ha='center'),
                     dict(xy=(0.5, 0.2), xycoords='axes fraction', va='center', ha='center'),
                     dict(xy=(0.25, 0.5), xycoords='axes fraction', va='center', ha='center')]

        # Vertical Triangles positions - e.g. South and North
        VerticalTriangles = np.array([[0, 0], [5, 5], [10, 0], [0, 10], [5, 5], [10, 10]])

        # Horizontal Triangles positions - e.g. West and East
        HorizontalTriangles = np.array([[0, 0], [5, 5], [0, 10], [10, 10], [5, 5], [10, 0]])

        # Triangles boarder anf fill colors for triangle's vertices
        basic_colors = ['black', 'black', 'black', 'black', 'black', 'black']
        optimal_colors = ['green', 'green', 'green', 'green', 'green', 'green']

        # Init row and columns
        row = 0
        col = -1

        # Plot each state square
        for state in range(1, self.get_nstates() + 1):

            # Modify state coordinates within grid
            if state % self.nRows == 1:
                row = 0
                col = col + 1
            else:
                row = row + 1

            # Plot square as Hole
            if state in self.stateHoles:

                QPlot.add_subplot(gridWorld[row, col]).set_facecolor('#b25858')

                QPlot.add_subplot(gridWorld[row, col]).annotate('{0}'.format(state), fontsize=24, color="#08043E",
                                                                **trianglePositions[0])
            # Plot square as Goal
            elif state in self.stateGoal:

                QPlot.add_subplot(gridWorld[row, col]).set_facecolor('#589b59')
                QPlot.add_subplot(gridWorld[row, col]).annotate('{0}'.format(state), fontsize=24, color="#08043E",
                                                                **trianglePositions[0])
            # Plot state with action values triangles
            else:
                # Set North Triangle
                QPlot.add_subplot(gridWorld[row, col]).annotate('{0}'.format(state), fontsize=24, color="#08043E",
                                                                **trianglePositions[0])
                north_triangle = plt.Polygon(VerticalTriangles[3:7, :], color=basic_colors[0], fill=False)
                plt.gca().add_patch(north_triangle)
                # Set East Triangle
                east_triangle = plt.Polygon(HorizontalTriangles[3:7, :], color=basic_colors[0], fill=False)
                plt.gca().add_patch(east_triangle)
                plt.scatter(VerticalTriangles[:, 0], VerticalTriangles[:, 1], s=1, color=basic_colors[:])
                # Set South Triangle
                south_triangle = plt.Polygon(VerticalTriangles[:3, :], color=basic_colors[0], fill=False)
                plt.gca().add_patch(south_triangle)
                plt.scatter(HorizontalTriangles[:, 0], HorizontalTriangles[:, 1], s=1, color=basic_colors[:])
                # Set West Triangle
                west_triangle = plt.Polygon(HorizontalTriangles[:3, :], color=basic_colors[0], fill=False)
                plt.gca().add_patch(west_triangle)

                # Annotate each triangle with its suitable Q values and fill with green in case of optimal action value
                for action in range(1, self.get_nactions() + 1):

                    if action == optimalPolicy[state - 1]:

                        if optimalPolicy[state - 1] == 1:

                            north_triangle = plt.Polygon(VerticalTriangles[3:7, :], color=optimal_colors[0])
                            plt.gca().add_patch(north_triangle)

                        elif optimalPolicy[state - 1] == 2:

                            east_triangle = plt.Polygon(HorizontalTriangles[3:7, :], color=optimal_colors[0])
                            plt.gca().add_patch(east_triangle)

                        elif optimalPolicy[state - 1] == 3:

                            south_triangle = plt.Polygon(VerticalTriangles[:3, :], color=optimal_colors[0])
                            plt.gca().add_patch(south_triangle)

                        else:

                            west_triangle = plt.Polygon(HorizontalTriangles[:3, :], color=optimal_colors[0])
                            plt.gca().add_patch(west_triangle)

                    # Annotate Q values
                    plt.gca().annotate('{:.3f}'.format(QMatrix.loc[state, action]), **trianglePositions[action])

            # Remove Axis tick marks
            ax = QPlot.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)

        plt.show()

        return QPlot

    def plot_state_values(self, QMatrix):

        # Calculate the optimal policy values based on the given Q-function
        optimalStateActionValues = np.array(QMatrix.max(axis=1))

        # Calculate the optimal policy based on the given Q-function
        optimalPolicy = np.array(QMatrix.idxmax(axis=1))

        # Terminal states list
        terminatedStates = np.array(self.stateHoles + self.stateGoal) - 1
        np.put(optimalPolicy, terminatedStates, [i * 0 for i in range(len(terminatedStates))])

        fig = plt.plot(1)
        self._plot_world()
        k = 0
        for i in range(self.nCols):
            for j in range(self.nRows, 0, -1):
                if k + 1 not in self.stateHoles + self.stateGoal:
                    plt.text(i + 0.5, j - 0.5, str(self._truncate(optimalStateActionValues[k], 3)), fontsize=12,
                             horizontalalignment='center', verticalalignment='center')
                k += 1


        plt.title('Frozenlake State Values', size=16)
        plt.axis("equal")
        plt.axis("off")
        plt.show()

        self.plot_policy(optimalPolicy)
