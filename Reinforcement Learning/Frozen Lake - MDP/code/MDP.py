from World import World
import numpy as np
import pandas as pd

class MDP:
    __world = World()

    def __init__(self, prob, reward=-0.04):
        self.__statesNumber = self.__world.get_nstates()
        self.__states = np.array([state for state in range(1, self.__statesNumber+1)])  # Denoted as "S-Cross"
        self.__stateHoles = np.array(self.__world.get_stateHoles())
        self.__stateGoal = np.array(self.__world.get_stateGoal())
        self.__terminated_states = np.concatenate((self.__stateHoles, self.__stateGoal))
        self.__S = np.array(self.__states[np.isin(self.__states, self.__terminated_states, invert=True)]) # Denoted as "S"
        self.__actionsNumber = self.__world.get_nactions() # Denoted as |A(S)|
        self.__actions = {'N': 1, 'E': 2, 'S': 3, 'W': 4} # Denoted as A(S)
        self.__transitionMatrix = self.buildMDPTransitionMatrix(prob)
        self.__rewards = np.array([-1, reward, reward, reward, reward, reward, -1, reward, reward, reward, reward, reward, 1, -1, -1, reward])
        self.__values = np.zeros((self.__statesNumber), dtype=float) # Denoted as V(S)
        self.__pie = np.zeros((self.__statesNumber), dtype=int) # Denoted as pie(S)

    def getValuesVector(self):
        return self.__values

    def getPieVector(self):
        return self.__pie

    def getTerminatedStates(self):
        return self.__terminated_states

    def getRewardsVector(self):
        return self.__rewards

    def getActions(self):
        return self.__actions

    def getTransitionMatrix(self):
        return self.__transitionMatrix

    def performValueIteration(self, gamma, theta):
        return self.valueIteration(theta, gamma)



    def valueIteration(self, theta, gamma):
        """
        ## Value Iteration Algorithm ##

        :param theta: Stopping parameter for algorithm convergence
        :param gamma: Discount factor
        :return: valuesVector, pieVector: State Values Vector "V" and Policy Vector "pie"
        """

        # Extract MDP Params: Transition Matrix, Values Vector, and Policy vector
        transitionMatrix = self.__transitionMatrix
        valuesVector = self.__values
        pieVector = self.__pie

        # Init a State-Action-Value Temp Array
        stateActionValue = np.zeros((4), dtype=float)

        # Vector of all states except Terminated States, denotes as "S".
        S = self.__S - 1

        # Init delta to enter 1st iteration
        delta = theta + 1

        # Repeat until delta < theta
        while delta >= theta:

            # Initialize delta
            delta = 0.

            # Iterate through all states in S
            for state in S:

                # Store current state's value
                v = valuesVector[state]

                # Iterate Actions { 1: "North", 2: "East", 3: "South", 4:"West" }
                for action in range(self.__actionsNumber):

                    # Calculate State-Action Value for current state and each action
                    stateActionValue[action] = np.dot(transitionMatrix[state, :, action], (self.__rewards + gamma*valuesVector))

                # Select best State-Action Value
                valuesVector[state] = np.max([stateActionValue[action] for action in range(self.__actionsNumber)])

                # Select best Action
                pieVector[state] = np.argmax([stateActionValue[action] for action in range(self.__actionsNumber)])

                # Calculate Delta
                delta = np.max([delta, np.abs(v - valuesVector[state])])

        return valuesVector, pieVector + 1

    def policyEvaluation(self, pieVector, theta, gamma):

        """
        ## Policy Evaluation Algorithm ##

        :param pieVector: Policy Vector "pie"
        :param theta: Stopping parameter for algorithm convergence
        :param gamma: Discount factor
        :return: State Values Vector "V" for the given policy vector "pie"
        """

        transitionMatrix = self.__transitionMatrix
        valuesVector = self.__values

        # Vector of all states except Terminated States, denotes as "S".
        S = self.__S - 1

        # Init delta to enter 1st iteration
        delta = theta + 1

        # Repeat until delta < theta
        while delta >= theta:

            # Initialize Delta
            delta = 0.

            # Iterate through all states in S (and not S+)
            for state in S:

                # Store current state's value
                v = valuesVector[state]

                # Calculate State-Action Value for current state and its action by the given policy
                valuesVector[state] = np.dot(transitionMatrix[state, :, pieVector[state]],
                                                      (self.__rewards + gamma * valuesVector))

                # Calculate Delta
                delta = np.max([delta, np.abs(v - valuesVector[state])])

        return valuesVector

    def policyImprovement(self, valuesVector, gamma):

        """
             ## Policy Improvement Algorithm ##

             :param valuesVector: State Values Vector "V" of a given policy
             :param gamma: Discount factor
             :return: valuesVector, pieVector: Improved State Values Vector "V" and Policy Vector "pie"
             """

        transitionMatrix = self.__transitionMatrix
        pieVector = np.zeros(self.__statesNumber, dtype=int)

        # Init a State-Action-Value Temp Array
        Q_Function = np.zeros(4, dtype=float)

        # Vector of all states except Terminated States, denotes as "S".
        S = self.__S - 1

        # Iterate through all states in S
        for state in S:

            # Iterate Actions { 1: "North", 2: "East", 3: "South", 4:"West" }
            for action in range(self.__actionsNumber):

                # Calculate Q Function for current state and each action
                Q_Function[action] = np.dot(transitionMatrix[state, :, action],(self.__rewards + gamma * valuesVector))

            # Choose the best Q-function value q*
            valuesVector[state] = np.max([Q_Function[action] for action in range(self.__actionsNumber)])

            # Choose the best action a
            pieVector[state] = np.argmax([Q_Function[action] for action in range(self.__actionsNumber)])

        return pieVector, valuesVector

    def performPolicyIteration(self, theta, gamma):
        """
        ## Policy Iteration Algorithm ##

        :param theta: Stopping parameter for algorithm convergence
        :param gamma: Discount factor
        :return: State Values Vector "V" for the given policy vector "pie"
        """

        # Seed for reproducible results
        np.random.seed(0)

        # Initialize policy vector with an arbitrary policy (~ Uniform Discrete Distribution - 0.25 per each action a)
        pieVector = np.random.randint(1, 5, self.__statesNumber) - 1

        # Initialize stability indicator
        policy_stable = False

        # Loop until convergence
        while not policy_stable:

            # Evaluate a given policy vector "pie" using Policy Evaluation algorithm
            V = self.policyEvaluation(pieVector, theta, gamma)

            # Improve the given policy based on the values vector "V"
            pieVectorPrime, valuesVector = self.policyImprovement(V, gamma)

            # In case the policy doesn't change for all of the states
            if (pieVectorPrime == pieVector).all():
                policy_stable = True

            # Update pie vector to pie' Vector
            pieVector = pieVectorPrime

            # Display plots per each iteration
            # self.__world.plot_value2(valuesVector, "Policy", 0.9, -0.04, 0.0001)
            # self.__world.plot_policy(pieVector+1)

        return pieVector+1, valuesVector

    def buildMDPTransitionMatrix(self, prob):

        """

        :param prob: Probability of walking straight
        :return: MDP's Transition Matrix
        """

        # Get number of states and actions
        statesNum = self.__world.get_nstates()
        actionsNum = self.__world.get_nactions()

        # Setting Grid edges
        upperEdgeStates = np.array([1, 5, 9, 13])
        leftEdgeStates = np.array([1, 2, 3, 4])
        rightEdgeStates = np.array([13, 14, 15, 16])
        lowerEdgeStates = np.array([4, 8, 12, 16])

        # Transition Matrix dimensions
        dim = (statesNum, statesNum, actionsNum)
        transitionMatrix = np.zeros(dim)

        # Iterating actions
        for action in range(actionsNum):

            # Iterating States
            for fromState in range(statesNum):

                # Terminated States Only ( Holes & Goal )
                if fromState in (self.__terminated_states - 1):

                    transitionMatrix[fromState, fromState, action] = 1

                # Lower Right Corner
                elif (fromState in (lowerEdgeStates - 1)) and (fromState in (rightEdgeStates - 1)):

                    if action == 0:
                        transitionMatrix[fromState, fromState, action] = (1 - prob) / 2
                        transitionMatrix[fromState, fromState - 1, action] = prob
                        transitionMatrix[fromState, fromState - 4, action] = (1 - prob) / 2

                    elif action == 1:
                        transitionMatrix[fromState, fromState, action] = prob + (1 - prob) / 2
                        transitionMatrix[fromState, fromState - 1, action] = (1 - prob) / 2

                    elif action == 2:
                        transitionMatrix[fromState, fromState, action] = prob + (1 - prob) / 2
                        transitionMatrix[fromState, fromState - 4, action] = (1 - prob) / 2

                    else:
                        transitionMatrix[fromState, fromState, action] = (1 - prob) / 2
                        transitionMatrix[fromState, fromState - 1, action] = (1 - prob) / 2
                        transitionMatrix[fromState, fromState - 4, action] = prob

                # Lower Left Corner
                elif (fromState in (lowerEdgeStates - 1)) and (fromState in (leftEdgeStates - 1)):

                    if action == 0:
                        transitionMatrix[fromState, fromState - 1, action] = prob
                        transitionMatrix[fromState, fromState, action] = (1 - prob) / 2
                        transitionMatrix[fromState, fromState + 4, action] = (1 - prob) / 2

                    elif action == 1:
                        transitionMatrix[fromState, fromState - 1, action] = (1 - prob) / 2
                        transitionMatrix[fromState, fromState, action] = (1 - prob) / 2
                        transitionMatrix[fromState, fromState + 4, action] = prob

                    elif action == 2:
                        transitionMatrix[fromState, fromState, action] = prob + (1 - prob) / 2
                        transitionMatrix[fromState, fromState + 4, action] = (1 - prob) / 2

                    else:
                        transitionMatrix[fromState, fromState, action] = prob + (1 - prob) / 2
                        transitionMatrix[fromState, fromState - 1, action] = (1 - prob) / 2

                # Rest of the Lower edges
                elif fromState in (lowerEdgeStates - 1):

                    if action == 0:

                        transitionMatrix[fromState, fromState - 1, action] = prob

                        transitionMatrix[fromState, fromState + 4, action] = (1 - prob) / 2
                        transitionMatrix[fromState, fromState - 4, action] = (1 - prob) / 2

                    elif action == 1:
                        transitionMatrix[fromState, fromState + 4, action] = prob
                        transitionMatrix[fromState, fromState, action] = (1 - prob) / 2
                        transitionMatrix[fromState, fromState - 1, action] = (1 - prob) / 2

                    elif action == 2:
                        transitionMatrix[fromState, fromState, action] = prob
                        transitionMatrix[fromState, fromState - 4, action] = (1 - prob) / 2
                        transitionMatrix[fromState, fromState + 4, action] = (1 - prob) / 2

                    else:
                        transitionMatrix[fromState, fromState - 4, action] = prob
                        transitionMatrix[fromState, fromState - 1, action] = (1 - prob) / 2
                        transitionMatrix[fromState, fromState, action] = (1 - prob) / 2

                # Rest of the Left edges
                elif fromState in (leftEdgeStates - 1):

                    if action == 0:

                        transitionMatrix[fromState, fromState - 1, action] = prob
                        transitionMatrix[fromState, fromState + 4, action] = (1 - prob) / 2
                        transitionMatrix[fromState, fromState, action] = (1 - prob) / 2

                    elif action == 1:
                        transitionMatrix[fromState, fromState + 4, action] = prob
                        transitionMatrix[fromState, fromState + 1, action] = (1 - prob) / 2
                        transitionMatrix[fromState, fromState - 1, action] = (1 - prob) / 2

                    elif action == 2:
                        transitionMatrix[fromState, fromState + 1, action] = prob
                        transitionMatrix[fromState, fromState, action] = (1 - prob) / 2
                        transitionMatrix[fromState, fromState + 4, action] = (1 - prob) / 2

                    else:
                        transitionMatrix[fromState, fromState, action] = prob
                        transitionMatrix[fromState, fromState - 1, action] = (1 - prob) / 2
                        transitionMatrix[fromState, fromState + 1, action] = (1 - prob) / 2

                # Rest of the Upper edges
                elif fromState in (upperEdgeStates - 1):

                    if action == 0:
                        transitionMatrix[fromState, fromState, action] = prob
                        transitionMatrix[fromState, fromState + 4, action] = (1 - prob) / 2
                        transitionMatrix[fromState, fromState - 4, action] = (1 - prob) / 2

                    elif action == 1:
                        transitionMatrix[fromState, fromState + 4, action] = prob
                        transitionMatrix[fromState, fromState + 1, action] = (1 - prob) / 2
                        transitionMatrix[fromState, fromState, action] = (1 - prob) / 2

                    elif action == 2:
                        transitionMatrix[fromState, fromState + 1, action] = prob
                        transitionMatrix[fromState, fromState - 4, action] = (1 - prob) / 2
                        transitionMatrix[fromState, fromState + 4, action] = (1 - prob) / 2

                    else:
                        transitionMatrix[fromState, fromState - 4, action] = prob
                        transitionMatrix[fromState, fromState, action] = (1 - prob) / 2
                        transitionMatrix[fromState, fromState + 1, action] = (1 - prob) / 2
                else:

                    if action == 0:
                        transitionMatrix[fromState, fromState - 1, action] = prob
                        transitionMatrix[fromState, fromState + 4, action] = (1 - prob) / 2
                        transitionMatrix[fromState, fromState - 4, action] = (1 - prob) / 2

                    elif action == 1:
                        transitionMatrix[fromState, fromState + 4, action] = prob
                        transitionMatrix[fromState, fromState + 1, action] = (1 - prob) / 2
                        transitionMatrix[fromState, fromState - 1, action] = (1 - prob) / 2

                    elif action == 2:
                        transitionMatrix[fromState, fromState + 1, action] = prob
                        transitionMatrix[fromState, fromState - 4, action] = (1 - prob) / 2
                        transitionMatrix[fromState, fromState + 4, action] = (1 - prob) / 2

                    else:
                        transitionMatrix[fromState, fromState - 4, action] = prob
                        transitionMatrix[fromState, fromState - 1, action] = (1 - prob) / 2
                        transitionMatrix[fromState, fromState + 1, action] = (1 - prob) / 2

        return transitionMatrix

    def getRewardFunction(self):
        """
        :return: Reward Function r(s, a, s')
        """
        # Initialize Data Frame
        rewardsFunction = pd.DataFrame(index=[i for i in range(1, self.__statesNumber+1)], columns=['N', 'E', 'S', 'W'])

        # Iterate Data Frame's action columns
        for ind, column in enumerate(list(rewardsFunction)):

            # Calculate Reward Function using equation (13)
            currentRewardVector = np.sum(np.multiply(self.getTransitionMatrix()[:, :, ind], self.getRewardsVector()), axis=1)

            # Set rewards within terminal states to zero
            np.put(currentRewardVector, [self.getTerminatedStates() - 1], [np.zeros(len(self.getTerminatedStates()))])

            # Assign Action Column
            rewardsFunction[column] = currentRewardVector

        return rewardsFunction

if __name__ == "__main__":

    world = World()

        # Question 2 - Section B
    frozenLake = MDP(prob=0.8, reward=-0.04)
    valuesVector, pieVector = frozenLake.performValueIteration(theta=0.0001, gamma=1)
    world.plot_value2(valuesVector, "Value", 1, -0.04, 0.0001)
    world.plot_policy(pieVector)

    # Question 2 - Section C
    frozenLake = MDP(prob=0.8, reward=-0.04)
    valuesVector, pieVector = frozenLake.performValueIteration(theta=0.0001, gamma=0.9)
    world.plot_value2(valuesVector, "Value", 0.9, -0.04, 0.0001)
    world.plot_policy(pieVector)

    # Question 2 - Section D
    frozenLake = MDP(prob=0.8, reward=-0.02)
    valuesVector, pieVector = frozenLake.performValueIteration(theta=0.0001, gamma=1)
    world.plot_value2(valuesVector, "Value", 1, -0.02, 0.0001)
    world.plot_policy(pieVector)

    # Question 2 - Section E
    frozenLake = MDP(prob=0.8, reward=-0.04)
    pieVector, valuesVector = frozenLake.performPolicyIteration(theta=0.0001, gamma=0.9)
    world.plot_value2(valuesVector, "Policy", 0.9, -0.04, 0.0001)
    world.plot_policy(pieVector)




