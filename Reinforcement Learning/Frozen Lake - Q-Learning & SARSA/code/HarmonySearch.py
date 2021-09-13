from pyharmonysearch import ObjectiveFunctionInterface, harmony_search
from math import pow
import random
from bisect import bisect_left
from multiprocessing import cpu_count
from pprint import pprint
from World import World
import numpy as np


class ObjectiveFunction(ObjectiveFunctionInterface):

    """
        This is a the objective function that contains a mixture of continuous and discrete variables.
        Goal:
            minimize L1 -Norm between V-mdp to V-Sarsa and V-QL
    """

    def __init__(self): # numOfepisodes, alpha, epsilon, decay_factor
        self._lower_bounds = [1000, 0, 0, 0.0001]
        self._upper_bounds = [100000, 1, 1, 0.1]
        self._variable = [True, False, True, True]
        self._discrete_values = [[x for x in np.arange(1000, 100000, 1000)],
                                 [y for y in np.arange(0, 1, 0.001)],
                                 [z for z in np.arange(0, 1, 0.1)],
                                 [dc for dc in np.arange(0.001, 0.1, 0.001)]]

        # define all input parameters
        self._maximize = False  # do we maximize or minimize?
        self._max_imp = 150  # maximum number of improvisations
        self._hms = 20  # harmony memory size
        self._hmcr = 0.7  # harmony memory considering rate
        self._par = 0.5  # pitch adjusting rate
        self._mpap = 0.25  # maximum pitch adjustment proportion (new parameter defined in pitch_adjustment()) - used for continuous variables only
        self._mpai = 10  # maximum pitch adjustment index (also defined in pitch_adjustment()) - used for discrete variables only
        self._random_seed = 468
        self.__env = World()
        self.__ImprovCounter = 0


    def use_random_seed(self):
        return hasattr(self, '_random_seed') and self._random_seed

    def get_random_seed(self):
        return self._random_seed

    def get_fitness(self, vector):
        self.__ImprovCounter = self.__ImprovCounter + 1

        print(
            "Current Improvisation: {0}, episodes: {1}, alpha: {2}, gamma: 0.9, Starting GLIE: {3}, decay_param:{4}".format(
                self.__ImprovCounter,
                vector[0],
                vector[1],
                vector[2], vector[3]))

        QMatrixSARSA, averageReward = self.__env.SARSA(vector[0], vector[1], 0.9, vector[2], vector[3])

        V_opt = np.array(
            [0., 0.28592528, 0.07677605, 0.00829162, 0.74737133, 0.57641035, 0., -0.08591965, 0.92809858, 0.58410791,
             0.1885577, 0.08029601, 0., 0., 0., -0.08591964])

        v_max = np.array(QMatrixSARSA.max(axis=1))

        print("L1 Norm: {0}".format(np.linalg.norm(v_max - V_opt, ord=1)))
        return np.linalg.norm(v_max - V_opt, ord=1)#averageReward

    def get_value(self, i, j=None):

        if self.is_discrete(i):
            if j:
                return self._discrete_values[i][j]
            return self._discrete_values[i][random.randint(0, len(self._discrete_values[i]) - 1)]
        return random.uniform(self._lower_bounds[i], self._upper_bounds[i])


    def get_lower_bound(self, i):
        """
            This won't be called except for continuous variables, so we don't need to worry about returning None.
        """
        return self._lower_bounds[i]

    def get_upper_bound(self, i):
        """
            This won't be called except for continuous variables.
        """
        return self._upper_bounds[i]

    def get_num_discrete_values(self, i):
        if self.is_discrete(i):
            return len(self._discrete_values[i])
        return float('+inf')

    def get_index(self, i, v):
        """
            Because self.discrete_values is in sorted order, we can use binary search.
        """
        return ObjectiveFunction.binary_search(self._discrete_values[i], v)

    @staticmethod
    def binary_search(a, x):
        """
            Code courtesy Python bisect module: http://docs.python.org/2/library/bisect.html#searching-sorted-lists
        """
        i = bisect_left(a, x)
        if i != len(a) and a[i] == x:
            return i
        raise ValueError

    def is_variable(self, i):
        return self._variable[i]

    def is_discrete(self, i):
        return self._discrete_values[i] is not None

    def get_num_parameters(self):
        return len(self._lower_bounds)

    def use_random_seed(self):
        return hasattr(self, '_random_seed') and self._random_seed

    def get_max_imp(self):
        return self._max_imp

    def get_hmcr(self):
        return self._hmcr

    def get_par(self):
        return self._par

    def get_hms(self):
        return self._hms

    def get_mpai(self):
        return self._mpai

    def get_mpap(self):
        return self._mpap

    def maximize(self):
        return self._maximize

if __name__ == '__main__':
    obj_fun = ObjectiveFunction()
    num_processes = cpu_count()  # use number of logical CPUs
    num_iterations = 1#num_processes * 5  # each process does 5 iterations
    # print([[random.randint(-1000, 1000) for _ in range(3)] for _ in range(10)])
    initial_harmonies = [[1000, 0.9, 1], [2000,0.9, 0.9], [3000,0.9, 1], [4000,0.9, 1], [5000,0.7, 1],
                         [6000,0.3, 1], [7000,0.5, 1], [8000,0.9, 1], [9000,0.85, 1], [12000,0.4, 0.8]]

    results = harmony_search(obj_fun, num_processes, num_iterations)#, initial_harmonies=initial_harmonies)
    print('Elapsed time: {}\nBest harmony: {}\nBest fitness: {}\nHarmony memories:'.format(results.elapsed_time, results.best_harmony, results.best_fitness))
    pprint(results.harmony_memories)
    best = results.best_harmony
    env = World()
    QMatrixSARSA, averagedReward = env.SARSA(best[0], best[1], 0.9, best[2])
    env.plot_QFunction(QMatrixSARSA)