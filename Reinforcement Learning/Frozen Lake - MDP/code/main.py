# -*- coding: utf-8 -*-

from World import World
from MDP import MDP

if __name__ == "__main__":

     world = World()

     # Question 2 - Section B
     frozenLake = MDP(prob=0.8, reward=-0.04)
     valuesVector, pieVector = frozenLake.performValueIteration(theta=0.0001, gamma=1)
     world.plot_value(valuesVector)
     world.plot_policy(pieVector)

     # Question 2 - Section C
     frozenLake = MDP(prob=0.8, reward=-0.04)
     valuesVector, pieVector = frozenLake.performValueIteration(theta=0.0001, gamma=0.9)
     world.plot_value(valuesVector)
     world.plot_policy(pieVector)

     #Question 2 - Section D
     frozenLake = MDP(prob=0.8, reward=-0.02)
     valuesVector, pieVector = frozenLake.performValueIteration(theta=0.0001, gamma=1)
     world.plot_value(valuesVector)
     world.plot_policy(pieVector)


     # Question 2 - Section E
     frozenLake = MDP(prob=0.8, reward=-0.04)
     pieVector, valuesVector = frozenLake.performPolicyIteration(theta=0.0001, gamma=0.9)
     world.plot_value(valuesVector)
     world.plot_policy(pieVector)




