import time
import HostNode
from HostNode import HostNode
# from DataGeneration import create_weighted_directed_graph
from DataGeneration import DataGeneration
# from DataGeneration import create_independent_cascades_model
# from DataGeneration import visualize_independent_cascades_model
from IndependentCascadesModel import IndependentCascadesModel
import pandas as pd
import numpy as np
import datetime
from sklearn.metrics import jaccard_score
import itertools
import threading
from subprocess import check_output
from RandomForestRegressor import *
import networkx as nx


class MemeticAlgorithm:

    def __init__(self, networkDataFrame, NetworkTopologyGraph, simThreshold, k_iter=20, maxGen=3, crossProb=0.5,
                 mutateProb=0.3):
        self.__networkDataFrame = networkDataFrame
        self.__networkHostNodes = networkDataFrame["HostNode"].to_numpy()
        self.__hostNodesConnectedHostsAmount = np.array(
            [hostNode.getConnectedHostsAmount() for hostNode in self.__networkHostNodes])
        self.__NetworkTopologyGraph = NetworkTopologyGraph
        self.__simThreshold = simThreshold
        self.__maxGen = maxGen
        self.__k_iter = k_iter
        self.__RF_DF = None
        self.__crossProb = crossProb
        self.__mutateProb = mutateProb
        self.__ICM = IndependentCascadesModel(NetworkTopologyGraph, networkDataFrame, k_iter)

    def getHostNodeCandidates(self):

        return self.__networkHostNodes

    def getMaxGen(self):

        return self.__maxGen

    def getHostNode(self, ip):
        """
        Extract Host Node object based on IP
        :param ip: a given IP address
        :return: Host Node object
        """
        idx = self.__networkDataFrame[self.__networkDataFrame['IP'] == ip].index.values[0] - 1

        return self.__networkDataFrame["HostNode"].iloc[idx]

    def getHostNodeRecord(self, hostNode):
        """
        Extract Host Node record from the data frame based on the HostNode object
        :param hostNode: a given hostNode object
        :return: Host Node record
        """
        idx = self.__networkDataFrame[self.__networkDataFrame['HostNode'] == hostNode].index.values[0] - 1

        return self.__networkDataFrame.iloc[idx]

    def getHostNodeID(self, hostNode):
        """
        Extract Host Node ID from the data frame based on the HostNode object
        :param hostNode: a given hostNode object
        :return: Host Node ID
        """
        return self.__networkDataFrame[self.__networkDataFrame['HostNode'] == hostNode].index.values[0] - 1

    @staticmethod
    def covertIPtoSimilarityRepresentation(hostNode):
        """
        Converts an IP Address to a digit-wise array

        :param hostNode: A HostNode object
        :return: IP represented as list of ordered integers
        """

        # Splits IPV4 address to 4 decimal sections
        hostNode_Vector = ''.join(digit + "" for digit in str(hostNode.getIP())).split('.')

        # Initialize a converted IP string
        convertedIPAddress = ''

        # Iterate IP sections and fill with zeros in case of a single\two digits number
        for chunk in hostNode_Vector:
            if len(chunk) > 2:

                convertedIPAddress += chunk

            elif len(chunk) > 1:

                convertedIPAddress += "0" + chunk

            else:
                convertedIPAddress += "00" + chunk

        # Returns IP as list of ordered integers
        return np.array([digit for digit in convertedIPAddress], dtype=np.int)

    def calculateIPsJaccardSimilarity(self, hostNode1, hostNode2):
        """
        Calculate Jaccard Similarity between 2 Host nodes' IPs

        :param hostNode1: 1st Host Node to be compared
        :param hostNode2: 2nd Host Node to be compared
        :return: Jaccard Similarity
        """
        # hostNode1_binary = '.'.join([bin(int(x)+256)[3:] for x in str(hostNode1.getIP()).split('.')])
        # hostNode2_binary = '.'.join([bin(int(x) + 256)[3:] for x in str(hostNode2.getIP()).split('.')])

        # Compare IPs' decimal digits
        isEqualDigits = (self.covertIPtoSimilarityRepresentation(hostNode1) == self.covertIPtoSimilarityRepresentation(
            hostNode2))

        # Calculate IPs' intersection score
        similarValuesIntersection = np.sum(isEqualDigits)

        # Calculate Jaccard Similarity score
        jaccardScore = similarValuesIntersection / len(isEqualDigits)

        return jaccardScore

    def simNeighbor(self, hostNode, hostNodeCandidatesList):
        """
        Creates a list of similar neighbors above the specified similarity threshold i.e. simThreshold
        :param hostNode: A given Host Node
        :param hostNodeCandidatesList: A list of Candidate Host Nodes
        :return: A list of similar neighbors
        """
        # Initialize an empty list for neighbors
        neighbors = []

        # Iterate Host Nodes Candidates
        for candidate in hostNodeCandidatesList:

            # Include neighbors with a similarity greater than specified threshold
            if self.calculateIPsJaccardSimilarity(hostNode, candidate) > self.__simThreshold:
                neighbors.append(candidate)

        return np.array(neighbors)

    def SHD_Algorithm(self, chromosome_size, candidates):

        """
        Similarity based High Degree method (SHD)

        :param chromosome_size: Outputted Chromosome size
        :param candidates: List of Candidates
        :return: A Chromosome with diversified highest degrees Host Nodes
        """

        # Initialize a k-node chromosome
        chromosome_X_a = np.array([])

        # Initialize list of total Host Nodes candidates within the whole network
        totalHostNodeCandidates = candidates

        # Create chromosome of k Host Nodes
        for i in range(chromosome_size):

            temp_candidates_out_degrees = [candidate.getConnectedHostsAmount() for candidate in totalHostNodeCandidates]

            # choose a Host Node with the highest out degree
            highestOutDegreeCandidate = totalHostNodeCandidates[np.argmax(temp_candidates_out_degrees)]

            # Add Host Node with the highest degree to Chromosome Xa
            chromosome_X_a = np.append(chromosome_X_a, highestOutDegreeCandidate)

            # Locate neighbor candidate nodes that are similar with the existing highest Degree Candidate node
            simNeighbor = self.simNeighbor(highestOutDegreeCandidate, totalHostNodeCandidates)

            # Excludes similar neighbor nodes from list of Host Nodes' candidate nodes
            totalHostNodeCandidates = np.setdiff1d(totalHostNodeCandidates, simNeighbor, assume_unique=True)

            # In case no more candidates nodes
            if totalHostNodeCandidates.size == 0:
                # The rest of the Host Nodes are chosen randomly from full Candidate list
                randomCandidate = np.random.choice(np.setdiff1d(
                    self.getHostNodeCandidates(), chromosome_X_a, assume_unique=True), chromosome_size - i - 1)

                # Update Chromosome with the rest of the candidate nodes
                chromosome_X_a = np.append(chromosome_X_a, randomCandidate)

                break

        return chromosome_X_a

    def populationInitialization(self, popSize=50, seedSize=3):
        """
        Initializes Population of Host Nodes chromosomes

        :param popSize: Size of the population to be generated
        :param seedSize: Size of seed chromosome (i.e. amount of host nodes )
        :return: population - A list of chosen population
        """

        print("#################################################################")
        print("########### Initializing Population with size of {0}  ###########".format(popSize))
        print("#################################################################")

        # Generate a half of population based on SHD function
        half_population_size = round(popSize / 2)

        # Initialize Population list
        population = np.array([None for _ in range(popSize)])

        # Find Host Node candidates using SHD method
        HostNodesCandidates = self.SHD_Algorithm(half_population_size, self.getHostNodeCandidates())

        # Create half of population
        for i in range(half_population_size):

            # Creating a Chromosome with a size according the given seed size
            population[i] = self.SHD_Algorithm(seedSize, self.getHostNodeCandidates())

            # Iterate Genes within SHD resulted Chromosome
            for j in range(seedSize):

                # 50% for Gene to be replaced with a random Host Node
                if np.random.uniform(0, 1) > 0.5:
                    # Select another Host Node
                    randomCandidateHostNode = np.random.choice(
                        np.setdiff1d(self.getHostNodeCandidates(), population[i][j],
                                     assume_unique=True), 1)
                    # Update Chromosome
                    np.put(population[i], [j], [randomCandidateHostNode])

        # Select k different nodes from the Candidate to initialize X_i based on SHD
        for i in range(half_population_size, popSize):
            population[i] = np.random.choice(HostNodesCandidates, seedSize)

        return population

    @staticmethod
    def is_status_delta_empty(single_iteration_dict):
        """

        :param single_iteration_dict:
        :return:
        """
        isEmpty = True
        for key in single_iteration_dict["status_delta"].keys():

            if single_iteration_dict["status_delta"][key] != 0:
                isEmpty = False
                break

        return isEmpty

    def extract_last_iteration(self, iterations_dict):
        """
        Extract last iteration of maximum infections based on Influence Cascade Model

        :param iterations_dict: A ICM Iterations scores dictionary
        :return: last iteration of Influence Cascade Model
        """

        lastIter = len(iterations_dict) - 1

        for i in np.arange(len(iterations_dict) - 1, 1, -1):

            if self.is_status_delta_empty(iterations_dict[i]):

                lastIter = i

            else:

                break

        return lastIter

    def extract_total_infected(self, iterations_dict):

        return iterations_dict[self.extract_last_iteration(iterations_dict)]["node_count"][2]

    # @staticmethod
    # def extractChromosomeIDs(chromosome):
    #
    #     return [hostNode.getID() for hostNode in chromosome]

    def extractChromosomeIDs(self, chromosome):
        """
        Extract Genes IDs from chromosome
        :param chromosome: A given Chromosome
        :return: A list of Genes IDs
        """

        return [self.__networkDataFrame[self.__networkDataFrame['HostNode'] == gene].index.values[0] for gene in
                chromosome]

    def ICMThreads(self, chromosome, originsIDsRes, ICMScoresRatiosRes, index):
        """
        Independent Cascade Model Storing results procedure

        :param chromosome: a given chromosome to be tested
        :param originsIDsRes: Origins IDs results
        :param ICMScoresRatiosRes: Origins ICM ratio results
        :param index: Thread index
        :return: True when completed successfully
        """

        try:

            # Extract Origin's Host Nodes IDs
            originIDs = self.extractChromosomeIDs(chromosome)
            originsIDsRes[index] = originIDs

            # Single Independent Cascade Episode per
            ICM, iterations = self.__ICM.create_independent_cascades_model(originIDs)

            # Calculate ICM Score Ratios by dividing total infected computers by maximal iteration for convergences
            ICM_RatioScore = self.extract_total_infected(iterations) / self.extract_last_iteration(iterations)

            # Update Results
            ICMScoresRatiosRes[index] = ICM_RatioScore

        except Exception:

            originsIDsRes[index] = False
            ICMScoresRatiosRes[index] = False
            raise Exception("Problem IN THREAD")

        return True

    def executeICMForMultipleOriginsWithThreads(self, origins_list):
        """
        Execute Influence Independent Cascade Model per each set of origins Host Nodes

        :param origins_list: List of origins to be tested
        :return:originsIDs, ICM_Ratio - Lists of Origins IDs, ICM Ratio score respectively
        """

        print("#################################################################")
        print("############ IM Independent Cascade Model Evaluation ############")
        print("#################################################################")

        # Initialize Threads, Scores and Iterations arrays
        threads = []
        originsIDsRes = [None for _ in origins_list]
        ICMScoresRatiosRes = [None for _ in origins_list]

        # initiate threads' ID counter
        T_ID = 0

        # Iterate chromosomes (i.e. Sets of origins)
        for chromosome in origins_list:
            T_ID += 1

            # Create process for each ICM
            process = threading.Thread(target=self.ICMThreads,
                                       args=(chromosome, originsIDsRes, ICMScoresRatiosRes, T_ID))

            # Start process
            process.start()

            # Append process to thread list
            threads.append(process)

        # Join threads results
        for process in threads:
            process.join()

        return originsIDsRes, ICMScoresRatiosRes

    def executeICMForMultipleOrigins(self, origins_list):
        """
        Execute Influence Independent Cascade Model per each set of origins Host Nodes

        :param origins_list: List of origins to be tested
        :return:originsIDs, ICM_Ratio - Lists of Origins IDs, ICM Ratio score respectively
        """

        print("#################################################################")
        print("############ IM Independent Cascade Model Evaluation ############")
        print("#################################################################")

        # Initialize Scores and Iterations arrays
        originsIDs = []
        ICMScoresRatios = []

        # Iterate chromosomes (i.e. Sets of origins)
        for chromosome in origins_list:
            # Extract Origin's Host Nodes IDs
            originIDs = self.extractChromosomeIDs(chromosome)

            # Single Independent Cascade Episode per
            ICM, iterations = self.__ICM.create_independent_cascades_model(originIDs)

            # Store Results
            originsIDs.append(originIDs)

            # Calculate ICM Score Ratios by dividing total infected computers by maximal iteration for convergences
            ICM_RatioScore = self.extract_total_infected(iterations) / self.extract_last_iteration(iterations)

            # Storing Results
            ICMScoresRatios.append(ICM_RatioScore)

        return originsIDs, ICMScoresRatios

    @staticmethod
    def parentsSelection(population, parentsPoolSize=9, tourSize=5):
        """
        Parent Selection Procedure for Genetic Operations
        :param population: Population of chromosomes candidate
        :param parentsPoolSize: Future parents' pool size
        :param tourSize: Tournament size
        :return: Pool of selected parents i.e. P_parent
        """

        print("#################################################################")
        print("######### Selecting parents from chromosomes population  ########")
        print("#################################################################")

        # Initialize list of parents, candidates and fitness scores
        parents = []
        candidates = population["Population"]
        fitness = population["ICMScoreRatios"]

        # Creation of Pool of parents chromosomes
        for parent in range(parentsPoolSize):

            # Initialize best parent index
            bestParentsInd = None

            # Execute tournament
            for i in range(tourSize):

                # Selects a random chromosome candidate
                candidateSetInd = np.random.randint(0, len(population["Population"]))

                # Update best parent index based on candidates tournament
                if (bestParentsInd is None) or fitness[candidateSetInd] > fitness[bestParentsInd]:
                    bestParentsInd = candidateSetInd

            # Store best parent chromosome
            bestParents = candidates[bestParentsInd]

            # Append best parents to the pool of parents
            parents.append(bestParents)

            # Exclude last chosen parent from candidates list
            candidates = np.setdiff1d(population["Population"], candidates, assume_unique=True)

        return np.array(parents)

    @staticmethod
    def shareSimilarGenes(parentA, parentB):
        """
        Checks if two given parents share similar genes
        :param parentA: Chromosome A of host nodes
        :param parentB: Chromosome B of host nodes
        :return: True if have common genes, otherwise False
        """
        # Init boolean status
        areSimilar = False

        # Iterate Parents genes
        for i in range(len(parentA)):

            # Check if parent A gene exist also in parent B, update status respectively
            if parentA[i] in parentB:
                areSimilar = True
                break

        return areSimilar

    @staticmethod
    def chromosomeSimilarity(chromosomeA, chromosomeB):
        """
        Calculate Jaccard similarity between two given chromosomes

        :param chromosomeA: Chromosome A of host nodes
        :param chromosomeB: Chromosome B of host nodes
        :return: Jaccard similarity score
        """

        # Similar genes counter
        similarGenes = 0

        # Iterate both chromosomes
        for i in range(len(chromosomeA)):
            for j in range(len(chromosomeB)):

                # Increase counter in case of identical genes
                if chromosomeA[i] == chromosomeB[j]:
                    similarGenes += 1

        # Calculate Jaccard Similarity
        chromosomeSimilarity = similarGenes / len(chromosomeA)

        return chromosomeSimilarity

    @staticmethod
    def hasSimilarGenes(chromosome):
        """
        Checks if a chromosome contains identical genes
        :param chromosome: Chromosome of host nodes
        :return: True if contains similar genes (i.e. similar host nodes), otherwise False
        """

        # Init boolean status
        hasSimilar = False

        # Iterate genes within the given chromosome
        for i in range(len(chromosome)):
            for j in range(len(chromosome)):

                # Check if genes are identical
                if chromosome[i] == chromosome[j] and i != j:
                    hasSimilar = True
                    break
                    break

        return hasSimilar

    @staticmethod
    def singleCrossover(parentA, parentB):
        """
        Executes a single crossover for a given chromosomes couple
        :param parentA: Chromosome Parent A
        :param parentB: Chromosome Parent B
        :return: Two offsprings chromosomes
        """

        # Generate crossover uniformly at random
        crossing_over_position = np.random.randint(1, len(parentA))

        # Create 1st offspring
        offspringA = np.concatenate(([parentA[i] for i in range(crossing_over_position)],
                                     [parentB[i] for i in range(crossing_over_position, len(parentB))]), axis=None)
        # Create 2nd offspring
        offspringB = np.concatenate(([parentB[i] for i in range(crossing_over_position)],
                                     [parentA[i] for i in range(crossing_over_position, len(parentA))]), axis=None)

        return np.array([offspringA, offspringB])

    def singleMutation(self, hostNodeGene):
        """

        :param hostNodeGene:
        :return:
        """

        # Initialize a mutant
        mutant = None

        # Iterate host node's connected hosts candidates
        for hostNodeCandidate in hostNodeGene.getConnectedHosts():

            # Calculate Jaccard Similarity between the given host node to candidates
            jaccardSimilarity = self.calculateIPsJaccardSimilarity(hostNodeGene, self.getHostNode(hostNodeCandidate))

            # Assign a mutant differ (dissimilar) to the given host node
            if jaccardSimilarity < self.__simThreshold:
                mutant = self.getHostNode(hostNodeCandidate)
                break

        # In case all host nodes are similar, choose uniformly at random
        if mutant is None:
            mutant = self.getHostNode(
                hostNodeGene.getConnectedHosts()[np.random.randint(len(hostNodeGene.getConnectedHosts()))])

        return mutant

    def Crossover(self, parentsForCross):

        """
        Genetic Crossover operations
        :param parentsForCross: A list of chromosome parents for mating
        :return: a list of chromosomes offsprings
        """

        print("#################################################################")
        print("################ Performs Crossover Operation ###################")
        print("#################################################################")

        # Init list of chromosomes offsprings
        p_child = []

        # Iterate all non-identical chromosomes couples
        for crossParentAInd in range(len(parentsForCross)):
            for crossParentBInd in range(len(parentsForCross)):

                # In case of non-identical chromosomes with no similar genes send to crossover list
                if crossParentAInd != crossParentBInd and not self.shareSimilarGenes(parentsForCross[crossParentAInd],
                                                                                     parentsForCross[crossParentBInd]):
                    p_child.append(
                        self.singleCrossover(parentsForCross[crossParentAInd], parentsForCross[crossParentBInd]))

        if len(p_child) > 0:
            return np.concatenate(p_child, axis=0)

        else:
            return parentsForCross

    def Mutation(self, p_child):

        print("#################################################################")
        print("################# Performs Mutation Operation ###################")
        print("#################################################################")
        print("######################## Before Mutation ########################")
        print("#################################################################")

        # Prints Child IPs before mutation
        self.printChildIPs(p_child)

        # Mutation Procedure
        for chromosomeInd in range(len(p_child)):

            tempChromosome = p_child[chromosomeInd]

            for geneInd in range(len(p_child[chromosomeInd])):

                if np.random.uniform(0, 1) <= self.__mutateProb:

                    tempChromosome[geneInd] = self.singleMutation(p_child[chromosomeInd][geneInd])
                    if not self.hasSimilarGenes(tempChromosome):
                        p_child[chromosomeInd][geneInd] = tempChromosome[geneInd]

        print("#################################################################")
        print("######################## After Mutation #########################")
        print("#################################################################")

        # Prints Child IPs after mutation
        self.printChildIPs(p_child)

        return p_child

    @staticmethod
    def printChildIPs(p_child):
        """
        Prints genes IP addresses
        :param p_child: A list of genes to be printed
        :return:
        """
        p_id = 0

        # Print each host node child IP address
        for child in p_child:
            p_id += 1

            print("Child {0} : {1}".format(p_id, [child[i].getIP() for i in range(len(child))]))

    def geneticOperation(self, p_parents):
        """
        Genetic Operation procedure for generating a new generation of chromosomes
        :param p_parents: List of parents chromosomes for mating
        :return: A new Generation of chromosomes
        """

        print("#################################################################")
        print("################## Performs Genetic Operations ##################")
        print("#################################################################")

        # Initialize a parents of chromosomes
        parentsForCross = []

        # Iterate parents chromosomes
        for parentInd in range(len(p_parents)):

            # Choose parent chromosome for crossover with a chance of the specified crossProb
            if np.random.uniform(0, 1) <= self.__crossProb:
                parentsForCross.append(p_parents[parentInd])

        # Send selected parent for Crossover operation
        if len(parentsForCross) > 1:
            p_child = self.Crossover(parentsForCross)
        else:
            p_child = p_parents

        # Send child chromosomes to Mutation Procedure
        p_child = self.Mutation(p_child)

        return p_child

    def findBestNeighbor(self, Chromosome, candidates):
        """
        Finds Chromosome's best nearest neighbor
        :param Chromosome: A given chromosome of host nodes
        :param candidates: A list of candidates
        :return: Most influential nearest neighbor chromosome
        """
        # Initialize a set of neighbors
        nearestNeighbors = []

        # Iterate candidates
        for candidate in candidates:

            # Append similar neighbors chromosomes
            if self.__simThreshold < self.chromosomeSimilarity(Chromosome, candidate) < 1:
                nearestNeighbors.append(candidate)

        if len(nearestNeighbors) == 0:
            nearestNeighbors.append(candidates[np.random.randint(0, len(candidates))])

        # Evaluate similar neighbors using ICM
        IDs, ICMScoresRatios = self.executeICMForMultipleOrigins(nearestNeighbors)

        # Retrieve the most influential nearest neighbor
        bestNeighbor = nearestNeighbors[np.argmax(ICMScoresRatios)]

        return bestNeighbor

    def Eval(self, chromosome):
        """
        Evaluate the fitness of a chromosome using ICM
        :param chromosome: Chromosome to be evaluated
        :return: Chromosome's ICM Score ratio
        """
        N_nextIDs, ICMScoreRatio = self.executeICMForMultipleOrigins([chromosome])

        return ICMScoreRatio[0]

    def localSearch(self, p_child):

        print("#################################################################")
        print("##################### Performs Local Search #####################")
        print("#################################################################")

        # Evaluate Child chromosomes
        childIDs, childICMScoreRatios = self.executeICMForMultipleOrigins(p_child)

        # Best Child chromosome
        N_current = p_child[np.argmax(childICMScoreRatios)]

        # Initialize locality status
        isLocal = False

        print("#################################################################")
        print("############### Searching Best Nearest Neighbor #################")
        print("#################################################################")

        # Repeat until convergence
        while not isLocal:

            # Find best nearest neighbor
            N_next = self.findBestNeighbor(N_current, p_child)

            # Evaluate nearest neighbor ICM Ratio
            N_next_ICM_score = self.Eval(N_next)

            # Evaluate best child chromsome
            N_current_ICM_score = self.Eval(N_current)

            # Update best chromosome in case of a better fitness
            if N_next_ICM_score > N_current_ICM_score:

                N_current = N_next
                N_current_ICM_score = N_next_ICM_score

            else:
                isLocal = True

        return np.array([N_current]), N_current_ICM_score

    @staticmethod
    def updatePopulation(popScoresDict, p_new, P_new_score):
        """
        Update chromosomes population by selecting the chromosomes with the highest fitness
        :param popScoresDict: A dictionary of chromosomes with their ICM scores
        :param p_new: New chromosome candidate
        :param P_new_score: New chromosome candidate ICM score
        :return: An updated chromosomes population
        """
        # Find weakest chromosome in population
        weakestChromosomeInd = np.argmin(popScoresDict["ICMScoreRatios"])

        # Find weakest chromosome ICM score
        weakestChromosomeICMScore = np.min(popScoresDict["ICMScoreRatios"])

        # If new chromosome is better replaced it with the weakest chromosome
        if P_new_score > weakestChromosomeICMScore:
            popScoresDict["Population"][weakestChromosomeInd] = p_new[0]
            popScoresDict["ICMScoreRatios"][weakestChromosomeInd] = P_new_score

        return popScoresDict

    def convertSeedToIDs(self, seed):
        """
        Convert seed of host nodes to a list of IDs
        :param seed: Seed (i.e. chromosome) of k Host Nodes objects
        :return: a list of Host Nodes' IDs
        """
        IDs = [self.getHostNodeID(hostNode) for hostNode in seed]

        return IDs

    @staticmethod
    def printBestSeedInfo(seed, duration, bestSeedICMscore, rf_avg_prediction):
        """
        Prints Chromosome's information
        :param seed: A chromosome seed conssists of HostNode objects
        :param duration: Total duration of Meme-IM algorithm
        :param bestSeedICMscore: Best seed Im-ICM score
        :param rf_avg_prediction: Best seed average probability to be infected using Random Forest Estimator
        :return:
        """
        print(
            "****************************************************************************************************\n"
            "- Total Duration: {0}\n"
            "- Most Influential Host Nodes resulted in an ICM Ratio of {1}\n"
            "- Random Forest Estimated Probability of chosen hosts to be infected is {2}%\n"
            "- The chosen Hosts to be secured are:".format(
                duration, bestSeedICMscore, rf_avg_prediction))

        for hostNode in seed:
            hostNode.print()

        print("****************************************************************************************************\n")


if __name__ == '__main__':
    # Random seed for reproducible results
    np.random.seed(1111)

    startTime = datetime.datetime.now()
    initialHostsList = []

    # Read df_for_ML_combined from pickle
    df_for_ML_combined_pkl_filename = "df_for_ML_combined.pkl"
    df_for_ML_combined_pkl = open(df_for_ML_combined_pkl_filename, 'rb')
    df_for_ML_combined = pickle.load(df_for_ML_combined_pkl)

    # Create Random Forest Regressor object
    RF = RandomForest(df_for_ML_combined)

    # network 1 - all hosts related to host_1:
    host1 = HostNode('192.168.7.1', 24, 'A', 9, 0.01, None, False, False)
    initialHostsList.append(host1)

    # network 2 - all hosts related to host_2:
    host2 = HostNode('10.10.8.2', 24, 'B', 6, 0.4, None, False, False)
    initialHostsList.append(host2)

    # Create Data Generator object
    DG = DataGeneration(initialHostsList)

    # Generate Network's data frame
    networkDataFrame, hostNodes = DG.create_whole_network_data()

    # Create Network Topology based on a directed graph with the edges calculated weights
    NetworkTopologyGraph = DG.create_weighted_directed_graph(networkDataFrame)

    # Creates a Meme-IM Object
    MA = MemeticAlgorithm(networkDataFrame, NetworkTopologyGraph, 0.6, 20, 3, 0.5, 0.3)

    # Population initialization
    Population = MA.populationInitialization(popSize=40, seedSize=3)

    # Calculate ICM Score Ratios
    OriginsIDs, ICMScoreRatios = MA.executeICMForMultipleOrigins(Population)

    # Best individual initialization
    P_Best = Population[np.argmax(ICMScoreRatios)]
    P_Best_score = np.max(ICMScoreRatios)

    # Create a dictionary of chromosomes with their ICM scores
    populationScoresDict = dict(Population=Population, ICMScoreRatios=ICMScoreRatios)

    # Generations Counter
    t = 0

    while t < MA.getMaxGen():

        print("#################################################################")
        print("##################### Breading Generation {0} #####################".format(t + 1))
        print("#################################################################")

        # Select parental chromosomes for mating
        P_Parents = MA.parentsSelection(populationScoresDict, parentsPoolSize=18, tourSize=5)

        # Perform Genetic operators
        P_Child = MA.geneticOperation(P_Parents)

        # Perform Local Search
        P_New, P_New_Score = MA.localSearch(P_Child)

        # Update Population
        populationScoresDict = MA.updatePopulation(populationScoresDict, P_New, P_New_Score)

        # Update P_best
        if P_New_Score > P_Best_score:
            P_Best = P_New
            P_Best_score = P_New_Score

        # Increase Generations Counter
        t += 1

    # Calculate total Duration
    total_duration = datetime.datetime.now() - startTime

    # Trained Regressor model
    RF_regressor_model_pkl_filename = 'Regressor_model.pkl'
    RF_regressor_model_pkl = open(RF_regressor_model_pkl_filename, 'rb')
    RF_regressor_model = pickle.load(RF_regressor_model_pkl)

    print("#################################################################")
    print("######## Random Forest Infection Probability Estimation #########")
    print("#################################################################")

    # Prediction for the chosen hostNodes:
    avg_prediction = round(
        RF.predict_HostNodes(MA.convertSeedToIDs(P_Best), df_for_ML_combined, RF_regressor_model) * 100, 2)

    # Print Best Seed information
    MA.printBestSeedInfo(P_Best, total_duration, P_Best_score, avg_prediction)