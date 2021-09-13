from HostNode import HostNode
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
from ndlib.viz.bokeh.DiffusionTrend import DiffusionTrend
from bokeh.io import show
from DataGeneration import *
import random


class IndependentCascadesModel:

    def __init__(self, G, df, iter_num):
        self.__graph = G
        self.__df = df
        self.__iter_num = iter_num

    def create_independent_cascades_model(self, infected_nodes_seed):
        '''
        Creates an Independent Cascades model and the model's iterations

        :param infected_nodes_seed: the hostNode that is the origin for the worm's spreading (Influence Maximization)
        :return: model (Independent Cascades model)
                iterations (of the algorithm for the given hostNode seed)
        '''
        # Initialize Independent Cascades Model
        model = ep.IndependentCascadesModel(self.__graph)

        # Model Configuration -  to specify model parameters:
        config = mc.Configuration()
        config.add_model_initial_configuration("Infected", infected_nodes_seed)

        # Create df from our graph:
        edges_df = pd.DataFrame(data=self.__graph.edges.data(), columns=['i', 'j', 'w'])
        weights_by_order = []  # by order in graph
        for i in range(0, len(edges_df['w'])):
            weight = edges_df['w'][i]['weight']  # 'w' is a dictionary
            weights_by_order.append(weight)
        edges_df = edges_df.drop(columns='w')
        edges_df['w'] = weights_by_order
        # edges_df.to_csv('edges_df.csv')

        # Setting the edge parameters:
        i = 0
        for e in self.__graph.edges():
            threshold = edges_df['w'][i]
            config.add_edge_configuration("threshold", e, threshold)
            i += 1

        model.set_initial_status(config)
        iterations = model.iteration_bunch(self.__iter_num)

        return model, iterations

    def change_df_for_ML(self, df):
        '''
        create the df for ML - for each hostNode seed:
        - Drop irrelevant features
        - Convert the Security Components feature to different Binary features
        - change the features' order in the dataframe

        :param df: the iterations dataframe
        :return: the  iterations dataframe - after changes
        '''
        new_df = df

        # Assign '1' in the security-componentes-columns:
        for index, row in df.iterrows():
            sec_comp_list = row['Security Components']
            for sec_comp in sec_comp_list:
                new_df.at[index, str(sec_comp)] = '1'

        # Drop irrelevant columns:
        new_df = new_df.drop(
            columns=['IP', 'Subnet Mask', 'Subnet Mask Decimal', 'Network Address', 'Broadcast Address', 'Network',
                     'Network Hosts Amount', 'Security Components', 'Security Weight', 'Spread Weight', 'HostNode', 'Connected Hosts',
                     'Total Score'])

        # Change the columns order in df:
        new_df = new_df[
            ['Node', 'Server Type', 'Importance Score', 'Is Connector', 'Connected Hosts Amount', 'Firewall',
             'AntiVirus', 'VPN', 'Proxy', 'Security Policies', 'OS Version',
             'Total nodes infected', 'Iterations to convergence', 'Nodes infected in 1 iter', 'Y']]

        new_df.fillna('0', inplace=True)

        return new_df

    def create_iterations_features(self, df, infected_nodes_seed, iterations, statuses_df_seed_i):
        '''
        Creates new features for the RF model based on the model's iterations.

        :param df: the network dataframe
        :param infected_nodes_seed: the hostNode that is the origin for the worm's spreading (Influence Maximization)
        :param iterations: iterations of the algorithm for the given hostNode seed
        :param statuses_df_seed_i: dataframe for counting the number of times each node in the network is infected
        :return: df - the network dataframe with the new features
                 statuses_df_seed_i - updated dataframe for counting the number of times each node in the network is infected
        '''
        df_index = infected_nodes_seed
        iterations_df = pd.DataFrame(data=iterations, columns=['iteration', 'status', 'node_count', 'status_delta'])
        # iterations_df.to_csv('iterations_df.csv')

        # Feature of amount of times infected for each node:
        for iteration_dict in iterations_df['status']:
            for key in iteration_dict:               # key = node id
                status = iteration_dict[key]
                if status == 2:
                    previous_times_infected = statuses_df_seed_i.at[key, 'times_infected']
                    statuses_df_seed_i.at[key, 'times_infected'] = previous_times_infected + 1

        # New features:
        nodes_infected_in_1_iter = 0
        iterations_to_convergence = 0
        total_nodes_infected = 0
        for i in range(0, len(iterations_df['iteration'])):
            if i != (len(iterations_df['iteration']) - 1):
                j = i + 1
                count_0_nodes_i = iterations_df['node_count'][i][0]  # Node status 0 - Susceptible (vulnerable)
                count_1_nodes_i = iterations_df['node_count'][i][1]  # Node status 1 - Infected
                count_2_nodes_i = iterations_df['node_count'][i][2]  # Node status 2 - Removed

                # Results of first iteration:
                if i == 1:
                    nodes_infected_in_1_iter = iterations_df['status_delta'][i][1]
                    # print('infected_in_1_iter: ', nodes_infected_in_1_iter)

                # Last iteration - Convergence:
                status_is_empty_i = not bool(iterations_df['status'][i])
                status_is_empty_j = not bool(iterations_df['status'][j])
                if (status_is_empty_i == False) & (status_is_empty_j == True):  # Last iteration
                    iterations_to_convergence = iterations_df['iteration'][i]
                    # print('iterations to convergence: ', iterations_to_convergence)
                    total_nodes_infected = count_2_nodes_i
                    # print('total_nodes_infected: ', total_nodes_infected)

        df.at[df_index, 'Total nodes infected'] = total_nodes_infected
        df.at[df_index, 'Iterations to convergence'] = iterations_to_convergence
        df.at[df_index, 'Nodes infected in 1 iter'] = nodes_infected_in_1_iter
        df.at[df_index, 'Y'] = (total_nodes_infected) / (iterations_to_convergence + 0.0000000001)

        # df.to_csv('df_with_iterations_features.csv')
        return df, statuses_df_seed_i

    def visualize_independent_cascades_model(self, model, iterations):
        '''
        Visualize the Independent Cascades model

        :param model: Independent Cascades model
        :param iterations: the model's iterations
        :return: None
        '''
        trends = model.build_trends(iterations)
        viz = DiffusionTrend(model, trends)
        p = viz.plot(width=400, height=400)
        show(p)
