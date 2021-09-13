import random
import pandas as pd
from HostNode import HostNode
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import datetime


class DataGeneration:

    def __init__(self, initialHostsList):
        self.__initialHostsList = initialHostsList

    def get_importance_score(self, type):
        '''
        Get the importance score of each server based on the server type

        :param type: the server type
        :return: importance score
        '''
        if type == 'Application':
            importance_score = 6
        if type == 'Computing':
            importance_score = 5
        if type == 'Database':
            importance_score = 8
        if type == 'FTP':
            importance_score = 7
        if type == 'Mail':
            importance_score = 8
        if type == 'Proxy':
            importance_score = 9
        if type == 'Web':
            importance_score = 6

        return importance_score

    def normalize(self, importance_score):
        '''
        Normalize the importance score by dividing it by the sum of all importance scores

        :param importance_score: importance score of the node
        :return: normalized importance score
        '''
        sum_importance_scores = sum([1, 6, 5, 8, 7, 8, 9, 6])
        normalized_importance_score = importance_score / sum_importance_scores

        return normalized_importance_score

    def get_security_weight(self, comp):
        '''
        Get the security weight of each security component

        :param comp: the security component name
        :return: the security weight
        '''
        if comp == 'Firewall':
            sec_weight = 0.3
        if comp == 'OS Version':
            sec_weight = 0.15
        if comp == 'AntiVirus':
            sec_weight = 0.2
        if comp == 'Security Policies':
            sec_weight = 0.15
        if comp == 'VPN':
            sec_weight = 0.1
        if comp == 'Proxy':
            sec_weight = 0.1

        return sec_weight

    def get_components(self, num_sec_components):
        '''
        Randomly chose the list of security components the node has
        based on prior probabilities and on the number of security components it has

        :param num_sec_components: the number of security components the node has
        :return: security_components - a list of security components for the node
                 security_weight - a summed weight calculated for each node based on the weight of each of its security components
        '''
        security_components = []
        num_chosen_comp = 0
        security_weight = 0
        while num_chosen_comp != num_sec_components:
            comp = random.choices(['Firewall', 'OS Version', 'AntiVirus', 'Security Policies', 'VPN', 'Proxy'],
                                  weights=[0.3, 0.1, 0.3, 0.15, 0.05, 0.1], k=1)
            comp = comp[0]
            if (comp in security_components) == False:
                security_components.append(comp)
                weight = self.get_security_weight(comp)
                security_weight = security_weight + weight
                num_chosen_comp += 1

        return security_components, security_weight

    def create_network_data(self, host):
        '''
        Creates the data of the "Network Features" (features regarding the network itself):
        Network Features: IP, Subnet Mask, Subnet Mask Decimal, Network Address, Broadcast Address, Network, Network Hosts Amount

        :param host: the initial host that the data is generated for
        :return: data - dataframe with the addition of Network Features
        '''
        ips = host.getNetworkHosts()  # with the initial host
        data = pd.DataFrame()
        data['IP'] = ips
        data['Subnet Mask'] = host.getSubnetMask()
        data['Subnet Mask Decimal'] = host.getSubnetMaskDecimal()
        data['Network Address'] = host.getNetworkAddress()
        data['Broadcast Address'] = host.getBroadcastAddress()
        data['Network'] = host.getNetworkType()
        data['Network Hosts Amount'] = host.getHostsAmount()

        return data

    def create_nodes_data(self, host, data):
        '''
        Creates the data of the "Nodes Features" (features regarding the nodes):
        Nodes Features: Node, Server Type, Importance score

        :param host:  the initial host that the data is generated for
        :param data:  data - dataframe with the addition of Nodes Features
        :return: data - dataframe with the addition of Nodes Features
        '''
        # Nodes - Node, Server Type, Importance Score.
        hosts_amount = host.getHostsAmount()
        nodes = []
        server_types = []
        importance_scores = []
        for i in range(hosts_amount):
            node_list = random.choices(['host', 'server'], weights=[0.8, 0.2],
                                       k=1)  # host_prob = 0.8, server_prob = 0.2
            node = node_list[0]
            if node == 'host':
                type = 'host'
                importance_score = 1
            if node == 'server':
                type_list = random.choices(['Application', 'Computing', 'Database', 'FTP', 'Mail', 'Proxy', 'Web'],
                                           weights=[0.1, 0.06, 0.23, 0.17, 0.23, 0.04, 0.17], k=1)
                type = type_list[0]
                importance_score = self.get_importance_score(type)

            importance_scores.append(importance_score)
            nodes.append(node)
            server_types.append(type)
        data['Node'] = nodes
        data['Server Type'] = server_types
        data['Importance Score'] = importance_scores
        return data

    def create_security_components_data(self, host, data):
        '''
        Creates the data of the "Security Components Features" (features regarding the security components of the node):
        Security Components Features: Security Components (list), Security Weight, Spread Weight

        :param host: the initial host that the data is generated for
        :param data: data - dataframe with the addition of Security Components Features
        :return: data - dataframe with the addition of Security Components Features
        '''
        # Network Security Components - Security Score, Security Weight, Spread Weight:
        hosts_amount = host.getHostsAmount()
        Security_components = []
        Security_Scores = []
        Security_Weights = []
        Spread_Weights = []
        for i in range(hosts_amount):
            num_sec_components = random.randint(0, 6)
            node_sec_components, node_sec_weight = self.get_components(num_sec_components)
            node_spread_weight = 1 - node_sec_weight

            Security_components.append(node_sec_components)
            Security_Weights.append(node_sec_weight)
            Spread_Weights.append(node_spread_weight)

        # host.setSecurityComponents(Security_components)
        # print('sec comps: ', host.getSecurityComponents())

        data['Security Components'] = Security_components
        data['Security Weight'] = Security_Weights
        data['Spread Weight'] = Spread_Weights

        return data

    def create_data_per_initial_host(self, host):
        '''
        create the sub-network (network A / B) data per initial host - Network Features, Node Features, Security Components Features

        :param host: the initial hostNode for the network creation
        :return: data - dataframe with all added features of the sub-network
        '''
        data = self.create_network_data(host)
        data = self.create_nodes_data(host, data)
        data = self.create_security_components_data(host, data)

        return data

    def create_connectors_hosts(self, hostNodes):
        '''
        Set connectors between the networks - hosts from the 2 networks (A,B) that are connected to each other
        Connectors amount = 15

        :param hostNodes: the list of Nodes (Hosts and Servers). contains hostNode objects.
        :return: hostNodes - the list of Nodes (Hosts and Servers). contains hostNode objects.
        '''
        ips_A = []
        ips_B = []
        network_A = []
        network_B = []

        for node in hostNodes:
            if str(node.getNetworkAddress()) == '192.168.7.0':
                network_A.append(node)
                ips_A.append(node.getIP())
            if str(node.getNetworkAddress()) == '10.10.8.0':
                network_B.append(node)
                ips_B.append(node.getIP())
        seeds = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        for i in range(15):
            random.seed(seeds[i])
            # print('seed: ', i)
            index_A = random.randint(0, len(ips_A))
            index_B = random.randint(0, len(ips_B))
            # print('index_A: ', index_A, '\n' , 'index_B: ', index_B)
            hostConnector_A = network_A[index_A]
            hostConnector_B = network_B[index_B]
            hostConnector_A.setIsConnector(True)
            hostConnector_B.setIsConnector(True)

            ipConnector_A = ips_A[index_A]
            ipConnector_B = ips_B[index_B]
            hostConnector_A.setConnectorHost(otherHostIP=ipConnector_B)
            hostConnector_B.setConnectorHost(otherHostIP=ipConnector_A)

            # print('ipConnector_A: ', ipConnector_A)
            # print('ipConnector_B: ', ipConnector_B)
            # print('hostConnector_A connected to: ', hostConnector_A.getConnectorHost())
            # print('hostConnector_B connected to: ', hostConnector_B.getConnectorHost())

        return hostNodes

    def create_sorted_df_for_netX(self, df):
        '''
        1) Sorts the dataframe for networkX by the order: hosts from network A -> connectors of network A -> connectors of network B -> hosts from network B
        2) Re-index the dataframe

        :param df: the dataframe of the network data
        :return: the dataframe of the network data - sorted
        '''
        net_A_connectors = df[(df['Network'] == 'A') & df['Is Connector']]
        net_A = df[(df['Network'] == 'A') & (~df['Is Connector'])]
        net_B_connectors = df[(df['Network'] == 'B') & df['Is Connector']]
        net_B = df[(df['Network'] == 'B') & (~df['Is Connector'])]

        frames = [net_A, net_A_connectors, net_B_connectors, net_B]
        df_sorted = pd.concat(frames)
        df_sorted.index = np.arange(1, len(df_sorted) + 1)
        df_sorted.index.name = 'ID'
        return df_sorted

    def create_whole_network_data(self):
        '''
        creates the dataframe of the whole network data

        :return: df - dataframe of the network data
                 hostNodes - the list of Nodes (Hosts and Servers). contains hostNode objects.
        '''
        hostNodesDataFrames = []

        # Create data frame per each given Host Node
        for hostNode in self.__initialHostsList:
            hostNodesDataFrames.append(self.create_data_per_initial_host(hostNode))

        # Unify Host Nodes' data frames to a single DF
        df = pd.concat(hostNodesDataFrames, axis=0)

        # Dropping duplicate values from data frame
        df.drop_duplicates(subset=['IP'], keep='first', inplace=True)

        # Set data frame's index column
        df.index = np.arange(1, len(df) + 1)
        df.index.name = 'ID'

        # Init list of Nodes (Hosts and Servers)
        hostNodes = []

        # Instantiate HostNode object per each generated node within data frame
        for index, row in df.iterrows():
            node = HostNode(str(row['IP']), row['Subnet Mask Decimal'], row['Network'], row['Importance Score'],
                            row['Security Weight'], row['Security Components'], False, False)
            hostNodes.append(node)
            node.setSecurityLevel(row['Security Weight'])
            node.setImportanceScore(row['Importance Score'])

        # Create connectors between the networks (hosts from the 2 networks that are connected to each other)
        hostNodes = self.create_connectors_hosts(hostNodes)

        # Add column of Host Node objects
        df['HostNode'] = hostNodes

        # Add columns for the connectors hosts
        isConnectorList = []
        hostConnectorIPList = []
        connectedHostsAmountList = []
        for node in hostNodes:
            isConnectorList.append(node.getIsConnector())
            hostConnectorIPList.append(node.getConnectorHost())
            connectedHostsAmountList.append(node.getConnectedHostsAmount())
        df['Is Connector'] = isConnectorList
        df['Connected Hosts'] = hostConnectorIPList
        df['Connected Hosts Amount'] = connectedHostsAmountList  # Add column of connected hosts amount

        # Create total score for each node:
        totalScores = []
        for node in hostNodes:
            normalized_importance_score = self.normalize(node.getImportanceScore())
            spread_weight = 1 - node.getSecurityLevel()
            node_score = 0.4 * spread_weight + 0.6 * normalized_importance_score
            node.setTotalScore(node_score)
            totalScores.append(node_score)
        df['Total Score'] = totalScores

        # sort df for networkX + create new csv:
        df = self.create_sorted_df_for_netX(df)

        return df, hostNodes

    def create_weighted_directed_graph(self, df):
        '''
        Convert the network dataframe to a weighted directed graph:
        - creates the nodes in the graph
        - creates the edges in the graph - two directions between the couples of nodes:
            - edges between all the nodes in network A
            - edges between all the nodes in network B
            - edges between all the connectors nodes (from each connector to its couple)
        - creates the weights on edges - based on the total score of the node the edge is directed to

        :param df: the network dataframe
        :return: weighted directed graph G
        '''
        net_A_ids = df[(df['Network'] == 'A')].index   # including connectors
        net_B_ids = df[(df['Network'] == 'B')].index   # including connectors
        connectors_ids = df[df['Is Connector']].index  # connectors of networks A & B

        net_A_scores = df[(df['Network'] == 'A')]['Total Score']   # including connectors
        net_B_scores = df[(df['Network'] == 'B')]['Total Score']   # including connectors
        connectors_scores = df[df['Is Connector']]['Total Score']  # connectors of networks A & B

        # Network X:
        G = nx.DiGraph()  # Directed Graph

        # Add nodes:
        G.add_nodes_from(net_A_ids)
        G.add_nodes_from(connectors_ids)
        G.add_nodes_from(net_B_ids)

        # Network A: Add edges between nodes in network A - 2 directions - with weights:
        for i in net_A_ids:
            for j in net_A_ids:
                if (i != j):
                    # print('i: ',i , 'score of edge that goes into i: ', net_A_scores[i])
                    # print('j: ',j , 'score of edge that goes into j: ', net_A_scores[j])
                    G.add_edge(i, j, weight=net_A_scores[j])
                    G.add_edge(j, i, weight=net_A_scores[i])
                    # print( 'weight i->j: ', G.get_edge_data(i,j))
                    # print( 'weight j->i: ', G.get_edge_data(j,i))

        # Network B: Add edges between nodes in network B - 2 directions - with weights:
        for i in net_B_ids:
            for j in net_B_ids:
                if (i != j):
                    # print('i: ',i , 'score of edge that goes into i: ', net_B_scores[i])       # score of edge that goes into i
                    # print('j: ',j , 'score of edge that goes into j: ', net_B_scores[j])       # score of edge that goes into j
                    G.add_edge(i, j, weight=net_B_scores[j])
                    G.add_edge(j, i, weight=net_B_scores[i])
                    # print( 'weight i->j: ', G.get_edge_data(i,j))
                    # print( 'weight j->i: ', G.get_edge_data(j,i))

        # Connectors: Add edges between the connectors from both networks A & B:
        for i in connectors_ids:
            for j in connectors_ids:
                if (i != j):
                    # print('i: ',i , 'score of edge that goes into i: ', connectors_scores[i])       # score of edge that goes into i
                    # print('j: ',j , 'score of edge that goes into j: ', connectors_scores[j])       # score of edge that goes into j
                    G.add_edge(i, j, weight=connectors_scores[j])
                    G.add_edge(j, i, weight=connectors_scores[i])
                    # print( 'weight i->j: ', G.get_edge_data(i,j))
                    # print( 'weight j->i: ', G.get_edge_data(j,i))

        return G

    def draw_graph(self, G):
        '''
        Draw the graph G
        :param G: the network's graph
        :return: None.
        '''
        nx.draw(G, with_labels=True)
        plt.draw()
        plt.show()
