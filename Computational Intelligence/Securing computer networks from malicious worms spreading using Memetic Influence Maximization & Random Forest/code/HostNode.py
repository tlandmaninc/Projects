import ipaddress as ipa
import numpy as np
import pandas as pd
import random


class HostNode:
    __idCounter = 0
    __infectedHosts = list()
    __connectorHosts = list()

    def __init__(self, ip, subnetMask, network, importanceScore, securityLevel, securityComponents, isConnector,
                 isInfected):
        HostNode.__idCounter += 1
        self.__id = HostNode.__idCounter
        self.__ip = ipa.ip_address(ip)
        self.__ipNetwork = ipa.ip_network(ip + '/' + str(subnetMask), strict=False)
        self.__subnetMask = self.__ipNetwork.with_netmask.split('/')[1]
        self.__subnetDecimal = subnetMask
        self.__networkAddress = self.__ipNetwork.network_address
        self.__broadcastAddress = self.__ipNetwork.broadcast_address
        # self.__hostsAmount = int(self.__broadcastAddress) - int(self.__networkAddress) - 2
        self.__hostsAmount = int(self.__broadcastAddress) - int(self.__networkAddress) - 1
        self.__importanceScore = importanceScore
        self.__securityLevel = securityLevel
        self.__isInfected = isInfected
        self.__securityComponents = securityComponents
        self.__isConnector = isConnector
        self.__hostConnectedIP = None
        self.__totalScore = 0
        self.__networkType = network

    # Getters:
    def getID(self):
        return self.__id

    def getIP(self):
        return self.__ip

    def getSubnetMask(self):
        return self.__subnetMask

    def getSubnetMaskDecimal(self):
        return self.__subnetDecimal

    def getNetworkAddress(self):
        return self.__networkAddress

    def getBroadcastAddress(self):
        return self.__broadcastAddress

    def getHostsAmount(self):
        return self.__hostsAmount

    def getConnectedHostsAmount(self):
        if self.__isConnector:
            return self.__hostsAmount + 1
        else:
            return self.__hostsAmount

    def getImportanceScore(self):
        return self.__importanceScore

    def getSecurityLevel(self):
        return self.__securityLevel

    def getIsInfected(self):
        return self.__isInfected

    def getSecurityComponents(self):
        return self.__securityComponents

    def getNetworkType(self):
        return self.__networkType

    def getTotalScore(self):
        return self.__totalScore

    def getIsConnector(self):
        return self.__isConnector

    def getConnectorHost(self):
        return self.__hostConnectedIP

    def getNetworkHosts(self):
        network = list(self.__ipNetwork.hosts())
        return network

    def getConnectedHosts(self):
        '''
        Returns an iterator over the usable hosts in the network.
        The usable hosts are all the IP addresses that belong to the network,
        except the network address itself and the network broadcast address.
        '''
        network = list(self.__ipNetwork.hosts())
        network.remove(self.__ip)

        if self.__isConnector == True:
            network.append(self.__hostConnectedIP)
        # network = np.array(self.__ipNetwork.hosts())
        # network = np.delete(network, np.where(network == self.__ip))
        return network

    # Get Lists:
    def getInfectedHosts(self):
        return self.__infectedHosts

    def getListConnectorHosts(self):
        return self.__connectorHosts

    # Setters:
    def setIsConnector(self, newStatus):
        self.__isConnector = newStatus
        self.__connectorHosts.append(self.__ip)

    def setIsInfected(self, newStatus):
        self.__isInfected = newStatus
        self.__infectedHosts.append(self.__ip)

    def setSecurityComponents(self, securityComponents):
        self.__securityComponents = securityComponents

    def setSecurityLevel(self, securityLevel):
        self.__securityLevel = securityLevel

    def setImportanceScore(self, importanceScore):
        self.__importanceScore = importanceScore

    def setConnectedHosts(self, newConnectedHosts):
        self.__connectedHosts = newConnectedHosts

    def setConnectorHost(self, otherHostIP):
        self.__hostConnectedIP = otherHostIP

    def setNetworkType(self, network):
        self.__networkType = network

    def setTotalScore(self, totalScore):
        self.__totalScore = totalScore

    def __str__(self):
        return "\n####################################################################################################\n" \
               "'Host {0}'\n - IP: {1}\n - Subnet Mask: {2}\n - Network Address: {3}\n - Broadcast Address: {4}\n" \
               " - Connected Hosts: {5}\n - Is Connector: {6}\n - Importance Score: {7}\n - Security Level: {8}\n - Security Components: {9}\n" \
               "####################################################################################################".format(
            self.__id, self.__ip, self.__subnetMask, self.__networkAddress,
            self.__broadcastAddress, self.getConnectedHostsAmount(), self.__isConnector, self.__importanceScore,
            self.__securityLevel, self.__securityComponents)

    def print(self):
        print("\n####################################################################################################\n" \
               "'Host {0}'\n - IP: {1}\n - Subnet Mask: {2}\n - Network Address: {3}\n - Broadcast Address: {4}\n" \
               " - Connected Hosts: {5}\n - Is Connector: {6}\n - Importance Score: {7}\n - Security Level: {8}\n - Security Components: {9}\n" \
               "####################################################################################################".format(
            self.__id, self.__ip, self.__subnetMask, self.__networkAddress,
            self.__broadcastAddress, self.getConnectedHostsAmount(), self.__isConnector, self.__importanceScore,
            self.__securityLevel, self.__securityComponents))
    # def __repr__(self):
    #     return self.__str__()
