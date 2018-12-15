#!/Applications/anaconda/bin/python

from matplotlib import pyplot as plt
import numpy as np
import urllib
import random
import math
import timeit
import time

# Set timeout for CodeSkulptor if necessary
#import codeskulptor
#codeskulptor.set_timeout(20)


###################################
# Code for loading citation graph
CITATION_URL = "http://storage.googleapis.com/codeskulptor-alg/alg_phys-cite.txt"

def load_graph(graph_url):
    """
    Function that loads a graph given the URL
    for a text representation of the graph
    
    Returns a dictionary that models a graph
    """
    graph_file = urllib.request.urlopen(graph_url)
    # graph_text = graph_file.read()
    graph_text = graph_file.read().decode("utf-8")
    #print(graph_text)
    graph_lines = graph_text.split('\n')
    graph_lines = graph_lines[ : -1]
    
    print ("Loaded graph with", len(graph_lines), "nodes")
    
    answer_graph = {}
    for line in graph_lines:
        neighbors = line.split(' ')
        node = int(neighbors[0])
        answer_graph[node] = set([])
        for neighbor in neighbors[1 : -1]:
            answer_graph[node].add(int(neighbor))

    return answer_graph

#####################################################################
def compute_in_degrees(digraph):

    """Takes a directed graph digraph (represented as a dictionary) and
    computes the in-degrees for the nodes in the graph. The function
    returns a dictionary with the same set of keys (nodes) as 𝚍𝚒𝚐𝚛𝚊𝚙𝚑
    whose corresponding values are the number of edges whose head matches
    a particular node.
    """
    result = {}

    for source in digraph.keys():
        result.update({source: 0})

    for destination_list in digraph.values():
        for destination in destination_list:
            result.update({destination: (result.get(destination) + 1)})

    return result

#####################################################################
def in_degree_distribution(digraph):

    """ Takes a directed graph digraph (represented as a dictionary) and computes
        the unnormalized distribution of the in-degrees of the graph. The function
        returns a dictionary whose keys correspond to in-degrees of nodes in the graph.
        The value associated with each particular in-degree is the number of nodes with that
        in-degree. In-degrees with no corresponding nodes in the graph are not included in
        the dictionary.
    """
    result = {}
    in_degrees = compute_in_degrees(digraph)

    for degree in in_degrees.values():
        frequency = result.get(degree, -1)
        if frequency == -1:
            result.update({degree: 1})
        else:
            result.update({degree: frequency + 1})

    return result

#####################################################################
def normalize_in_degree(in_degree_distr):
    total_sum = 0
    for value in in_degree_distr.values():
        total_sum += value

    result = {}
    for key in in_degree_distr.keys():
        result.update({key: in_degree_distr.get(key)/total_sum})
        
    return result
    

#####################################################################
def plot_normalized_distribution( normalized_distr ):

    xaxis = []
    for value in normalized_distr.keys():
        xaxis.append(value)

    yaxis = []    
    for value in normalized_distr.values():
        yaxis.append(value)

    plt.plot(xaxis, yaxis, 'ro')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('In-Degree Distribution')
    plt.ylabel('Frequecy Represented as a Fraction')
    plt.title('Plot of the DPA Generated Graph')
    # plt.axis([0, 6, 0, 20])
    plt.show()
    

#####################################################################
def generate_graph(num_of_nodes, probability):

    if (num_of_nodes <= 0):
        return {}
    else:
        graph = {}
        for source in range(0, num_of_nodes):
            nodes = []
            for destination in range(0, num_of_nodes):
                if source != destination:
                    random_a = random.random()
                    if (random_a < probability):
                        nodes.append(destination)
            graph.update({source: nodes})

        return graph
    
####################################################################
def make_complete_graph(num_of_nodes):

    """ Takes the number of nodes num_of_nodes and returns a dictionary
    corresponding to a complete directed graph with the specified number
    of nodes. A complete graph contains all possible edges subject to
    the restriction that self-loops are not allowed.
    """

    if num_of_nodes <= 0:
        return {}
    else:
        graph = {}
        for source in range(0, num_of_nodes):
            nodes = []
            for destination in range(0, num_of_nodes):
                if source != destination:
                    nodes.append(destination)
            graph.update({source: nodes})
        return graph

#####################################################################
class DPATrial:
    """
    Simple class to encapsulate optimized trials for DPA algorithm
    
    Maintains a list of node numbers with multiple instances of each number.
    The number of instances of each node number are
    in the same proportion as the desired probabilities
    
    Uses random.choice() to select a node number from this list for each trial.
    """
    
    def __init__(self, num_nodes):
        """
        Initialize a DPATrial object corresponding to a 
        complete graph with num_nodes nodes
        
        Note the initial list of node numbers has num_nodes copies of
        each node number
        """
        self._num_nodes = num_nodes
        self._node_numbers = [node for node in range(num_nodes) for dummy_idx in range(num_nodes)]


    def run_trial(self, num_nodes):
        """
        Conduct num_node trials using by applying random.choice()
        to the list of node numbers
        
        Updates the list of node numbers so that the number of instances of
        each node number is in the same ratio as the desired probabilities
        
        Returns:
        Set of nodes
        """
        
        # compute the neighbors for the newly-created node
        new_node_neighbors = set()
        for dummy_idx in range(num_nodes):
            new_node_neighbors.add(random.choice(self._node_numbers))
        
        # update the list of node numbers so that each node number 
        # appears in the correct ratio
        self._node_numbers.append(self._num_nodes)
        self._node_numbers.extend(list(new_node_neighbors))
        
        #update the number of nodes
        self._num_nodes += 1
        return new_node_neighbors
        
#######################################################
def dpa(final_size, seed_size):
    """
    This algorithm starts with a complete (fully connected) graph) and 
    incrementally adds nodes and their adjacency lists. The seed size is the 
    number of nodes for the complete graph. The final size is the total number
    of nodes in the final graph.
    
    final size - seed size = number of nodes to be added with the DPA trials
    algorithm. When the DPA trial object is created, pass in the total number
    nodes desired in the final graph. Each time the trial function is called,
    passed in the node ID and the trial function returns the adjacency list 
    for that node.
    """
    if not seed_size <= final_size:
        print('Error. Seed size must be smaller than or equal to final size.')
        exit(-1)
         
    graph = make_complete_graph(seed_size)
    dpa_helper = DPATrial(seed_size)
    # The node IDs for the seed graph range from 0 - seed_size -1.
    # Therefore the the number seed_size is really the ID for the first
    # node to be generated by DPA trial 
    for node in range(seed_size, final_size):
        adj_list = dpa_helper.run_trial(seed_size)
        graph[node] = adj_list

    return graph
               
                             
#####################################################################
#graph = load_graph(CITATION_URL)
#graph = generate_graph(2000, .7)

# graph = dpa(28000, 13)
# in_deg_dis = in_degree_distribution( graph )
# norm_deg_distr = normalize_in_degree(in_deg_dis)
#plot_normalized_distribution( norm_deg_distr )

