"""This module contains methods implementing the algorithms for the following:
    1. Construction of a complete directed graph
    2. Calculating the in-degrees of the nodes in a graph
    3. Calculating the distribution of the in-degrees of a graph
"""

EX_GRAPH0 = {0: set([1, 2]), 1: set([]), 2: set([])}
EX_GRAPH1 = {0: set([1, 4, 5]), 1: set([2, 6]), 2: set([3]), 3: set([0]), 4: set([1]), 5: set([2]), 6: set([])}
EX_GRAPH2 = {0: set([1, 4, 5]), 1: set([2, 6]), 2: set([3, 7]), 3: set([7]), 4: set([1]), 5: set([2]), 6: set([]),
             7: set([3]), 8: set([1, 2]), 9: set([0, 3, 4, 5, 6, 7])}


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
            graph.update({source: set(nodes)})
        return graph


def compute_in_degrees(digraph):

    """
    Takes a directed graph digraph (represented as a dictionary) and
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


def in_degree_distribution(digraph):

    """ 
        Takes a directed graph digraph (represented as a dictionary) and computes
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
