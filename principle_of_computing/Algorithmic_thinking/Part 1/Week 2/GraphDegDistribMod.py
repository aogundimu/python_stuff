# This module contains methods implementing the algorithms for the following:
# 1. Construction of a complete directed graph
# 2. Calculating the in-degrees of the nodes in a graph
# 3. Calculating the distribution of the in-degrees of a graph

EX_GRAPH0 = {0: [1, 2], 1: [], 2: []}
EX_GRAPH1 = {0: [1, 4, 5], 1: [2, 6], 2: [3], 3: [0], 4: [1], 5: [2], 6: []}
EX_GRAPH2 = {0: [1, 4, 5], 1: [2, 6], 2: [3, 7], 3: [7], 4: [1], 5: [2], 6: [], 7: [3], 8: [1, 2],9: [0, 3, 4, 5, 6, 7]}

def make_complete_graph(num_of_nodes):

    # Takes the number of nodes num_nodes and returns a dictionary
    # corresponding to a complete directed graph with the specified number
    # of nodes. A complete graph contains all possible edges subject to
    # the restriction that self-loops are not allowed.

    if num_of_nodes <= 1:
        return {}
    else:
        graph = {}
        for x in range(0, num_of_nodes):
            nodes = []
            for y in range(0, num_of_nodes):
                if x != y:
                    nodes.append(y)
            graph.update({x: nodes})
        return graph


def compute_in_degrees(digraph):

    # Takes a directed graph digraph (represented as a dictionary) and
    # computes the in-degrees for the nodes in the graph. The function
    # returns a dictionary with the same set of keys (nodes) as digraph
    # whose corresponding values are the number of edges whose head matches
    # a particular node.

    result = {}

    for x in digraph.keys():
        result.update({x: 0})

    for y in digraph.values():
        for z in y:
            result.update({z: (result.get(z) + 1)})

    return result


def in_degree_distribution(digraph):

    # Takes a directed graph digraph (represented as a dictionary) and computes
    # the unnormalized distribution of the in-degrees of the graph. The function
    # returns a dictionary whose keys correspond to in-degrees of nodes in the graph.
    # The value associated with each particular in-degree is the number of nodes with that
    # in-degree. In-degrees with no corresponding nodes in the graph are not included in
    # the dictionary.

    result = {}
    in_degrees = compute_in_degrees(digraph)

    for degree in in_degrees.values():
        frequency  = result.get(degree, -1)
        if frequency == -1:
            result.update({degree: 1})
        else:
            result.update({degree: frequency+1})

    return result
