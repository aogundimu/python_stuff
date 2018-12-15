"""
Week 4 Project. 
1. bfs_visited
2. cc_visited
3. largest_cc_size
4. compute_resilience
"""

from collections import deque

##############################################################
#
def bfs_visited(ugraph, start_node):
    
    """
    This is an implementation of the breadth_first_search algorithm.
    input: ugraph - the graph; start_node - the start node
    output: A set of visited nodes reached from the start_node
    """

    node_queue = deque()
    visited = set([start_node])
    node_queue.append(start_node)

    while len(node_queue) > 0:
        current_node = node_queue.popleft()
        neighbors = ugraph.get(current_node)
        for node in neighbors:
            if node not in visited:
                visited = visited.union( set([node]))
                node_queue.append(node)

    return visited

############################################################
#
def cc_visited(ugraph):   
    """
    This method determines all the connected components of a graph
    input: the undirected graph
    output:
    """

    remaining_nodes = set(ugraph.keys())
    connected_components = []

    while ( len(remaining_nodes) != 0):
        next_node = remaining_nodes.pop()
        visited_nodes = bfs_visited(ugraph, next_node)
        connected_components.append(visited_nodes)
        remaining_nodes.difference_update(visited_nodes)

    return connected_components

    

############################################################
#
def largest_cc_size(ugraph):
    """
    Function to compute the size of the largest connected components
    of a graph.
    """
    connected_components = cc_visited(ugraph)

    size = 0

    for con_comp in connected_components:
        new_size = len(con_comp)
        if new_size > size:
            size = new_size

    return size   

###############################################################
#
def compute_resilience(ugraph, attack_order):
    
    """
    Takes the undirected graph 𝚞𝚐𝚛𝚊𝚙𝚑, a list of nodes 𝚊𝚝𝚝𝚊𝚌𝚔_𝚘𝚛𝚍𝚎𝚛 and 
    iterates through the nodes in 𝚊𝚝𝚝𝚊𝚌𝚔_𝚘𝚛𝚍𝚎𝚛. For each node in the list, 
    the function removes the given node and its edges from the graph and 
    then computes the size of the largest connected component for the 
    resulting graph. The function should return a list whose k+1th entry 
    is the size of the largest connected component in the graph after the 
    removal of the first k nodes in 𝚊𝚝𝚝𝚊𝚌𝚔_𝚘𝚛𝚍𝚎𝚛. The first entry (indexed 
    by zero) is the size of the largest connected component in the original 
    graph.
    """

    # create copy of the input graph
    graph_copy = dict(ugraph)
    result_list = []
    result_list.append(largest_cc_size(graph_copy))
    
    for node in attack_order:
        curr_adj_list = graph_copy.get(node)
        del graph_copy[node]
        for adj_node in curr_adj_list:
            adj_set = graph_copy.get(adj_node)
            adj_set.remove(node)

        result_list.append( largest_cc_size(graph_copy) )

    return result_list
         
    
