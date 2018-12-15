#!/Applications/anaconda/bin/python


from collections import deque


def bfs_visited(ugraph, start_node):
    """

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

    """
    # This is a list, these are the nodes in the graph
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
    removal of the first k nodes in 𝚊𝚝𝚝𝚊𝚌𝚔_𝚘𝚛𝚍𝚎𝚛. The first entry (indexed by zero) 
    is the size of the largest connected component in the original graph.


    def removekey(d, key):
    r = dict(d)
    del r[key]
    return r

    """

    # create copy of the input graph
    graph_copy = dict(ugraph)
    result_list = []
    result_list.append(largest_cc_size(graph_copy))
    
    for node in attack_order:
        curr_adj_list = graph_copy.get(node)
        #remove the node from the dict
        del graph_copy[node]
        # remove the current node from the rest
        for adj_node in curr_adj_list:
            adj_set = graph_copy.get(adj_node)
            adj_set.remove(node)

        result_list.append( largest_cc_size(graph_copy) )

    return result_list
         
    
    


#############################################################

EX_GRAPH0 = {0: set([1, 2]), 1: set([]), 2: set([])}
EX_GRAPH1 = {0: set([1, 4, 5]), 1: set([2, 6]), 2: set([3]), 3: set([0]), 4: set([1]), \
             5: set([2]), 6: set([])}
EX_GRAPH2 = {0: set([1, 4, 5]), 1: set([2, 6]), 2: set([3, 7]), 3: set([7]), 4: set([1]), \
             5: set([2]), 6: set([]), 7: set([3]), 8: set([1, 2]), 9: set([0, 3, 4, 5, 6, 7])}

EX_GRAPH4 = {0: set([]), 1: set([]), 2: set([])}

GRAPH = {0: set([1]), 1: set([]), 2: set([3]), 3: set([]), 4: set([1]) }

# print( bfs_visited(EX_GRAPH2, 5) )
#print( cc_visited(GRAPH) )

#print( largest_cc_size( GRAPH ))

my_dict = {1: set([3,4,5]), 2: set([6,7,8]) }
print( my_dict )
one = my_dict.get(1)
one.remove(3)
print( one )
#del my_dict[1]
print( my_dict )
