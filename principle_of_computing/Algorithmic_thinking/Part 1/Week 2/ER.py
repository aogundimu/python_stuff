#!/Applications/anaconda/bin/python

import random

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
            graph.update({source: set(nodes)})

        return graph

#################

graph = generate_graph(20, .5)

print(graph)

                    
