#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
PAA 2 algorithmic thinking course
'''

import random
import time
import math
from UPATrial import UPATrial as upa


def copy_graph(graph):
    """
    Make a copy of a graph
    """
    new_graph = {}
    for node in graph:
        new_graph[node] = set(graph[node])
    return new_graph


def delete_node(ugraph, node):
    """
    Delete a node from an undirected graph
    """
    neighbors = ugraph[node]
    ugraph.pop(node)
    for neighbor in neighbors:
        ugraph[neighbor].remove(node)


def targeted_order(ugraph):
    """
    Compute a targeted attack order consisting
    of nodes of maximal degree
    
    Returns:
    A list of nodes
    """
    # copy the graph
    new_graph = copy_graph(ugraph)
    order = []    
    while len(new_graph) > 0:
        max_degree = -1
        for node in new_graph:
            if len(new_graph[node]) > max_degree:
                max_degree = len(new_graph[node])
                max_degree_node = node
        neighbors = new_graph[max_degree_node]
        new_graph.pop(max_degree_node)
        for neighbor in neighbors:
            new_graph[neighbor].remove(max_degree_node)
        order.append(max_degree_node)
    return order


def load_graph(graph_file):
    """
    Function that loads a graph for a text representation of the graph
    
    Returns a dictionary that models a graph
    """
    graph_file = open(graph_file)
    graph_text = graph_file.read()
    graph_lines = graph_text.split('\n')
    graph_lines = graph_lines[:-1]
    answer_graph = {}
    for line in graph_lines:
        neighbors = line.split(' ')
        node = int(neighbors[0])
        answer_graph[node] = set([])
        for neighbor in neighbors[1 : -1]:
            answer_graph[node].add(int(neighbor))
    return answer_graph


def undirected_ER_graph_generator(num_nodes, probability):
    '''
    generate a undirected ER graph
    input: number of notdes and the probability to generate a edge
    output: dictionary representating a generated graph
    '''
    graph = {node:set() for node in range(num_nodes)}
    for node in graph:
        for target in range(num_nodes):
            if target != node:
                p = random.random()
                if p <= probability:
                    graph[node].add(target)
                    graph[target].add(node)
    return graph


def number_of_edges(ugraph):
    num_of_edges = 0
    for edge in ugraph.values():
        num_of_edges += len(edge)
    return num_of_edges/2


def upa_graph(total_number_of_nodes, initial_number_of_nodes, num_edges):
    """ 
    generate an undirected preferencial attachment graph
    """
    ugraph = undirected_ER_graph_generator(initial_number_of_nodes, 1)
    random_connect = upa(initial_number_of_nodes)
    for new_node in range(initial_number_of_nodes, total_number_of_nodes):
        new_edges = random_connect.run_trial(num_edges)
        ugraph[new_node] = new_edges
        for old_node in new_edges:
            ugraph[old_node].add(new_node)
    return ugraph


def bfs_visited(ugraph, start_node):
    '''
    input: undirected graph represented as adj list, a starting node
    output: a set containing the nodes connected to the starting node
    '''
    queue = [start_node]
    visited = [start_node]
    while queue != []:
        node = queue.pop()
        for neighbor in ugraph[node]:
            if neighbor not in visited:
                visited.append(neighbor)
                queue.append(neighbor)
    return set(visited)


def cc_visited(ugraph):
    '''
    input: undirected graph represented as adj list
    output: list of sets, each set represents a connected component
    '''
    remaining_nodes = ugraph.keys()
    connected_components = []
    while remaining_nodes != []:
        node = remaining_nodes[random.randint(0,len(remaining_nodes)-1)]
        component = bfs_visited(ugraph, node)
        connected_components.append(component)
        remaining_nodes = [remain for remain in remaining_nodes if remain not in component]
    return connected_components


def largest_cc_size(ugraph):
    '''
    input: undirected graph represented as adj list
    output: integer representing the size of the largest connected component
    '''
    connected_components = cc_visited(ugraph)
    length = [len(component) for component in connected_components]
    if length == []:
        return 0
    return max([len(component) for component in connected_components])


def compute_resilience(ugraph, attack_order):
    '''
    input: undirected graph represented as adj list, list of nodes to be removed from graph
    output, list of the size of the largest connected component in the remaining graph
    '''
    new_graph = copy_graph(ugraph)
    remaining_largest_cc_size = [largest_cc_size(new_graph)]
    for node in attack_order:
        new_graph.pop(node)
        for edge in new_graph.values():
            edge.discard(node)
        remaining_largest_cc_size.append(largest_cc_size(new_graph))
    return remaining_largest_cc_size


def random_order(graph):
    '''
    generate random attack order
    '''
    return random.sample(graph.keys(),len(graph.keys()))


def write_resilience(resilience, filename):
    '''
    function write the resiliences of grpahs to a csv file
    '''
    dis_file = open(filename, 'w')
    for i in resilience:
        dis_file.write(str(i)+'\n')
    dis_file.close()


def test_resilient(resilience):
    '''
    test if the graph is resilient, that is:
    if the size of the largest cc is within 25% of its original value,
        after 20% nodes removed
    '''
    return resilience[int((len(resilience)-1)*0.2)]>((len(resilience)-1)-(len(resilience)-1)*0.25)


def fast_targeted_order(ugraph):
    '''
    genrate a attack order which always attack the currently 
        'richest' node in the graph for each step
    '''
    new_graph = copy_graph(ugraph)
    degree_sets = [set() for i in range(len(new_graph))]
    for k in range(len(new_graph)):
        degree_sets[len(new_graph[k])].add(k)
    L = []
    for k in range(len(new_graph)):
        iter = len(new_graph) - k - 1
        while degree_sets[iter] != set():
            u = degree_sets[iter].pop()
            for neighbor in new_graph[u]:
                d = len(new_graph[neighbor])
                degree_sets[d].remove(neighbor)
                degree_sets[d-1].add(neighbor)
            L.append(u)
            delete_node(new_graph, u)
    return L


def test_order_generator():
    fast_time = []
    slow_time = []
    for i in range(10,1000,10):
        new_graph = upa_graph(i, 5, 5)
        new_graph2 = copy_graph(new_graph)
        start = time.clock()
        fast_targeted_order(new_graph)
        run_time = time.clock() - start
        fast_time.append(run_time)
        start = time.clock()
        targeted_order(new_graph2)
        run_time = time.clock() - start
        slow_time.append(run_time)
    return slow_time, fast_time


if __name__ == "__main__":
    # load the computer network graph
    print "Question 1"
    graph_file = 'alg_rf7.txt'
    cngraph = load_graph(graph_file)
    # calculating the number of nodes and edges in the computer network graph
    number_of_original_nodes = len(cngraph)
    number_of_original_edges = number_of_edges(cngraph)
    # calculating the probability to generate an er-graph with approximately the same number of edges
    probability = float(number_of_original_edges)/(number_of_original_nodes*(number_of_original_nodes-1))
    # generating the er-graph
    ergraph = undirected_ER_graph_generator(number_of_original_nodes,probability)
    # generating a preferential attachment graph with similar number of edges
    upagraph = upa_graph(number_of_original_nodes, 2, 2)
    # attack each network and generate the resiliences
    attack_order = random_order(cngraph)
    resi_graph = compute_resilience(cngraph, attack_order)
    erattack_order = random_order(ergraph)
    resi_ergraph = compute_resilience(ergraph, erattack_order)
    upaattack_order = random_order(upagraph)
    resi_upagraph = compute_resilience(upagraph, upaattack_order)
    write_resilience(resi_graph,'graph_resiliences.txt')
    write_resilience(resi_ergraph,'ergraph_resiliences.txt')
    write_resilience(resi_upagraph,'upagraph_resiliences.txt')
    # finish the ploting in R, plot1.R

    # question 2
    print "Question 2"
    print test_resilient(resi_graph)
    print test_resilient(resi_ergraph)
    print test_resilient(resi_upagraph)

    # question 3
    print "Question 3"
    slow_time, fast_time = test_order_generator()
    write_resilience(slow_time, 'time.txt')
    write_resilience(fast_time, 'fast_time.txt')

    # question 4
    print "Question 4"
    targeted_attack_order = targeted_order(cngraph)
    targeted_resi_graph = compute_resilience(cngraph, targeted_attack_order)
    targeted_erattack_order = targeted_order(ergraph)
    targeted_resi_ergraph = compute_resilience(ergraph, targeted_erattack_order)
    targeted_upaattack_order = targeted_order(upagraph)
    targeted_resi_upagraph = compute_resilience(upagraph, targeted_upaattack_order)
    write_resilience(targeted_resi_graph,'targeted_graph_resiliences.txt')
    write_resilience(targeted_resi_ergraph,'targeted_ergraph_resiliences.txt')
    write_resilience(targeted_resi_upagraph,'targeted_upagraph_resiliences.txt')

    # question 5
    print "Question 5"
    print test_resilient(targeted_resi_graph)
    print test_resilient(targeted_resi_ergraph)
    print test_resilient(targeted_resi_upagraph)
