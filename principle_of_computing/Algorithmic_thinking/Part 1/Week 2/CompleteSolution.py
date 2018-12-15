import urllib2
import matplotlib.pyplot as plt
import pickle
import random

# Set timeout for CodeSkulptor if necessary
# import codeskulptor
# codeskulptor.set_timeout(20)

# Citation graph for 27,770 high energy physics theory papers. This graph has 352,807 edges.
CITATION_URL = "http://storage.googleapis.com/codeskulptor-alg/alg_phys-cite.txt"
RANDOM_SEED = 1

# Initialize random generator for repeatable results
random.seed(RANDOM_SEED)

def plot(distribution_dict, title=''):
    coords = dict_to_lists(distribution_dict)
    plt.loglog(coords[0], coords[1], 'ro')
    plt.ylabel('Frequency')
    plt.xlabel('In-degree')
    plt.grid(True)
    plt.title(title)
    plt.show()
    
def dict_to_lists(input_dict):
    '''
    Input is a dictionary whose keys are comparable.
    The function returns a list with two elements.
    The first element is the keys of the dictionary, sorted
    in ascending order. The second element are the the corresponding
    values. The values are normalized such that sum of all values equals 1.
    '''
    keys = sorted(input_dict.keys())
    values = [input_dict[key] for key in keys]
    return [normalize(keys), normalize(values)]   
    
def average(in_list):
    '''
    Return average of a list of values
    '''
    length = len(in_list)
    return float(sum(in_list))/length if length > 0 else float('nan')
    
def normalize(in_list):
    '''
    Given a list of numbers, normalize the values in the list so 
    values in the list sum to 1. 
`    '''
    if len(in_list) < 1:
        return float('nan')
    divisor = float(sum(in_list))
    return [element / divisor for element in in_list] 

def freq_count(in_list): 
    '''
    Given a list of objects, return a dictionary whose keys 
    are objects and whose values are the number of times those
    objects appeared in the list
    '''
    # Get the items whose frequency should be counted.
    occurrences = set(in_list)
    # Remove zero occurrences from the in_list 
#     occurrences = [each for each in occurrences if each > 0]
    # Create empty histogram object
    histogram = {}
    # Create dictionary where the values are the frequency counts of
    # each element in the list
    for element in occurrences:
        histogram[element] = in_list.count(element)
    return histogram

def save_object(filename, python_obj):
    '''
    Save a python object to a file
    '''
    with open(filename, 'wb') as file_obj:
        pickle.dump(python_obj, file_obj, pickle.HIGHEST_PROTOCOL) 
        
def load_object(filename):   
    '''
    Load a python object from a file
    '''
    _loaded_obj = None
    with open(filename, 'rb') as file_obj:
        _loaded_obj = pickle.load(file_obj)
    
    return _loaded_obj

def load_graph(graph_url):
    """
    Function that loads a graph given the URL
    for a text representation of the graph
    
    Returns a dictionary that models a graph
    """
    graph_file = urllib2.urlopen(graph_url)
    graph_text = graph_file.read()
    graph_lines = graph_text.split('\n')
    graph_lines = graph_lines[ :-1]
#     graph_lines = graph_lines[ 1:100]
    
    print "Loaded graph with", len(graph_lines), "nodes"
    
    answer_graph = {}
    for line in graph_lines:
        neighbors = line.split(' ')
        node = int(neighbors[0])
        answer_graph[node] = set([])
        for neighbor in neighbors[1 :-1]:
            answer_graph[node].add(int(neighbor))

    return answer_graph

'''    
Homework 1, you saw Algorithm ER for generating random graphs and reasoned 
analytically about the properties of the ER graphs it generates. Consider 
the simple modification of the algorithm to generate random directed graphs:
For every pair of nodes, i and j, the modified algorithm considers the pair
twice, once to add an edge (i,j) with probability p, and another to add an
edge (j,i) with probability p. 
For this question, your task is to consider 
the shape of the in-degree distribution for an ER graph and compare its shape
to that of the physics citation graph. In the homework, we considered the 
probability of a specific in-degree, k.  Now, we are interested in the 
in-degree distribution for the entire ER graph. To determine the shape of
this distribution, you are welcome to compute several examples of in-degree
distributions or determine the shape mathematically.
Once you have determined the shape of the in-degree distributions for ER 
graphs, compare the shape of this distribution to the shape of the in-degree
distribution for the citation graph. When answering this question, make sure
to address the following points:
1. Is the expected in-degree the same for every node in an ER graph? Please 
answer yes or no and include a short explanation for your answer.
ANSWER: The short answer is yes, at least in my implementation of ER. In my 
implementation, the expected in-degree of any node is n*p. 
2. What does the in-degree distribution for an ER graph look like? You may 
either provide a plot (linear or log/log) of the degree distribution for a 
small value of n or a short written description of the shape of the 
distribution. 
ANSWER: The normalized log-log plot of the in-degree distribution for an ER
graph look like a binomial distribution where the number of nodes is the 
number of trials and the probability of an edge, p, is the probability for
success for a trial.   
3. Does the shape of the in-degree distribution plot for ER look
similar to the shape of the in-degree distribution for the citation graph? 
Provide a short explanation of the similarities or differences. Focus on 
comparing the shape of the two plots.
ANSWER: The shape of the in-degree distribution plot for ER does not 
look like the same plot for the citation graph. The shape of the in-degree distribution
for the citation graph looks like logarithmic decay. The graph starts 
in the upper-left of the plot, where low in-degrees are common. The plot
heads toward the lower-right of the graph where higher in-degree values
become rare. At very low frequency, the number of in-degrees starts to
spread or smear across the plot-- there is more variability of in-degree
when frequency is low.
In contrast, the shape of the ER plot looks like a statistical distribution.
Not surprisingly, the ER plot is based on a parametric distribution and it
looks very. It also looks synthetic. The plot of the citation graph is
more complicated and harder to describe and that makes feel like 
real-world data.
'''

class Graph:
    def __init__(self, in_dictionary={}):
        if type(in_dictionary) != type({}):
            print('Cannot create graph from non-dictionary type. Exiting.')
            exit(-1) 
        #TODO: Check that all the node IDs are consecutive integers
        #whose min value is 0. 
        self._adjacency_dict = in_dictionary
     
    def __setitem__(self, node, edges):
        #TODO: Throw error is edges is not a set
        self._adjacency_dict[node] = edges.copy()
        return self
       
    def __getitem__(self,node):      
        return self._adjacency_dict[node].copy()
        
    def num_nodes(self):
        return len(self._adjacency_dict)
        
    def out_degrees(self):
        '''
        Return a dictionary whose keys are node IDs and whose values is the 
        out degree of that node
        '''
        output = {}
        for node, edges in self._adjacency_dict.iteritems():
            output[node] = len(edges)
        return output
    
    def in_degrees(self): 
        '''
        The function returns a dictionary whose keys are node IDs 
        whose corresponding value is the number of edges pointed to that node.
        '''
        freq_dict = {}      
        # Visit every node and count the number of times it appears 
        # in the list of tail nodes to determine its degree.
        all_in_nodes = self._flat_adj_lists()
        for node in self.all_nodes():
            freq_dict[node] = all_in_nodes.count(node)  
        return freq_dict       

    def in_degree_distribution(self):
        '''
         Computes the unnormalized distribution of the in-degrees 
         of the graph. The function returns a dictionary whose keys
         correspond to in-degrees of nodes in the graph. The value 
         associated with each in-degree is the number of nodes 
         with that in-degree. In-degrees with no corresponding instances in 
         the graph are not included in the dictionary.
         '''         
        #Pass in the frequencies to the frequency counter
        return freq_count(self.in_degrees().values())  
        
    def head_nodes(self):
        #Return set.
        return set(self._adjacency_dict.keys())
    
    def _flat_adj_lists(self):
        #Get all adjacency lists. It will be a list-of-lists.
        all_adj_lists = self._adjacency_dict.values()    
        # Concatenate the tail nodes of every adjacency list
        return [node for adj_list in all_adj_lists for node in adj_list]

    def tail_nodes(self):
        #The the unique set of nodes across all the adjacency lists
        #is the set of all tail nodes in the graph
        return set(self._flat_adj_lists())

    def all_nodes(self):
        #Convert to set because some head nodes will also be tail nodes
        return self.head_nodes().union(self.tail_nodes()) 
    
    def avg_out_degree(self):
        return average(self.out_degrees().values()) 

    def add_directed_edge(self, from_node, to_node):
        if from_node == to_node:
            print 'Error. Cannot add an edge between a node and itself'
            exit(-1)
        
        #Make sure both the from and to nodes are in the dictionary.
        #It is necessary that the from node be in the dictionary 
        #because it is a key and its value must be an set to which 
        #we can add edges.
        if from_node not in self._adjacency_dict:
            self[from_node] = set()
        #The to-node needs to be in the dictionary because when we 
        #compute the number of nodes in the graph or take an average
        #based on the number of nodes, it must be a key in the dictionary
        #or else it won't be counted. 
        if to_node not in self._adjacency_dict:
            self[to_node] = set()
            
        #Do not use __setitem__ here because it returns a copy
        self._adjacency_dict[from_node].add(to_node)

    def add_undirected_edge(self, node_i, node_j):
        self.add_directed_edge(node_i, node_j)
        self.add_directed_edge(node_j, node_i)

    def as_dictionary(self):
        return self._adjacency_dict.copy()

    def __str__(self):
        output = ''
        for key, value in self._adjacency_dict.iteritems():
            output += str(key) + ' -> ' + ','.join([str(node) for node in value]) + '\n'
        return output
      
    def is_valid(self):
        #TODO: Check that node IDs in the edge list do not exceed the max/min 
        #value of node IDs in the keys
        nodes = sorted(self._adjacency_dict.keys())
        cardinality = len(nodes)
        
        #Empty graph is considered valid
        if cardinality == 0:
            return True
        
        #First node ID must be zero
        #NOTE: This condition is not true for the citation graph
#         if nodes[0] != 0:
#             return False
        
        #Check for missing entries
        for idx in range(cardinality):
            if idx < cardinality-1:
                #The next node ID must be exactly one greater than the current node ID.
                #If not, there are gaps in the node IDs.
                #When building a graph, we can add edges one at a time. 
                #That means there is a chance of not adding a node that has no outbound edges
                if nodes[idx]+1 != nodes[idx+1]:
                    return False
        return True
      
def er(num_nodes, prob): 
    '''
    Algorithm 1: ER from Homework 1.
    Modified to create a digraph instead of an undirected graph.
    Input constraints:
        num_nodes  > 1
        0 <= prob < 1
    '''
    graph = Graph()
    nodes = range(0, num_nodes)
    for node_i in nodes:
        for node_j in nodes:
            if node_i != node_j:
                if  random.random() < prob:
                    graph.add_directed_edge(node_i, node_j)
    return graph

class DPATrial:
    """
    Simple class to encapsulate optimized trials for DPA algorithm
    
    Maintains a list of node numbers with multiple instances of each number.
    The number of instances of each node number are
    in the same proportion as the desired probabilities.
    
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
#         The list comprehension in the next statement makes more sense when 
#         written out as blocks:
#             for node in range(num_nodes):
#                 for idx in range(num_nodes):
#                     output_list.append(node)
#         Therefore if num_nodes = 3, the output list is [0, 0, 0, 1, 1, 1, 2, 2, 2]
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
        
        # Compute the neighbors for the newly-created node
        new_node_neighbors = set()

        #The smaller the value of the seed size, the fewer edges will be 
        #added in this loop. For small values of num_nodes, the list of 
        #values in _node_numbers is small and so the first few num_nodes
        #will pick small values because the seed_size is small. In turn,
        #all of those values then get added to _node_numbers, which makes 
        #it more likely they will be picked again. And when they are picked
        #again they are added to _node_number again, which increases the probability
        #they will be picked again, and so on. Coincidentally, this has the effect
        #that the graph is less connected because the _num_nodes list is dominated
        #by duplicate values and the algorithm does not add an edge unless it is unique.
        #is created, it can only be added to the graph once. 
        for dummy_idx in range(num_nodes):
            new_node_neighbors.add(random.choice(self._node_numbers))
        
        # update the list of node numbers so that each node number 
        # appears in the correct ratio
        self._node_numbers.append(self._num_nodes)
        
        # Older nodes have a better chance of being tail nodes because they 
        # appear in this list more frequently
        self._node_numbers.extend(list(new_node_neighbors))
        
        # update the number of nodes
        self._num_nodes += 1
        return new_node_neighbors


'''
Question 1
Load a provided citation graph for 27,770 high energy physics theory papers.
Compute the in-degree distribution for this citation graph. 
Normalize the distribution (make the values in the dictionary sum to one)
Create a log/log plot of the points in this normalized distribution. 
'''

# Create distribution plot for the citation graph
# citation_graph = load_graph(CITATION_URL)
# distribution_dict = in_degree_distribution(citation_graph)
# plot(distribution_dict)

# Create distribution plot for ER graphs
# print '----------------------'
# histogram = in_degree_distribution(er(600, 0.05))
# save_object('histrogram_5000_0.2.pickle', histogram)
# print 'DONE'
# plot(histogram, 'Frequency of in-degrees for ER, n=300, p=0.05')

def make_complete_graph(num_nodes):    
    """
    Takes the number of nodes num_nodes and returns a dictionary corresponding
    to a complete directed graph with the specified number of nodes. 
    A complete graph contains all possible edges subject to the restriction
    that self-loops are not allowed. The nodes of the graph should be 
    numbered 0 to num_nodes - 1 when num_nodes is positive. 
    Otherwise, the function returns a dictionary 
    corresponding to the empty graph.
    """
    
    # Create empty dictionary to hold the graph
    graph = {}
    
    # Double-loop over the nodes to create a complete graph
    for node1 in range(num_nodes):
        # Create an empty list to hold the target nodes
        graph[node1] = set([])
        for node2 in range(num_nodes):
            # Do not create an edge between a node and itself; no self-loops
            if (node1 != node2):
                graph[node1].add(node2)
    return graph
            
def dpa(final_size, seed_size):
    '''
    This algorithm starts with a complete (fully connected) graph) and 
    incrementally adds nodes and their adjacency lists. The seed size is the 
    number of nodes for the complete graph. The final size is the total number
    of nodes in the final graph.
    
    final size - seed size = number of nodes to be added with the DPA trials
    algorithm. When the DPA trial object is created, pass in the total number
    nodes desired in the final graph. Each time the trial function is called,
    passed in the node ID and the trial function returns the adjacency list 
    for that node.
    '''
    if not seed_size <= final_size:
        print 'Error. Seed size must be smaller than or equal to final size.'
        exit(-1)    
    graph = Graph(make_complete_graph(seed_size))
    dpa_helper = DPATrial(seed_size)
    # The node IDs for the seed graph range from 0 - seed_size -1.
    # Therefore the the number seed_size is really the ID for the first
    # node to be generated by DPA trial 
    for node in range(seed_size, final_size):
        adj_list = dpa_helper.run_trial(seed_size)
        graph[node] = adj_list
    return graph

# g = Graph(load_graph(CITATION_URL))
# save_object('citiation_dist.pickle', g.in_degree_distribution())
# g=dpa(27770,12)
# save_object('dpa_27770_12.pickle', g)
# dist = g.in_degree_distribution()
# save_object('dpa_dist.pickle', dist)
g=load_object('dpa_dist.pickle')
h=load_object('citation_dist.pickle')
gg = dict_to_lists(g)
hh = dict_to_lists(h)
plt.plot(gg[0], gg[1], 'bo', label='Citation graph')
plt.plot(hh[0], hh[1], 'ro', label='DPA n=200, m=12')
plt.ylabel('Frequency')
plt.xlabel('In-degree')
plt.grid(True)
plt.legend()
plt.title('dfsdf')
plt.show()


'''
To help you in your analysis, you should consider the following three phenomena:
The "six degrees of separation" phenomenon,
The "rich gets richer" phenomenon, and
The "Hierarchical structure of networks" phenomenon.
If you're not familiar with these phenomena, you can read about them by conducting a simple Google or Wikipedia search. Your task for this problem is to consider how one of these phenomena might explain the structure of the citation graph or, alternatively, how the citations patterns follow one of these phenomena. 
When answering this question, please include answers to the following:
Is the in-degree distribution for the DPA graph similar to that of the citation graph? Provide a short explanation of the similarities or differences. Focus on the various properties of the two plots as discussed in the class page on "Creating, formatting, and comparing plots".
Which one of the three social phenomena listed above mimics the behavior of the DPA process? Provide a short explanation for your answer.
Could one of these phenomena explain the structure of the physics citation graph? Provide a short explanation for your answer.
