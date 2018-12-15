#!/Applications/anaconda/bin/python

import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
import random
import math
import timeit

######################################################################
# Code for ClosestPairStrip
def closest_pair_strip(cluster_list, horiz_center, half_width):
    """
    Helper function to compute the closest pair of clusters in a vertical strip

    Input: cluster_list is a list of clusters produced by fast_closest_pair
    horiz_center is the horizontal position of the strip's vertical center line
    half_width is the half the width of the strip (i.e; the maximum horizontal distance
    that a cluster can lie from the center line)

    Output: tuple of the form (dist, idx1, idx2) where the centers of the clusters
    cluster_list[idx1] and cluster_list[idx2] lie in the strip and have minimum distance dist.
    """


    # 2. sort the indexes according to the y values
    # cluster_list.sort(key = lambda cluster: cluster.vert_center())

    sorted_clusters = []
    for cluster in cluster_list:
        sorted_clusters.append(cluster.copy())

    sorted_clusters.sort(key = lambda cluster: cluster.vert_center())

    # 1. create the set
    indexes_s = []

    for index in range (0, len(sorted_clusters)):
        if abs(sorted_clusters[index].horiz_center() - horiz_center) < half_width:
            indexes_s.append(index)

    # 3. K length
    length_k = len(indexes_s)

    # 4.
    result = (float('inf'), -1, -1)

    index_u = 0
    while index_u <= length_k - 2:
        index_v = index_u + 1
        while index_v <= min(index_u+3, length_k -1):
            distance = cluster_list[indexes_s[index_u]].distance(cluster_list[indexes_s[index_v]])
            if distance < result[0]:
                result = (distance, indexes_s[index_u], indexes_s[index_v])
            index_v += 1
        index_u += 1

    return(result[0], cluster_list.index(sorted_clusters[result[1]]), \
           cluster_list.index(sorted_clusters[result[2]]))

####################################################################
def slow_closest_pair(cluster_list):
    """
    Compute the distance between the closest pair of clusters in a list (slow)

    Input: cluster_list is the list of clusters

    Output: tuple of the form (dist, idx1, idx2) where the centers of the clusters
    cluster_list[idx1] and cluster_list[idx2] have minimum distance dist.
    """
    result = (float('inf'), -1, -1)

    for index_u in range(0, len(cluster_list)):
        for index_v in range(0, len(cluster_list)):
            if index_u != index_v:
                distance = cluster_list[index_u].distance(cluster_list[index_v])
                if distance < result[0]:
                    result = (distance, index_u, index_v)

    return result

###################################################################
def fast_closest_pair(cluster_list):
    """
    Compute the distance between the closest pair of clusters in a list (fast)

    Input: cluster_list is list of clusters SORTED such that horizontal positions of their
    centers are in ascending order

    Output: tuple of the form (dist, idx1, idx2) where the centers of the clusters
    cluster_list[idx1] and cluster_list[idx2] have minimum distance dist.
    """

    total_clusters = len(cluster_list)
    result = ()

    if total_clusters <= 3:
        return slow_closest_pair(cluster_list)
    else:
        mid_idx = math.floor(total_clusters / 2)
        clusters_left = cluster_list[0:mid_idx]
        clusters_right = cluster_list[mid_idx:total_clusters]
        closest_left = fast_closest_pair(clusters_left)
        closest_right = fast_closest_pair(clusters_right)
        if closest_left[0] <= closest_right[0]:
            result = closest_left
        else:
            result = (closest_right[0], closest_right[1] + mid_idx, closest_right[2] + mid_idx)

        mid_x = (cluster_list[mid_idx - 1].horiz_center() + cluster_list[mid_idx].horiz_center()) / 2
        result2 = closest_pair_strip(cluster_list, mid_x, result[0])

        if result2[0] < result[0]:
            result = result2

    return result

##################################################################
class Cluster:
    """
    Class for creating and merging clusters of counties
    """

    def __init__(self, fips_codes, horiz_pos, vert_pos, population, risk):
        """
        Create a cluster based the models a set of counties' data
        """
        self._fips_codes = fips_codes
        self._horiz_center = horiz_pos
        self._vert_center = vert_pos
        self._total_population = population
        self._averaged_risk = risk

    def __eq__(self, other):
        """
        Compares this cluster with anothe cluster
        """
        return self._fips_codes == other._fips_codes and \
               self._horiz_center == other.horiz_center() and \
               self._vert_center == other.vert_center() and \
               self._total_population == other.total_population() and \
               self._averaged_risk == other.averaged_risk()

    def __repr__(self):
        """
        String representation assuming the module is "alg_cluster".
        """
        rep = "alg_cluster.Cluster("
        rep += str(self._fips_codes) + ", "
        rep += str(self._horiz_center) + ", "
        rep += str(self._vert_center) + ", "
        rep += str(self._total_population) + ", "
        rep += str(self._averaged_risk) + ")"
        return rep

    def fips_codes(self):
        """
        Get the cluster's set of FIPS codes
        """
        return self._fips_codes

    def horiz_center(self):
        """
        Get the averged horizontal center of cluster
        """
        return self._horiz_center

    def vert_center(self):
        """
        Get the averaged vertical center of the cluster
        """
        return self._vert_center

    def total_population(self):
        """
        Get the total population for the cluster
        """
        return self._total_population

    def averaged_risk(self):
        """
        Get the averaged risk for the cluster
        """
        return self._averaged_risk

    def copy(self):
        """
        Return a copy of a cluster
        """
        copy_cluster = Cluster(self._fips_codes, self._horiz_center, self._vert_center,
                               self._total_population, self._averaged_risk)
        return copy_cluster

    def distance(self, other_cluster):
        """
        Compute the Euclidean distance between two clusters
        """
        vert_dist = self._vert_center - other_cluster.vert_center()
        horiz_dist = self._horiz_center - other_cluster.horiz_center()
        return math.sqrt(vert_dist ** 2 + horiz_dist ** 2)

    def merge_clusters(self, other_cluster):
        """
        Merge one cluster into another
        The merge uses the relatively populations of each
        cluster in computing a new center and risk

        Note that this method mutates self
        """
        #if len(other_cluster.fips_codes()) == 0:
        if other_cluster.fips_codes() == 0:
            return self
        else:
            self._fips_codes.update(set(other_cluster.fips_codes()))
            #self._fips_codes = other_cluster.fips_codes()
            # compute weights for averaging
            self_weight = float(self._total_population)
            other_weight = float(other_cluster.total_population())
            self._total_population = self._total_population + other_cluster.total_population()
            self_weight /= self._total_population
            other_weight /= self._total_population

            # update center and risk using weights
            self._vert_center = self_weight * self._vert_center + other_weight * other_cluster.vert_center()
            self._horiz_center = self_weight * self._horiz_center + other_weight * other_cluster.horiz_center()
            self._averaged_risk = self_weight * self._averaged_risk + other_weight * other_cluster.averaged_risk()
            return self

    def cluster_error(self, data_table):
        """
        Input: data_table is the original table of cancer data used in creating the cluster.

        Output: The error as the sum of the square of the distance from each county
        in the cluster to the cluster center (weighted by its population)
        """
        # Build hash table to accelerate error computation
        fips_to_line = {}
        for line_idx in range(len(data_table)):
            line = data_table[line_idx]
            fips_to_line[line[0]] = line_idx

        # compute error as weighted squared distance from counties to cluster center
        total_error = 0
        counties = self.fips_codes()
        for county in counties:
            line = data_table[fips_to_line[county]]
            singleton_cluster = Cluster(set([line[0]]), line[1], line[2], line[3], line[4])
            singleton_distance = self.distance(singleton_cluster)
            total_error += (singleton_distance ** 2) * singleton_cluster.total_population()

        return total_error

#####################################################################
#
def gen_random_clusters(num_clusters):
    """
    My code
    """
    clusters = []
    while num_clusters > 0:
        clusters.append( Cluster( set(), random.uniform(-1, 1), random.uniform(-1, 1), 0, 0) )
        num_clusters -= 1

    return clusters

########################################################################
def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped



###################################################################
def do_simulation(min_num_of_nodes, max_num_of_nodes):
    """
    """
    num_of_nodes = min_num_of_nodes
    slow_values = {}
    fast_values = {}
    
    while num_of_nodes < max_num_of_nodes:
        cluster = gen_random_clusters(num_of_nodes)
        
        wrapped = wrapper(fast_closest_pair, cluster)
        total_time = timeit.timeit(wrapped, number=1)
        fast_values.update({num_of_nodes: total_time})

        wrapped = wrapper(slow_closest_pair, cluster)
        total_time = timeit.timeit(wrapped, number=1)
        slow_values.update({num_of_nodes: total_time})
        
        num_of_nodes += 1

    return (slow_values, fast_values)

#####################################################################
def plot_graphs( slow_values, fast_values ):

    """
    The horizontal axis for your plot should be the the number of initial 
    clusters while the vertical axis should be the running time of the function 
    in seconds. Please include a legend in your plot that distinguishes the two curves.
    """

    slow_xaxis = []
    for value in slow_values.keys():
        slow_xaxis.append(value)

    slow_yaxis = []    
    for value in slow_values.values():
        slow_yaxis.append(value)

    fast_xaxis = []
    for value in fast_values.keys():
        fast_xaxis.append(value)

    fast_yaxis = []    
    for value in fast_values.values():
        fast_yaxis.append(value)
        

    plt.plot(slow_xaxis, slow_yaxis, 'ro', color='r')
    plt.plot(fast_xaxis, fast_yaxis, 'ro', color='b')
    #plt.yscale('log')
    #plt.xscale('log')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Time in Seconds')
    plt.title('Comparison of SlowClosestPair And FastClosestPair Algorithms')
    red_patch = mpatches.Patch(color='red', label='SlowClosestPair')
    blue_patch = mpatches.Patch(color='blue', label='FastClosestPair')
    plt.legend(handles=[red_patch,blue_patch], loc=2)
    # plt.axis([0, 6, 0, 20])
    plt.show()

################################################################

#def costly_func(lst):
#    return map(lambda x: x^2, lst)

#short_list = range(10) 
#wrapped = wrapper(costly_func, short_list)
#print( timeit.timeit(wrapped, number=1))
#print( timeit.timeit(wrapped))

#clusters = gen_random_clusters(10)



values = do_simulation(2, 200)

plot_graphs(values[0], values[1])

#print('Slow Values #####################')
#print( values[0] )

#print('Fase Values #####################')
#print( values[1] )
