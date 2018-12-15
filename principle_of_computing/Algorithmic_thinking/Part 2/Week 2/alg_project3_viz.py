#!/Applications/anaconda/bin/python

"""
Example code for creating and visualizing
cluster of county-based cancer risk data

Note that you must download the file
http://www.codeskulptor.org/#alg_clusters_matplotlib.py
to use the matplotlib version of this code
"""

# Flavor of Python - desktop or CodeSkulptor
DESKTOP = True

import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
import math
import random
#import urllib2
import urllib
import alg_cluster
import fixed

# conditional imports
if DESKTOP:
    import alg_project3_solution      # desktop project solution
    import alg_clusters_matplotlib
else:
    #import userXX_XXXXXXXX as alg_project3_solution   # CodeSkulptor project solution
    import alg_clusters_simplegui
    import codeskulptor
    codeskulptor.set_timeout(30)


###################################################
# Code to load data tables

# URLs for cancer risk data tables of various sizes
# Numbers indicate number of counties in data table

DIRECTORY = "http://commondatastorage.googleapis.com/codeskulptor-assets/"
DATA_3108_URL = DIRECTORY + "data_clustering/unifiedCancerData_3108.csv"
DATA_896_URL = DIRECTORY + "data_clustering/unifiedCancerData_896.csv"
DATA_290_URL = DIRECTORY + "data_clustering/unifiedCancerData_290.csv"
DATA_111_URL = DIRECTORY + "data_clustering/unifiedCancerData_111.csv"


def load_data_table(data_url):
    """
    Import a table of county-based cancer risk data
    from a csv format file
    """
    #data_file = urllib2.urlopen(data_url)
    data_file = urllib.request.urlopen(data_url)
    #data = data_file.read()
    data = data_file.read().decode("utf-8")
    data_lines = data.split('\n')
    print ("Loaded", len(data_lines), "data points")
    data_tokens = [line.split(',') for line in data_lines]
    return [[tokens[0], float(tokens[1]), float(tokens[2]), int(tokens[3]), float(tokens[4])] 
            for tokens in data_tokens]


############################################################
# Code to create sequential clustering
# Create alphabetical clusters for county data

def sequential_clustering(singleton_list, num_clusters):
    """
    Take a data table and create a list of clusters
    by partitioning the table into clusters based on its ordering
    
    Note that method may return num_clusters or num_clusters + 1 final clusters
    """
    
    cluster_list = []
    cluster_idx = 0
    total_clusters = len(singleton_list)
    cluster_size = float(total_clusters)  / num_clusters
    
    for cluster_idx in range(len(singleton_list)):
        new_cluster = singleton_list[cluster_idx]
        if math.floor(cluster_idx / cluster_size) != \
           math.floor((cluster_idx - 1) / cluster_size):
            cluster_list.append(new_cluster)
        else:
            cluster_list[-1] = cluster_list[-1].merge_clusters(new_cluster)
            
    return cluster_list
                
####################################################################
def compute_distortion(cluster_list,data_table):

    # data_table = load_data_table(DATA_111_URL)

    distortion = 0
    
    for cluster in cluster_list:      
        distortion += cluster.cluster_error(data_table)

    return distortion   

####################################################################
def run_distortion():
    
    data_table = load_data_table(DATA_111_URL)
    
    singleton_list = []
    for line in data_table:
        singleton_list.append(alg_cluster.Cluster(set([line[0]]), line[1], line[2], line[3], line[4]))
        
    #cluster_list = sequential_clustering(singleton_list, 15)	
    #print("Displaying", len(cluster_list), "sequential clusters")

    #cluster_list = alg_project3_solution.hierarchical_clustering(singleton_list, 9)
    #print( 'Hierarchical Distortion = ', compute_distortion(cluster_list, data_table) )

    cluster_list2 = []
    cluster_list = alg_project3_solution.kmeans_clustering(singleton_list, 9, 5)	
    print( 'KMeans Distortion = ', compute_distortion(cluster_list, data_table) )

#####################################################################
# Code to load cancer data, compute a clustering and 
# visualize the results


def run_example():
    """
    Load a data table, compute a list of clusters and 
    plot a list of clusters

    Set DESKTOP = True/False to use either matplotlib or simplegui
    """
    data_table = load_data_table(DATA_3108_URL)
    # data_table = load_data_table(DATA_111_URL)
    
    singleton_list = []
    for line in data_table:
        singleton_list.append(alg_cluster.Cluster(set([line[0]]), line[1], line[2], line[3], line[4]))
        
    #cluster_list = sequential_clustering(singleton_list, 15)	
    #print("Displaying", len(cluster_list), "sequential clusters")

    # cluster_list = alg_project3_solution.hierarchical_clustering(singleton_list, 9)
    # cluster_list = alg_project3_solution.hierarchical_clustering(singleton_list, 15)
    #print ("Displaying", len(cluster_list), "hierarchical clusters")

    #cluster_list = alg_project3_solution.kmeans_clustering(singleton_list, 9, 5)
    #cluster_list = alg_project3_solution.kmeans_clustering(singleton_list, 15, 5)	
    #print ("Displaying", len(cluster_list), "k-means clusters")

            
    # draw the clusters using matplotlib or simplegui
    if DESKTOP:
        alg_clusters_matplotlib.plot_clusters(data_table, cluster_list, False)
        #alg_clusters_matplotlib.plot_clusters(data_table, cluster_list, True)  #add cluster centers
    else:
        alg_clusters_simplegui.PlotClusters(data_table, cluster_list)   # use toggle in GUI to add cluster centers


#######################################################
def plot_graphs( hierarchical, kmeans ):

    """
    """

    h_xaxis = []
    for value in hierarchical.keys():
        h_xaxis.append(value)

    h_yaxis = []    
    for value in hierarchical.values():
        h_yaxis.append(value)

    k_xaxis = []
    for value in kmeans.keys():
        k_xaxis.append(value)

    k_yaxis = []    
    for value in kmeans.values():
        k_yaxis.append(value)
        

    plt.plot(h_xaxis, h_yaxis, color='r')
    plt.plot(k_xaxis, k_yaxis, color='b')
    plt.yscale('log')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Total Distortion')
    plt.title('Comparison of Distortion in Clustering Algorithms - DATA_111_URL')
    red_patch = mpatches.Patch(color='red', label='Hierarchical')
    blue_patch = mpatches.Patch(color='blue', label='K-Means')
    plt.legend(handles=[red_patch,blue_patch], loc=1)
    plt.show()
    
###################################
def run_example_two():
    
    #data_table = load_data_table(DATA_896_URL)
    #data_table = load_data_table(DATA_290_URL)
    data_table = load_data_table(DATA_111_URL)

    min_num_of_clusters = 6
    max_num_of_clusters = 20

    kmeans_points = {}
    hierarchical_points = {}
    
    num_of_clusters = min_num_of_clusters

    while num_of_clusters <= max_num_of_clusters:
        singleton_list = []
        for line in data_table:
            singleton_list.append(alg_cluster.Cluster(set([line[0]]), line[1], line[2], line[3], line[4]))
        
        # generate the clusters
        cluster_list = alg_project3_solution.hierarchical_clustering(singleton_list, num_of_clusters)

        # calculate the distortion
        distortion = compute_distortion(cluster_list, data_table)
        #print(distortion)
        hierarchical_points.update({num_of_clusters: distortion})

        singleton_list = []
        for line in data_table:
            singleton_list.append(alg_cluster.Cluster(set([line[0]]), line[1], line[2], line[3], line[4]))

        cluster_list = alg_project3_solution.kmeans_clustering(singleton_list, num_of_clusters, 5)
        distortion = compute_distortion(cluster_list, data_table)
        #print(distortion)
        kmeans_points.update({num_of_clusters: distortion})
        
        num_of_clusters += 1

    plot_graphs(hierarchical_points, kmeans_points)

#################################################
#run_example()
#run_distortion()

run_example_two()
