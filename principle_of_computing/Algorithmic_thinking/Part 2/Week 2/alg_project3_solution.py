"""
Implementation of clustering algorithms
"""

import alg_cluster

######################################################
# Code for closest pairs of clusters

def pair_distance(cluster_list, idx1, idx2):
    """
    Helper function that computes Euclidean distance between two clusters in a list

    Input: cluster_list is list of clusters, idx1 and idx2 are integer indices for two clusters

    Output: tuple (dist, idx1, idx2) where dist is distance between
    cluster_list[idx1] and cluster_list[idx2]
    """

    return (cluster_list[idx1].distance(cluster_list[idx2]), min(idx1, idx2), max(idx1, idx2))


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
    # we need not sort cluster_list because this list has already been sorted.
    sorted_clusters = cluster_list

    # 1. create the set
    indexes_s = []

    for index in range (0, len(sorted_clusters)):
        if abs(sorted_clusters[index].horiz_center() - horiz_center) < half_width:
            indexes_s.append(index)

    # 2. sort indexes in strip
    indexes_s.sort(key = lambda idx: sorted_clusters[idx].vert_center())

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

    if result[1] > result[2]:
        result = (result[0], result[2], result[1])
    return result


######################################################################
# Code for SlowClosestPair
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
    if result[1] > result[2]:
        result = (result[0], result[2], result[1])
    return result

######################################################################
# Code for FastClosestPair
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
        mid_idx = total_clusters // 2
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

    if result[1] > result[2]:
        result = (result[0], result[2], result[1])

    return result


######################################################################
# Code for hierarchical clustering
def hierarchical_clustering(cluster_list, num_clusters):
    """
    Compute a hierarchical clustering of a set of clusters
    Note: the function may mutate cluster_list

    Input: List of clusters, integer number of clusters
    Output: List of clusters whose length is num_clusters
    """

    # 1.
    # clusters_n = len(cluster_list)

    # 2. Initialize
    #    clusters_c = []
    #    for cluster in cluster_list:
    #      clusters_c.append(cluster.copy())
    # 2. Initialize
    clusters_c = []
    for cluster in cluster_list:
        clusters_c.append(cluster.copy())

    while len(clusters_c) > num_clusters:
        clusters_c.sort(key = lambda cluster: cluster.horiz_center())
        result = fast_closest_pair(clusters_c)

        clusters_c[result[1]].merge_clusters(clusters_c[result[2]])
        clusters_c.remove( clusters_c[result[2]])

    return clusters_c


######################################################################
# Code for k-means clustering
def kmeans_clustering(cluster_list, num_clusters, num_iterations):

    """
    Compute the k-means clustering of a set of clusters
    Note: the function may not mutate cluster_list

    Input: List of clusters, integers number of clusters and number of iterations
    Output: List of clusters whose length is num_clusters
    """
    # 1.
    cluster_n = len(cluster_list)

    # 2. Initialize k centers
    sorted_clusters = []
    for cluster in cluster_list:
        sorted_clusters.append(cluster.copy())

    # sort in descending order
    sorted_clusters.sort(key = lambda cluster: cluster.total_population(), reverse=True)

    # select the first num_clusters from the sorted list. This will yield the top
    # num_clusters from the perspective of population.
    old_clusters = []
    for index in range(0, num_clusters):
        old_clusters.append(sorted_clusters[index])

    # 3.
    ##for index_i in range(0, num_iterations):
    while num_iterations > 0:
        # 4 initialize k empty sets
        new_clusters = []
        index_k = num_clusters
        while index_k > 0:
            new_clusters.append( alg_cluster.Cluster(set(),0,0,0,0) )
            index_k -= 1

        # 5
        for index_j in range(0, cluster_n):
            # 6 argmin
            index_e = 0
            distance = float('inf')
            for index_f in range(0,num_clusters):
                current_distance = old_clusters[index_f].distance(cluster_list[index_j])
                if (current_distance < distance):
                    distance = current_distance
                    index_e = index_f
            new_clusters[index_e].merge_clusters(cluster_list[index_j])

        for index_f in range(0, num_clusters):
            old_clusters[index_f] = new_clusters[index_f]
        num_iterations -= 1

    return new_clusters
