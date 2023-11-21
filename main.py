import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram


class Cluster:
    def __init__(self, center, ID):
        self.number = ID
        self.center = center
        self.IDs = [ID]
        self.size = 1
        self.linkage_num = ID - 1

    def __str__(self):
        return f"(ID: {self.number}, Size: {self.size}, Coords{self.center})"

    def combine(self, other):
        self.number = min(self.number, other.number)
        self.IDs.extend(other.IDs)
        self.center = np.mean([self.center, other.center], axis=0)
        self.size += other.size


def part_A(data):
    correlation_matrix = data.corr()
    correlation_matrix = correlation_matrix.round(2)
    for i in range(len(correlation_matrix)):
        correlation_matrix.iloc[i, i] = None
    print(correlation_matrix)
    max_corr = correlation_matrix.stack().nlargest(1).idxmin()
    print(max_corr)
    min_corr = correlation_matrix.stack().nsmallest(1).idxmin()
    print(min_corr)
    correlation_matrix_abs = correlation_matrix.abs()
    correlation_sums = correlation_matrix_abs.sum(axis=1)
    print(correlation_sums)


def create_dendrogram(linkage_matrix):
    plt.figure(figsize=(10, 5))
    dendrogram(linkage_matrix, orientation='top', labels=None, distance_sort='descending', show_leaf_counts=True)
    plt.title('Dendrogram')
    plt.xlabel('Sample Index')
    plt.ylabel('Cluster Distance')
    plt.show()


def agglomerate(data):
    clusters = []
    linkage_matrix = np.zeros((len(data) - 1, 4), dtype=float)
    # loop through the dataframe and create a cluster object for each row
    for i in range(len(data)):
        center = data.iloc[i].values
        clusters.append(Cluster(center, i + 1))

    # linkage number is the number of clusters that have been combined
    # used in creating a linkage matrix for the dendrogram
    linkage_number = len(clusters)
    # the index in the linkage matrix
    linkage_index = 0


    while len(clusters) > 1:
        min_distance = float("inf")
        min_i = 0
        min_j = 0
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                # calculate the Euclidean distance between the two clusters
                distance = np.linalg.norm(clusters[i].center - clusters[j].center)
                # if it's the smallest distance so far, save it
                if distance < min_distance:
                    min_distance = distance
                    min_i = i
                    min_j = j

        cluster1 = clusters[min_i]
        cluster2 = clusters[min_j]
        linkage_matrix[linkage_index] = [clusters[min_i].linkage_num, clusters[min_j].linkage_num, min_distance, clusters[min_i].size + clusters[min_j].size]
        clusters[min_i].combine(clusters[min_j])
        clusters[min_i].linkage_num = linkage_number
        linkage_number += 1
        linkage_index += 1

        clusters.pop(min_j)

    return clusters[0], linkage_matrix


if __name__ == "__main__":
    filename = sys.argv[1]

    data = pd.read_csv(filename)
    data = data.drop("ID", axis=1)
    cluster, linkage_matrix = agglomerate(data)
    create_dendrogram(linkage_matrix)


