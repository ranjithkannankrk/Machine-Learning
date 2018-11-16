import os
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import sys


def main():
    file_name = sys.argv[1]
    no_of_clusters = int(sys.argv[2])
    my_data = genfromtxt(file_name, delimiter=' ')
    kmeans_lyod(my_data, no_of_clusters)

def kmeans_lyod(data, no_of_clusters):
    centroid = []
    centroid = initialize_centroids(data, centroid, no_of_clusters)  

    previous_centroid = [[] for i in range(no_of_clusters)] 

    iter = 0
    while not (check_convergence(centroid, previous_centroid, iter)):
        iter += 1

        clusters = [[] for i in range(no_of_clusters)]

        clusters = calc_euclidean_dist(data, centroid, clusters)

        m = 0
        for cl in clusters:
            previous_centroid[m] = centroid[m]
            centroid[m] = np.mean(cl, axis=0).tolist()
            m += 1

    print("The number of iterations for the algorithm to converge is")
    print(iter)
    j = 0
    for cl in clusters:
        print("The number of data points in this cluster")
        print(str(len(cl)))
        if j == 0:
            plt.scatter(np.array(cl).T[0], np.array(cl).T[1], c = 'blue')
        if j == 1:
            plt.scatter(np.array(cl).T[0], np.array(cl).T[1], c = 'red')
        if j == 2:
            plt.scatter(np.array(cl).T[0], np.array(cl).T[1], c = 'green')
        if j == 3:
            plt.scatter(np.array(cl).T[0], np.array(cl).T[1], c = 'black')
        if j == 4:
            plt.scatter(np.array(cl).T[0], np.array(cl).T[1], c = 'violet')
        if j == 5:
            plt.scatter(np.array(cl).T[0], np.array(cl).T[1], c = 'yellow')
        j = j + 1

    plt.show()

    return

def check_convergence(centroid, previous_centroids, iterate):
    max_iterations = 1000
    if iterate > max_iterations:
        return True
    return previous_centroids == centroid

def initialize_centroids(data, centroid, no_of_clusters):
    for i in range(0, no_of_clusters):
        centroid.append(data[np.random.randint(0, len(data), size=1)].flatten().tolist())
    return centroid
    
def calc_euclidean_dist(data, centroid, cluster):
    for point in data:
        index = min([(i[0], np.linalg.norm(point-centroid[i[0]])) \
                            for i in enumerate(centroid)], key=lambda g:g[1])[0]
        try:
            cluster[index].append(point)
        except KeyError:
            cluster[index] = [point]
       
    for cl in cluster:
        if not cluster:
            cl.append(data[np.random.randint(0, len(data), size=1)].flatten().tolist())

    return cluster


def initialize_centroids(data, centroid, no_of_clusters):
    for i in range(0, no_of_clusters):
        centroid.append(data[np.random.randint(0, len(data), size=1)].flatten().tolist())
    return centroid

if __name__== "__main__":
  main()
