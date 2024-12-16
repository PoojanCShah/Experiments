import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs as blobs
import time
import logging
from collections import defaultdict


logger = logging.getLogger()
logging.basicConfig(filename="experiment_logs.txt", level=logging.INFO)


def saveplot(data) : 

    x = [d[0] for d in data]
    y = [d[1] for d in data]

    plt.scatter(x,y)
    plt.savefig("plot.png")



'''
Experiments for comparing the performance of seedings for clustering algorithms :

1. Random Seeding
2. Kmeans++
3. AFKMC2
4. RSmeans

'''



def random_seeding(data: np.array, k: int) -> list[int]:
    '''
    input : 
    data : n x d numpy array representing the dataset.
    k : number of cluster centers to initialize.

    output : 
    indices : a list of length k consisting of integer indices 
              in the range [0,n-1] representing the randomly initialized cluster centers.
    '''
    n, _ = data.shape

    # Randomly select k unique indices from the range [0, n-1]
    indices = np.random.choice(n, size=k, replace=False).tolist()

    return indices


def kmeanspp(data: np.array, k: int) -> list[int]:
    '''
    input : 
    data : n x d numpy array representing the dataset.
    k : number of cluster centers to initialize.

    output : 
    indices : a list of length k consisting of integer indices 
              in the range [0,n-1] representing the initialized cluster centers using k-means++.
    '''
    n, d = data.shape

    # Initialize the list of indices for cluster centers
    indices = []

    # Step 1: Randomly choose the first cluster center
    first_center = np.random.randint(0, n)
    indices.append(first_center)

    # Step 2: Initialize distances to track the minimum distance to any center
    distances = np.linalg.norm(data - data[first_center], axis=1) ** 2

    # Step 3: Select the remaining k-1 cluster centers
    for _ in range(1, k):
        # Choose a new center with a probability proportional to the squared distance
        probabilities = distances / np.sum(distances)
        new_center = np.random.choice(n, p=probabilities)
        indices.append(new_center)

        # Update the minimum distances for all points
        new_distances = np.linalg.norm(data - data[new_center], axis=1) ** 2
        distances = np.minimum(distances, new_distances)

    return indices



def afkmc2(data: np.array, k: int, m: int) -> list[int]:
    '''
    input : 
    data : n x d numpy array representing the dataset.
    k : number of cluster centers to initialize.
    m : chain length parameter for AFK-MC2.

    output : 
    indices : a list of length k consisting of integer indices 
              in the range [0,n-1] representing the initialized cluster centers using AFK-MC2.
    '''
    n, d = data.shape

    # Step 1: Uniformly sample the first center
    indices = []
    c1_index = np.random.randint(0, n)
    indices.append(c1_index)

    # Step 2: Compute the initial distribution q(x) for all points
    distances = np.linalg.norm(data - data[c1_index], axis=1) ** 2
    q = 0.5 * distances / np.sum(distances) + (0.5 / n)

    # Step 3: Main loop to select k centers
    for i in range(1, k):
        # Sample x according to q
        x_index = np.random.choice(n, p=q)
        dx = np.linalg.norm(data[x_index] - data[indices], axis=1).min() ** 2

        # Perform m steps of Metropolis-Hastings
        for _ in range(1, m):
            y_index = np.random.choice(n, p=q)
            dy = np.linalg.norm(data[y_index] - data[indices], axis=1).min() ** 2
            
            # Acceptance criterion
            if (dy * q[x_index]) > (dx * q[y_index]) * np.random.uniform(0, 1):
                x_index, dx = y_index, dy

        indices.append(x_index)

    return indices




def rsmeans(data: np.array, k: int) -> list[int]:
    '''
    KMeans++ using Rejection Sampling

    Input : 
    data : n x d numpy array representing the dataset.
    k : number of clusters to initialize.


    Output :
    indices : a list of length k consisting of integer indices
              representing the initialized cluster centers using RSMeans.
    '''
    n, d = data.shape

    # Step 1: Center the data by subtracting the mean
    mean = np.mean(data, axis=0)
    data_centered = data - mean

    # Step 2: Precompute norms for sampling
    norms_squared = np.sum(data_centered ** 2, axis=1)
    frobenius_norm_squared = np.sum(norms_squared)

    # Step 3: Initialize centers
    indices = []
    first_center = np.random.randint(0, n)
    indices.append(first_center)

    # Step 4: Select remaining k-1 centers
    for _ in range(1, k):
        while True:
            r = np.random.uniform(0, 1)
            r_prime = np.random.uniform(0,1)

            if r <= frobenius_norm_squared / (frobenius_norm_squared + n * norms_squared[first_center]):
                # Sample from Dv
                i = np.random.choice(n, p=norms_squared / frobenius_norm_squared)
            else:
                # Sample uniformly
                i = np.random.randint(0, n)

            # Compute rejection probability
            min_dist_squared = np.linalg.norm(data[i] - data[indices], axis = 1).min() **2
            rejection_prob = 0.5 * min_dist_squared / (norms_squared[i] + norms_squared[first_center])

            if r_prime <= rejection_prob:
                indices.append(i)
                break

    return indices
 


def compute_cost(data : np.array, indices : list[int]) -> float : 

    '''
    input : 
    data : n x d numpy array representing the dataset.
    indices : a list of length k consisting of integer indices 
            in the range [0,n-1] representing the cluster centers.

    output : 
    cost : the kmeans cost of clustering the data using the goven cluster centers.
    '''

    # Extract the cluster center points from the data using the indices
    cluster_centers = data[indices]

    # Compute all pairwise distances between data points and cluster centers
    distances = np.linalg.norm(data[:, np.newaxis, :] - cluster_centers, axis=2)

    # Find the minimum distance for each data point
    min_distances = np.min(distances, axis=1)

    # Compute the total cost as the sum of squared minimum distances
    cost = np.sum(min_distances ** 2)

    return cost




