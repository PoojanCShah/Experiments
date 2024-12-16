import logging
from sklearn.datasets import make_blobs as blobs
import time 
import matplotlib.pyplot as plt
from collections import defaultdict

from clustering import compute_cost, afkmc2, rsmeans

import numpy as np

logger = logging.getLogger()
logging.basicConfig(filename="experiment_logs.txt", format='%(message)s')


def experiment_1() : 

    data_seed = 42
    k_vals = [5,10,20,50,100]

    for k in k_vals : 

        logger.info(f"Running for k = {k}")

        afkmc2_costs = defaultdict(list)
        rsmeans_cost = []


        afkmc2_time = defaultdict(list)
        rsmeans_time = []

        iters = 100
        m = 200

        data, labels = blobs(n_samples=10000, n_features=2, cluster_std=1, centers=k, random_state=data_seed)

        logger.info("\n")

        for _ in range(iters):
            np.random.seed(_)
            print(_)
            start_time = time.time()
            rsmeans_cost.append(compute_cost(data, rsmeans(data, k)))
            rsmeans_time.append(time.time() - start_time)

        logger.info(f"rsmeans_cost : {np.mean(rsmeans_cost)}")
        logger.info(f"rsmeans_dev : {np.std(rsmeans_cost)}")
        logger.info(f"rsmeans_time : {np.mean(rsmeans_time):.5f} seconds")

        logger.info("\n")

        for _ in range(iters):
            np.random.seed(_)
            print(_)
            start_time = time.time()
            afkmc2_costs[m].append(compute_cost(data, afkmc2(data, k, m)))
            afkmc2_time[m].append(time.time() - start_time)
        logger.info(f"afkcmc2_cost for m = {m} : {np.mean(afkmc2_costs[m])}")
        logger.info(f"afkmc2_dev for m = {m} : {np.std(afkmc2_costs[m])}")
        logger.info(f"afkcmc2_time for m = {m} : {np.mean(afkmc2_time[m]):.5f} seconds")

        logger.info("\n")
        logger.info(f"{'-'*50}")


def experiment_2(k , sigma) : 

    data_seed = 42
    m_vals = [1,5,10,25,50,75,100,150,200]

    data, labels = blobs(n_samples=10000, n_features=2, cluster_std=sigma, centers=k, random_state=data_seed)

    iters = 10
    rsmeans_costs = []

    for _ in range(iters):
        np.random.seed(_)
        print(_)
        rsmeans_costs.append(compute_cost(data, rsmeans(data, k)))

    rsmeans_cost = np.mean(rsmeans_costs)
    rsmeans_dev = np.std(rsmeans_costs)

    afkmc_costs = []
    afkmc_devs = []


    for m in m_vals : 
        costs = []
        for _ in range(iters) : 
            np.random.seed(_)
            print(_)
            costs.append(compute_cost(data, afkmc2(data, k, m)))
        afkmc_costs.append(np.mean(costs))
        afkmc_devs.append(np.std(costs))

    
    plot_results(k,sigma, iters, m_vals, afkmc_costs, afkmc_devs, rsmeans_cost, rsmeans_dev)




def plot_results(k,sigma ,iters, m_vals, afkmc_costs, afkmc_devs, rsmeans_cost, rsmeans_dev):

    z_score = 1.96  # for 95% CI
    afkmc_upper = np.array(afkmc_costs) + z_score * np.array(afkmc_devs) / np.sqrt(iters)  
    afkmc_lower = np.array(afkmc_costs) - z_score * np.array(afkmc_devs) / np.sqrt(iters)
    
    rsmeans_upper = rsmeans_cost + z_score * rsmeans_dev / np.sqrt(iters) 
    rsmeans_lower = rsmeans_cost - z_score * rsmeans_dev / np.sqrt(iters)

    # Plotting
    plt.figure(figsize=(5, 5))

    # AFKMC2 costs and shading
    plt.plot(m_vals, afkmc_costs, label="AFkmc2 Cost", color='blue', marker = 'o')
    plt.fill_between(m_vals, afkmc_lower, afkmc_upper, color='blue', alpha=0.2)

    # RSMeans cost and shading (horizontal line since it's independent of m)
    plt.axhline(rsmeans_cost, color='red', linestyle='--', label="RSkmeans++ Cost")
    plt.fill_between(m_vals, rsmeans_lower, rsmeans_upper, color='red', alpha=0.2)

    # Plot labels and legend
    plt.xscale("log")
    plt.xlabel("m (markov chain length)")
    plt.ylabel("Clustering cost")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"exp2_{k}_{sigma}.png")





if __name__ == "__main__":

    experiment_1()
    for k in [5,10,20,50,100] : 
        experiment_2(k,1)
    