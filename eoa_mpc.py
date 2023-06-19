from re import L
import numpy as np

from gurobipy import *

import matplotlib.pyplot as plt
import math

from os.path import exists


# num_rankings here is number of users
datasets = {'Amazon Ratings':{'num_items': 93332, 'num_rankings': 3000, 'sensitivity': 4, 'rating_range': [1, 5]}, # 'num_items': 93332, 'num_rankings': 3000
            'Amazon Digital Music':{'num_items': 3000, 'num_rankings': 3000, 'sensitivity': 4, 'rating_range': [1, 5]}, # 'num_items': 7854, 'num_rankings': 3000
            'Book Crossing':{'num_items': 3000, 'num_rankings': 3000, 'sensitivity': 9, 'rating_range': [1, 10]}, # 'num_items': 12020, 'num_rankings': 3000
            'jester':{'num_items': 140, 'num_rankings': 3000, 'sensitivity': 20, 'rating_range': [0, 20]}, # 'num_rankings': 
            'MovieLens 100K':{'num_items': 1682, 'num_rankings': 943, 'sensitivity': 4, 'rating_range': [1, 5]}, # 'num_items': 1682, 'num_rankings': 943
            'MovieLens 1M':{'num_items': 3706, 'num_rankings': 6040, 'sensitivity': 4, 'rating_range': [1, 5]}, # 'num_items': 3706, 'num_rankings': 6040
            'Netflix Prize':{'num_items': 462188, 'num_rankings': 3000, 'sensitivity': 4, 'rating_range': [1, 5]}, # 'num_items': 462188, 'num_rankings': 3000
            'yahoo-song':{'num_items': 208093, 'num_rankings': 3000, 'sensitivity': 100}}

dataset_index = 1
datasets_names = ['Amazon Ratings', 'Amazon Digital Music', 'Book Crossing', 'jester', 'MovieLens 100K', 'MovieLens 1M', 'Netflix Prize', 'yahoo-song']
dataset_name = datasets_names[dataset_index]


# number of items
n = 100
k = n

num_rankings = datasets[dataset_name]['num_rankings']

rating_min_vector = np.ones((k,)) * datasets[dataset_name]['rating_range'][0]
rating_max_vector = np.ones((k,)) * datasets[dataset_name]['rating_range'][1]

# attention parameter
p = 0.5

# theta parameter (quality constraint)
theta_values = [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
theta_index = 5
theta_value = theta_values[theta_index]

A_relevance = np.zeros((n,))
R_relevance = np.zeros((n,))

A_reranked = np.zeros((n,))
R_reranked = np.zeros((n,))

A_reranked_uf_metric = np.zeros((n,))
R_reranked_uf_metric = np.zeros((n,))

A_reranked_uf_metric_noisy = np.zeros((n,))
R_reranked_uf_metric_noisy = np.zeros((n,))

# attention weights
w = np.zeros((k,))
for i in range(k):
    w[i] = p * ((1 - p) ** i)
w = w / np.sum(w)

unfairness_measures_relevance = []
unfairness_measures_reranked = []
unfairness_measures_reranked_uf_metric = []
unfairness_measures_reranked_uf_metric_noisy = []

DCG_relevance_unnormalized = []
DCG_reranked = []
DCG_reranked_unnormalized = []
DCG_reranked_uf_metric = []
DCG_reranked_uf_metric_noisy_unnormalized = []

DCG_reranked_over_IDCG = []
DCG_reranked_unnormalized_over_IDCG_unnormalized = []
DCG_reranked_uf_metric_over_IDCG = []
DCG_reranked_uf_metric_noisy_over_IDCG = []
DCG_reranked_uf_metric_noisy_unnormalized_over_IDCG_unnormalized = []

global_epsilon = 0.5 # [0.5, 1, 10, 100, 1000, 10000, 100000, n * num_rankings]

for cid in range(num_rankings):
    user_item_rating_preds_all = np.load(f'user_relevance_rankings_{dataset_name}/{cid}_relevance_ranking.npy')
    user_item_rating_preds = user_item_rating_preds_all[:n]

    user_item_rating_preds_sorted_array_indices = np.argsort(user_item_rating_preds)[::-1]
    user_item_rating_preds_sorted = user_item_rating_preds[user_item_rating_preds_sorted_array_indices]
    user_item_rating_preds_sorted_unnormalized = user_item_rating_preds_sorted.copy()

    user_item_rating_preds_sorted_normalized = (user_item_rating_preds_sorted - rating_min_vector) / (rating_max_vector - rating_min_vector)
    user_item_rating_preds_sorted_normalized = user_item_rating_preds_sorted_normalized / np.sum(user_item_rating_preds_sorted_normalized)

    # add attention and relevance scores of k-top relevant items in this ranking to A_relevance and R_relevance of entire item set vectors
    for i in range(n):
        A_relevance[user_item_rating_preds_sorted_array_indices[i]] += w[i]
        R_relevance[user_item_rating_preds_sorted_array_indices[i]] += user_item_rating_preds_sorted_normalized[i]

    unfairness_measure_relevance = np.sum(np.abs(A_relevance - R_relevance))
    unfairness_measures_relevance.append(unfairness_measure_relevance)

    ################################################################
    ################## Unfairness Metric, w/noise ##################
    ################################################################
    milp_model = Model("milp")

    X = milp_model.addMVar(shape=(k, k), vtype=GRB.BINARY, name="X")

    sensitivity = 1
    noise = np.random.laplace(loc=0.0, scale=sensitivity/(global_epsilon / (n * num_rankings)), size=len(user_item_rating_preds))
    unfairness_metric_noisy = A_reranked_uf_metric_noisy - R_reranked_uf_metric_noisy + noise
    np.save(f'unfairness_metric_noisy_{cid}', unfairness_metric_noisy)
    unfairness_metric_noisy /= (num_rankings / global_epsilon)
    np.save(f'unfairness_metric_noisy_{cid}_scaled', unfairness_metric_noisy)

    # for k-most relevant items per ranking
    unfairness_obj = np.zeros((n, n)) # (item_current_indices, item_reranked_indices)
    for i in range(n):
        for j in range(n):
            unfairness_obj[i, j] = np.absolute(unfairness_metric_noisy[user_item_rating_preds_sorted_array_indices[i]] + 
                                            w[j] -
                                            user_item_rating_preds_sorted_normalized[i])

    milp_model.setObjective(sum(unfairness_obj[i, :] @ X[i, :] for i in range(n)), GRB.MINIMIZE)

    # calculate DCG value using first k items
    DCG = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            DCG[i, j] = ((2 ** user_item_rating_preds_sorted_normalized[i]) - 1) / np.log2(j + 2)

    # calculate IDCG for ranking based only on relevance
    IDCG_k = 0
    for i in range(k):
        IDCG_k += ((2 ** user_item_rating_preds_sorted_normalized[i]) - 1) / np.log2(i + 2)

    IDCG_n = 0
    for i in range(n):
        IDCG_n += ((2 ** user_item_rating_preds_sorted_normalized[i]) - 1) / np.log2(i + 2)

    c1 = milp_model.addConstr(theta_value * IDCG_k <= sum(DCG[i, :k] @ X[i, :k] for i in range(n)), "c1")
    c2 = milp_model.addConstrs((1 == X[i, :].sum() for i in range(n)), "c2")
    c3 = milp_model.addConstrs((1 == X[:, i].sum() for i in range(n)), "c3")

    milp_model.optimize()

    optimized_X = np.zeros((n * n,))
    for i in range(n * n):
        optimized_X[i] = milp_model.getVars()[i].x

    optimized_X = optimized_X.reshape((n, n))
    optimized_X_indices = np.where(optimized_X==1)[1]

    optimized_DCG_value = sum(DCG[i, :] @ optimized_X[i, :] for i in range(n))
    DCG_reranked_uf_metric_noisy_over_IDCG.append(optimized_DCG_value / IDCG_n)

    for i in range(k):
        A_reranked_uf_metric_noisy[user_item_rating_preds_sorted_array_indices[i]] += w[optimized_X_indices[i]]
        R_reranked_uf_metric_noisy[user_item_rating_preds_sorted_array_indices[i]] += user_item_rating_preds_sorted_normalized[i]

    unfairness_measure_reranked_uf_metric_noisy = np.sum(np.abs(A_reranked_uf_metric_noisy - R_reranked_uf_metric_noisy))
    unfairness_measures_reranked_uf_metric_noisy.append(unfairness_measure_reranked_uf_metric_noisy)

    ################################################################
    ################## Unfairness Metric, w/noise ##################
    ################################################################

iteration = np.arange(0, num_rankings)

if len(unfairness_measures_reranked) == num_rankings:
    np.save(f'unfairness_measures_relevance_{dataset_name}_numrankings{num_rankings}', unfairness_measures_relevance)
    np.save(f'unfairness_measures_reranked_{dataset_name}_numrankings{num_rankings}', unfairness_measures_reranked)
    np.save(f'DCG_reranked_over_IDCG_{dataset_name}_numrankings{num_rankings}', DCG_reranked_over_IDCG)

np.save(f'unfairness_measures_reranked_uf_metric_noisy_{dataset_name}_{global_epsilon}_numrankings{num_rankings}_old_scaled', unfairness_measures_reranked_uf_metric_noisy)
np.save(f'DCG_reranked_uf_metric_noisy_over_IDCG_{dataset_name}_{global_epsilon}_numrankings{num_rankings}_old_scaled', DCG_reranked_uf_metric_noisy_over_IDCG)

plt.subplot(2, 1, 1)
plt.plot(iteration, unfairness_measures_relevance, label='unfairness_measures_relevance')
if len(unfairness_measures_reranked) == num_rankings:
    plt.plot(iteration, unfairness_measures_reranked, label='unfairness_measures_reranked')
plt.plot(iteration, unfairness_measures_reranked_uf_metric_noisy, label='unfairness_measures_reranked_uf_metric_noisy')
plt.title(f'Amortized Unfairness Score over n Iterations (MPC: {global_epsilon}) - Dataset: {dataset_name}')
plt.xlabel('Iteration')
plt.ylabel('Unfairness Measure')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(iteration, list(np.ones((num_rankings,))), label='1')
plt.plot(iteration, list(np.ones((num_rankings,)) * theta_value), label='theta value')
if len(DCG_reranked_over_IDCG) == num_rankings:
    plt.plot(iteration, DCG_reranked_over_IDCG, label='DCG_reranked_over_IDCG')
if len(DCG_reranked_uf_metric_noisy_over_IDCG) == num_rankings:
    plt.plot(iteration, DCG_reranked_uf_metric_noisy_over_IDCG, label='DCG_reranked_uf_metric_noisy_over_IDCG')
plt.title(f'NDCG (MPC {global_epsilon})')
plt.xlabel('Iteration')
plt.ylabel('NDCG')
plt.legend()

plt.show()
