import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import seaborn as sns
from itertools import repeat
import json
from collections import defaultdict

# Seaborn plot settings
sns.set_style("ticks")
sns.set_context("paper", font_scale=1.2)

# color mapping for plots
label_dict = {"ABCs": "ABCs", "OS-MCCFR": "OS-MCCFR", "Boltzmann Q-Learning": "BQL", "MAX-CFR": "MAX-CFR", "ES-MCCFR": "ES-MCCFR"}
color_dict = {"ABCs": sns.color_palette("Set2")[0], "OS-MCCFR": sns.color_palette("Set2")[1], "Boltzmann Q-Learning": sns.color_palette("Set2")[2], "MAX-CFR": sns.color_palette("Set2")[3], "ES-MCCFR": sns.color_palette("Set2")[4]}

def create_dfs(results, score_type="normal", cartpole=False):

    all_dfs = {}
    method_dicts = defaultdict(lambda: defaultdict(list))
    methods = set([x for x in results[0].keys()])

    method_iters = defaultdict(lambda: [])
    for method in methods:
        for trial in results:
            nodes, scores, stat_counts, pvals = trial[method][:4]
            method_iters[method].append(nodes)

    avg_iters = {}
    for method in method_iters:
        print(method_iters[method])
        avg_iters[method] = np.mean(method_iters[method], axis=0)

    for trial in results:
        for method in trial:
            nodes, scores, stat_counts, pvals = trial[method][:4]
            for index in range(len(nodes)):
                method_dicts[method]["iteration"].append(avg_iters[method][index])
                if score_type == "normal":
                    if not cartpole:
                        method_dicts[method]["score"].append(scores[index])
                    else:
                        method_dicts[method]["score"].append(200 - scores[index])
                elif score_type == "combined":
                    combined_score = (200 - scores[index]['cartpole'])/200 + scores[index]['spiel']/2.373611111111111
                    method_dicts[method]["score"].append(combined_score)
                elif score_type == "spiel":
                    method_dicts[method]["score"].append(scores[index]['spiel'])
                elif score_type == "cartpole":
                    method_dicts[method]["score"].append(200 - scores[index]['cartpole'])
                    
                try:
                    method_dicts[method]["nonstat_percentage"].append((stat_counts[index][0]-1)/(stat_counts[index][0] + stat_counts[index][1]))
                except:
                    method_dicts[method]["nonstat_percentage"].append(None)

    for method in methods:
        all_dfs[method] = pd.DataFrame.from_dict(dict(method_dicts[method]))
        
    return all_dfs

def plot_scores(results, ylabel, title, save_path, mixed=False, log=True):
    for key in results:
        results_df = results[key]
        sns.lineplot(data=results_df, x="iteration", y="score", color=color_dict[key], label=label_dict[key])
        
    plt.xlabel("Nodes Touched")
    if log:
        plt.ylabel(ylabel + " (log scale)")
        plt.yscale('log')
    else:
        plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc='best')
    sns.despine()
    plt.savefig(save_path, bbox_inches = "tight")
    plt.show()