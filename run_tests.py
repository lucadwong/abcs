from algorithms.value_iteration import ValueIteration
from algorithms.mccfr import MCCFR
from algorithms.max_cfr import MaxCFR
from algorithms.abcs import ABCs
from envs.open_spiel.spiel_wrapper import GymSpiel
from envs.open_spiel.weighted_rps import WeightedRPS
from score_logger import run_tests
import matplotlib.pyplot as plt
from envs.gym import cartpole
from envs.mixed.cartpole_leduc import MixedEnv
from plotting_helpers import create_dfs, plot_scores

import random
import numpy as np
import json


def run_test_on_game(*, game_name, eval_len, seed=None, len_type="iters"):
    if game_name in ["leduc_poker", "tic_tac_toe", "kuhn_poker"]:
        env = GymSpiel(game_name)
        game_type = "spiel"

    elif game_name == "weighted_rps":
        env = WeightedRPS()
        game_type = "spiel"

    elif game_name == "cartpole":
        env = cartpole.get_new_markov_cartpole_env(term_prob=1/200, max_steps=500, seed=seed)
        game_type = "gym"

    elif game_name == "cartpole_leduc":
        env = MixedEnv("leduc_poker", seed=seed)
        game_type = "mixed"

    results = {}
    if game_name == "tic_tac_toe":
        eval_freq = eval_len // 10
    else:
        eval_freq = eval_len // 20

    if len_type == "iters":
        max_nodes = np.inf
    else:
        max_nodes = eval_len

    ES_CFR = MCCFR(env, exp_type='external')
    OS_CFR = MCCFR(env, exp_type='outcome')
    MAX_CFR = MaxCFR(env, boltzmann=True, max_nodes=max_nodes)

    ABC = ABCs(env, boltzmann=True, max_nodes=max_nodes, stat_level="action")
    BOLTZMANN = ValueIteration(env, exp_type="trajectory", boot=True, boltzmann=True)

    results["ABCs"] = run_tests(ABC, eval_len, eval_freq, "ABCs", game_type, len_type)
    
    if game_type != "gym":
        results["MAX-CFR"] = run_tests(MAX_CFR, eval_len, eval_freq, "MAX-CFR", game_type, len_type)
        if game_type != "mixed":
            results["ES-MCCFR"] = run_tests(ES_CFR, eval_len, eval_freq, "ES-MCCFR", game_type, len_type)

    results["OS-MCCFR"] = run_tests(OS_CFR, eval_len, eval_freq, "OS-MCCFR", game_type, len_type)
    results["Boltzmann Q-Learning"] = run_tests(BOLTZMANN, eval_len, eval_freq, "BQL", game_type, len_type)

    return results

def main():

    env_list = ["cartpole", "leduc_poker", "kuhn_poker", "weighted_rps", "cartpole_leduc"]
    env_lens = {"cartpole": 5e5, "leduc_poker": 5e6, "kuhn_poker": 5e5,
                  "weighted_rps": 1e6, "cartpole_leduc": 5e6}
    
    env_titles = {"cartpole": "Cartpole", "leduc_poker": "Leduc Poker", "kuhn_poker": "Kuhn Poker",
                  "weighted_rps": "Weighted Rock-Paper-Scissors"}

    for env in env_list:
        env_results = []
        for SEED in [0, 1, 2]:
            np.random.seed(SEED)
            random.seed(SEED)
            env_results.append(run_test_on_game(game_name=env, eval_len=env_lens[env], len_type="nodes_touched"))

        # save results to json file
        with open(f"results/raw/{env}_results.json", 'w') as f:
            json.dump(env_results, f)

        # plot results
        env_df = create_dfs(env_results)

        if env == "cartpole_leduc":
            spiel_dfs = create_dfs(env_results, score_type="spiel")
            cartpole_dfs = create_dfs(env_results, score_type="cartpole")

            plot_scores(cartpole_dfs, "Regret", "Cartpole + Leduc Poker (Cartpole Portion)", 'results/plots/cartleduc_cartpole.pdf', log=False)
            plot_scores(spiel_dfs, "Exploitability", "Cartpole + Leduc Poker (Leduc Portion)", 'results/plots/cartleduc_spiel.pdf')

        else:
            if env == "cartpole":
                env_df = create_dfs(env_results, cartpole=True)
                plot_scores(env_df, "Regret", title=env_titles[env], save_path=f"results/plots/{env}.pdf", log=False)
            else:
                env_df = create_dfs(env_results)
                plot_scores(env_df, "Exploitability", title=env_titles[env], save_path=f"results/plots/{env}.pdf")


if __name__ == "__main__":
    main()

