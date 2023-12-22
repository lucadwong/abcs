from tqdm import tqdm

def run_tests(solver, eval_len, eval_freq, label, env_type, x_axis):
    iters = []
    scores = []
    stat_counts = []
    env_counts = {"cartpole": [], "spiel": []}
    env_stat_counts = {"cartpole": [], "spiel": []}

    if x_axis == "iters":
        for i in tqdm(range(eval_len)):
            solver.run_round()

            if (i + 1) % eval_freq == 0:
                iters.append(i)
                scores.append(solver.eval(env_type, render=False))
                try:
                    stat_counts.append((solver._nonstat_counts, solver._stat_counts))
                except:
                    pass
                print("Nodes Touched:", solver._nodes_touched)
                if env_type != "mixed":
                    print(f'{"Exploitability:" if env_type == "spiel" else "Reward:"}', scores[-1])
                else:
                    print(f"Exploitability: {scores[-1]['spiel']}, Reward: {scores[-1]['cartpole']}")

    elif x_axis == "nodes_touched":
        cur_len = 0
        last_check = 0

        with tqdm(total=eval_len) as pbar:

            while cur_len < eval_len:
                pbar.update(cur_len - pbar.n)

                solver.run_round()
                cur_len = solver._nodes_touched
                if last_check == 0 or solver._nodes_touched - last_check >= eval_freq:
                    iters.append(solver._nodes_touched)
                    scores.append(solver.eval(env_type, render=False, nash_conv=False))

                    try:
                        stat_counts.append((solver._nonstat_counts, solver._stat_counts))
                    except:
                        pass

                    last_check = solver._nodes_touched
                    print("Nodes Touched:", solver._nodes_touched)
                    if env_type != "mixed":
                        print(f'{"Exploitability:" if env_type == "spiel" else "Reward:"}', scores[-1])
                    else:
                        print(f"Exploitability: {scores[-1]['spiel']}, Reward: {scores[-1]['cartpole']}")


            try:
                stat_counts.append((solver._nonstat_counts, solver._stat_counts))
            except:
                pass

            iters.append(solver._nodes_touched)
            scores.append(solver.eval(env_type, render=False))
            print("Nodes Touched:", solver._nodes_touched)
            if env_type != "mixed":
                print(f'{"Exploitability:" if env_type == "spiel" else "Reward:"}', scores[-1])
            else:
                print(f"Exploitability: {scores[-1]['spiel']}, Reward: {scores[-1]['cartpole']}")

    try:
        return iters, scores, stat_counts, solver._nonstat_counts["pvals"], env_counts, env_stat_counts
    except: 
        return iters, scores, stat_counts, None, env_counts, env_stat_counts