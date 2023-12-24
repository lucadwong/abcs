# ABCs (Adaptive Branching through Child Stationarity)

This is the official repository for the paper: <strong>Easy as ABCs: Unifying Boltzmann Q-Learning and Counterfactual Regret
Minimization</strong>. 

ABCs is a best-of-both-worlds algorithm combining Boltzmann Q-learning (BQL), a classic reinforcement learning algorithm for single-agent domains, and counterfactual regret minimization (CFR), a central algorithm for learning
in multi-agent domains. ABCs adaptively chooses what fraction of the environment to explore each iteration by measuring the stationarity of the environment’s reward and transition
dynamics.

## Installing and Running Experiments

All required dependecies are listed in `requirements.txt`. Many of the environments we benchmark on, as well as some utilities for measuring performance, come from DeepMind's 
<a href=https://github.com/google-deepmind/open_spiel>OpenSpiel</a> and OpenAI's <a href="https://github.com/openai/gym">Gym</a>.

The complete code for ABCs and other benchmarks is provided in `algorithms/abcs`. We also provide the code to reproduce the experiments presented in our paper in `run_tests.py`. You can view the raw results in `results/raw` and the plotted performance
of the different methods in `results/plots`. We provide the code to benchmark ABCs on the following environments:

- Leduc Poker
- Kuhn Poker
- Cartpole
- Weighted Rock-Paper-Scissors
- Stacked Environment of Cartpole + Leduc Poker




