import gym
import numpy as np
from .gym_wrapper import WrappedGymEnv
from IPython import embed
import random

num_bins = 10

def discretize_range(lower_bound, upper_bound):
            return np.linspace(lower_bound, upper_bound, num_bins + 1)[1:-1]

def discretize_value(value, bins):
        return np.digitize(x=value, bins=bins)



RANGES = [2.4, 3.0, 0.5, 2.0]


state_bins = [
            # Cart position.
            discretize_range(-2.4, 2.4),
            # Cart velocity.
            discretize_range(-3.0, 3.0),
            # Pole angle.
            discretize_range(-0.5, 0.5),
            # Tip velocity.
            discretize_range(-2.0, 2.0)
        ]

def unhash(info_state):
    raw_state = []

    # Unhash the state by adding a random number in [0, 1) and then undoing the binning.
    for i, v in enumerate(info_state):
        raw_state.append((v + random.random()) / num_bins * 2 * RANGES[i] - RANGES[i])
    
    return np.array(raw_state)

def hash(state_info):
        obs = tuple([discretize_value(feature, state_bins[i]) for i, feature in enumerate(tuple(state_info))])
        return obs

def details(state):
    (state_info, steps_taken) = state
    # return (0, (hash(state_info), steps_taken), [0, 1])
    return (0, hash(state_info), [0, 1])

def get_state_helper(env):
    steps_taken = env._elapsed_steps
    return (env.unwrapped.state, steps_taken)

def get_state_helper_discrete(env):
    steps_taken = env._elapsed_steps

    mixed_state = unhash(hash(env.unwrapped.state))
    assert hash(mixed_state) == hash(env.unwrapped.state), "Hashing is not invertible"
    return (mixed_state, steps_taken)

def set_state_helper(env, state):
    (state_info, steps_taken) = state
    env._elapsed_steps = steps_taken
    env.state = env.unwrapped.state = state_info
    return env

def get_cartpole_env():
    return WrappedGymEnv("CartPole-v0", details, set_state_helper, get_state_helper)

# A version of cartpole which is Markovian. Should be identical to cartpole.py with two exceptions
# 1) There is no time limit, as steps_taken is set to 0 always.
# 2) Noise is added so the state ends in finite time.
def get_markov_cartpole_env(noise=0.0, seed=None):
    return WrappedGymEnv("CartPole-v0", details, set_state_helper, get_state_helper, env_noise=noise, no_timer=True, seed=seed)

def get_new_markov_cartpole_env(term_prob=0.0, seed=None, max_steps=None):
    return WrappedGymEnv("CartPole-v0", details, set_state_helper, get_state_helper, term_prob=term_prob, no_timer=True, max_steps=max_steps, seed=seed)