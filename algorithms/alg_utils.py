import numpy as np
from IPython import embed
import random

def stable_softmax(Q):
    ''' Return softmax values given array of q-values. '''
    max_elem = max(Q)
    probs = np.exp(Q - max_elem)/sum(np.exp(Q - max_elem))
    return probs

# Find the rolling average of an interval with window_size
def rolling_average(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

# Computes the percentage difference between A and B, with the smaller value as the base
def percent_diff(A, B):
    assert A > 0 and B > 0, "Must be positive values"
    return abs(A - B) / min(A, B)

def hardmax(probs):
    best_options = np.argwhere(probs == np.amax(probs))
    max_probs = np.zeros(probs.shape)
    max_probs[best_options] = 1/len(best_options)
    
    return max_probs

def online_avg(last_avg, last_N, new_val):
    return ((last_avg*last_N)+new_val)/(last_N+1)