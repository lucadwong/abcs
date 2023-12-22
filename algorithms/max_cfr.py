import random
from collections import defaultdict, deque
import numpy as np
from scipy.stats import ttest_ind, ks_2samp
import statistics
import copy
from IPython import embed
import math
import scipy.stats
from tqdm import tqdm
from matplotlib import pyplot as plt

from .alg_utils import stable_softmax, hardmax, online_avg

class MaxCFR():
    def __init__(self, env, max_nodes=np.inf, boltzmann=True, temp=10.0, gamma=1.0):

        self.gamma = gamma
        
        # Initialize environment
        self._env = env
        self._MAX_NODES = max_nodes

        # Boltzmann Q parameters
        self._boltzmann = boltzmann
        self._temp = temp

        self._average_Q = defaultdict(lambda: defaultdict(float))
        self._action_counts = defaultdict(lambda: defaultdict(int))
        self._max_action = defaultdict(int)
        self._all_visited = defaultdict(lambda: False)
        self._majority_threshold = 0.5
        self.nodes_touched_by_iteration = []

        # Store cumulative, average, and current policies
        self._cumulative_policy = defaultdict(lambda: defaultdict(int))
        self._average_policy = defaultdict(lambda: defaultdict(float))
        self._current_policy = defaultdict(lambda: defaultdict(float))

        # Keep track of current learning iteration
        self._iteration = 0
        self._nodes_touched = 0

        # Stacked environment counts
        self._cartpole_count = 0
        self._spiel_count = 0

    def _temp_schedule(self):
        '''Updates BQL temperature according to exponential scheme.'''
        if self._iteration % 10 == 0:
            self._temp = self._temp * 0.995

    def _get_action_probs(self, details, policy_type):
        """Given details on a state, returns softmax action probabilities.
        
        Args:
            details: A (player, info_set, legal_actions) tuple containing information about the state.
            policy_type: Determines which policy we are using (exploratory, current, average, fixed).

        Returns:
            Array of probabilities for each action.
        """
        _, info_set, legal_actions = details

        if policy_type == "exploration":
            # Make sure that we explore every action if we haven't done so already
            all_visited = True
            if not self._all_visited[info_set]:
                action_counts = np.array([self._action_counts[info_set][action] for action in legal_actions])

                if 0 in action_counts:
                    probs = np.array([0 if action_counts[action] != 0 else 1 for action in range(len(legal_actions))])
                    probs = probs/sum(probs)
                    all_visited = False
                else:
                    self._all_visited[info_set] = True

            if all_visited:
                q_values = np.array(list(self._average_Q[info_set].values()))

                softmax_factor = 1/self._max_action[info_set]

                q_values *= (1/softmax_factor)
                probs = stable_softmax(np.asarray(q_values))

        elif policy_type == "current":
            action_counts = np.array([self._action_counts[info_set][action] for action in legal_actions])
            q_values = np.array([self._average_Q[info_set][action] for action in legal_actions]) * max(action_counts)
            probs = stable_softmax(np.asarray(q_values))

        elif policy_type == "average":
            # Take average policy
            if info_set in self._average_policy:
                probs = [self._average_policy[info_set][x] if x in self._average_policy[info_set] else 0 for x in legal_actions]
            else:
                return np.ones(legal_actions)/len(legal_actions)

        elif policy_type == "fixed":
            if len(self.fixed_policy[info_set]) < len(legal_actions):
                self.fixed_policy[info_set] = np.ones(len(legal_actions)) / len(legal_actions)

            probs = self.fixed_policy[info_set]

        return probs

    def _get_action(self, policy_type):
        """Returns policy function given policy type.

        Args:
            policy_type: Determines which policy we are using (exploratory, current, average, fixed).
            
        Returns:
            Policy function that chooses action given info state.
        """
        assert policy_type in ["exploration", "current", "average", "fixed"], "Invalid policy type."

        if policy_type == "average":
            self._update_average_policy()

        def eval_policy(details):
            _, _, legal_actions = details
            probs = self._get_action_probs(details, policy_type)
            return random.choices(legal_actions, weights=probs)[0]

        return eval_policy

    def _query_child(self, state, action, expand_child):
        """Retrieves information about the child of the given (state, action) pair.

        Args:
            state: State for which to query children.
            action: The action to simulate taking.
            expand_child: Whether to continue exploration down the tree.

        Returns:
            Tuple of the form (Q-values of child, transition info, Q-value estimate of (state, action) pair).
        """
        player, info_set, legal_actions = self._env.details(state)
        if action not in legal_actions:
            raise Exception("Invalid action")
            
        self._nodes_touched += 1

        self._env.set_state(state)
        step_data = self._env.step(action, self._get_action("exploration"), player=player)
        next_state, reward, done, next_info = step_data

        try:
            (game_name, _) = next_state
            if game_name == "cartpole":
                self._cartpole_count += 1
            else:
                self._spiel_count += 1
        except:
            pass

        if not done:
            next_player, next_infoset, next_actions = self._env.details(next_state)
            old_q_values = self._get_q_values(next_state)
            continuation_value = None

            if expand_child:
                self.expand_repeat[info_set, action] = True
                continuation_value = reward + self.gamma * self._max_cfr_helper(next_player, next_state)

            return old_q_values, step_data, continuation_value
        
        # If episode is done, return None instead of Q-values
        return None, step_data, reward
    
    def _core_update(self, state, child_data, cur_policy, update_child):
        """Core update function that is used by both CFR and Q-learning.The ONLY difference in the updates 
        is that CFR contains a "nonstationary" correction factor, designed to be bit-for-bit identical to standard CFR.

        Args:
            state: State we are currently updating.
            child_data: Data for children (see query_child above).
            cur_policy: Policy used to update cumulative policy.
            update_child: List of booleans used to determine which Q-value actions to update.
        """
        player, info_set, legal_actions = self._env.details(state)
        child_values, nonstationary_correction, value = defaultdict(float), defaultdict(float), 0.0

        for idx, (action, do_update) in enumerate(zip(legal_actions, update_child)):

            if do_update:
                # To simplify the node_count, we don't call env.step here. Instead, we merely
                # retrieve the data that was gathered from _query_child.
                old_q, (next_state, reward, done, next_info), _ = child_data[idx]
                child_values[action] = reward

                if not done:
                    next_player, next_infoset, next_actions = self._env.details(next_state)

                    child_policy = hardmax(old_q)
                    child_values[action] += max(old_q)

                    if not self.update_repeat[info_set, action]:
                        new_q = self._get_q_values(next_state)
                        reconstructed_value = np.zeros(len(next_actions))
                        for i, action_count in enumerate(self._action_counts[next_infoset].values()):
                            reconstructed_value[i] = action_count * new_q[i] - (action_count - 1) * old_q[i]

                        nonstationary_correction[action] = sum(child_policy * (reconstructed_value - new_q))
                    
                corrected_update = child_values[action] + nonstationary_correction[action]

                # Update average Q-values
                self._average_Q[info_set][action] = online_avg(self._average_Q[info_set][action], self._action_counts[info_set][action], corrected_update)
                self._action_counts[info_set][action] += 1
                self._max_action[info_set] = max(self._max_action[info_set], self._action_counts[info_set][action])
                
            self._cumulative_policy[info_set][action] += cur_policy[idx]
            self.update_repeat[info_set, action] = True

    # Get q-values and pads it with zeros if the infoset is new.
    def _get_q_values(self, state):
        player, infoset, legal_actions = self._env.details(state)
        if len(self._average_Q[infoset]) != len(legal_actions):
            values = np.array([self._average_Q[infoset][action] for action in legal_actions])
        else:
            values = np.array(list(self._average_Q[infoset].values()))
        
        return values

    def _max_cfr_helper(self, player, state):
        """Recursively explores the game tree.

        Args:
            player: Player we are currently simulating play for.
            state: Current state from which we are exploring.

        Returns: 
            Discounted reward for the given state.
        """
        # Short circuit if we've hit our visit limit
        if self._nodes_touched > self._MAX_NODES:
            return 0

        # Gets state data
        player, info_set, legal_actions = self._env.details(state)
        cur_policy = self._get_action_probs((player, info_set, legal_actions), policy_type="exploration")

        # Keeps track of all values going forwards
        child_data = [None] * len(legal_actions)

        sampled_action = (self._get_action("exploration"))(self._env.details(state))
        update_child = []

        current_continuation_value = 0.
        expanded_policy_weight = 0.

        for idx, action in enumerate(legal_actions):

            # We only ever branch the FIRST instance we see an infostate (if there are repeats)
            expand_child = (action == sampled_action or not self.expand_repeat[info_set, action])
            if expand_child:
                child_data[idx] = self._query_child(state, action, expand_child=expand_child)
            else:
                child_data[idx] = (None, None, None)
            update_child.append(expand_child)

            _, _, continuation_value = child_data[idx]

            if continuation_value is not None:
                current_continuation_value += cur_policy[idx] * continuation_value
                expanded_policy_weight += cur_policy[idx]
            
        # Note that we expand the children BEFORE doing the update.
        self._core_update(state, child_data, cur_policy, update_child)

        # If we don't expand all the children, we need to normalize the continuation value
        return current_continuation_value / expanded_policy_weight

    def _update_average_policy(self):
        """Updates the average policy."""
        for info_state in self._cumulative_policy:
            prob_sum = sum(self._cumulative_policy[info_state].values())
            for action in self._env.get_legal_actions(info_state):
                if prob_sum == 0:
                    self._average_policy[info_state][action] = 1/len(self._cumulative_policy[info_state])
                else:
                    self._average_policy[info_state][action] = self._cumulative_policy[info_state][action]/prob_sum

    def average_policy(self):
        """Returns the average policy over self-play."""
        self._update_average_policy()
        return copy.deepcopy(self._average_policy)

    def _get_current_policy(self):
        """Returns tabular form of the current policy."""
        for info_set in self._average_Q:
            legal_actions = self._env.get_legal_actions(info_set)
            action_counts = np.array([self._action_counts[info_set][action] for action in legal_actions])
            q_values = np.array([self._average_Q[info_set][action] for action in legal_actions]) * max(action_counts)
            probs = stable_softmax(np.asarray(q_values))
            for idx, action in enumerate(legal_actions):
                self._current_policy[info_set][action] = probs[idx]
        
        return copy.deepcopy(self._current_policy)

    def run_round(self):
        """Runs a single iteration of MaxCFR."""

        self.update_repeat = defaultdict(bool)
        self.expand_repeat = defaultdict(bool)

        for player in self._env.players:
            init_state = self._env.reset(player, self._get_action("exploration"))[0]
            _ = self._max_cfr_helper(player, init_state)

        self._iteration += 1

        if self._boltzmann:
            self._temp_schedule()

    def eval(self, env_type, render, nash_conv=False):
        """Evaluates the current policy on the environment.
        
        Args:
            env_type: Specify either "gym" or "spiel" depending on the type of the environment.
            render: Toggle rendering of Gym environment on/off.

        Returns:
            Performance on the specified environment (exploitability/average reward).
        """
        if env_type == "spiel":
            assert not render, "Openspiel has no render"
            return self._env.eval(self.average_policy(), self._get_current_policy(), self._get_action("current"), nash_conv)
        elif env_type == "gym":
            return self._env.eval(self._get_action("current"), render=render)
        elif env_type == "mixed":
            return self._env.eval(self.average_policy(), self._get_current_policy(), self._get_action("current"), nash_conv)
        else:
            raise Exception("Only OpenSpiel and Gym environments are currently supported.")