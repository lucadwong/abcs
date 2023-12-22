import random
from collections import defaultdict, deque
import numpy as np
from scipy.stats import ttest_ind, ks_2samp
import statistics
import copy
from IPython import embed

from .alg_utils import stable_softmax

class MCCFR():
    def __init__(self, env, exp_type, max_nodes=np.inf):
        # Store cumulative Q-values/regrets
        self._cumulative_Q = defaultdict(lambda: defaultdict(float))
        self._cumulative_regrets = defaultdict(lambda: defaultdict(float))

        # Store cumulative and average policies
        self._cumulative_policy = defaultdict(lambda: defaultdict(int))
        self._average_policy = defaultdict(lambda: defaultdict(float))
        self._current_policy = defaultdict(lambda: defaultdict(float))

        # Store sampled policy per-round
        self._sampled_policy = {}

        # Initialize environment
        self._env = env
        self._exp_type = exp_type
        self._MAX_NODES = max_nodes

        # Keep track of current learning iteration
        self._iteration = 0
        self._nodes_touched = 0

        # Stacked environment counts
        self._cartpole_count = 0
        self._spiel_count = 0

    def _get_action_probs(self, details, rm=False):
        '''Given details on a state, returns softmax action probabilities given current Q-values.
        
        Args:
        details: A (player, info_set, legal_actions) tuple containing information about the state
        '''

        _, info_set, legal_actions = details

        if rm:
            num_legal_actions = len(legal_actions)
            regrets = [self._cumulative_regrets[info_set][action] if action in self._cumulative_regrets[info_set] else 0 for action in legal_actions]
            positive_regrets = np.maximum(regrets,
                                  np.zeros(num_legal_actions, dtype=np.float64))
            sum_pos_regret = positive_regrets.sum()
            if sum_pos_regret <= 0:
                return np.ones(num_legal_actions, dtype=np.float64) / num_legal_actions
            else:
                return positive_regrets / sum_pos_regret

        # Softmax on cumulative Q-values
        if info_set in self._cumulative_Q:
            q_values = [self._cumulative_Q[info_set][action] if action in self._cumulative_Q[info_set] else 0 for action in legal_actions]
            return stable_softmax(np.asarray(q_values))

        # If we have not seen this state before, select random legal action
        else:
            return [1/len(legal_actions) for _ in legal_actions]

    def _outcome_action_probs(self, details, eps=0.6, rm=False):
        original_probs = np.array(self._get_action_probs(details, rm))
        uniform_probs = np.array([1/len(original_probs) for _ in original_probs])

        return eps * uniform_probs + (1 - eps) * original_probs
        

    def _get_action(self, details, outcome=False):
        '''Given details on a state, returns action to take given current Q-values.
        
        Args:
        details: A (player, info_set, legal_actions) tuple containing information about the state
        outcome: Flag to note whether we are using outcome sampling exploration policy
        '''

        _, _, legal_actions = details

        if outcome:
            probs = self._outcome_action_probs(details)
        else:
            probs = self._get_action_probs(details)

        return random.choices(legal_actions, weights=probs)[0]

    def _get_eval_action(self, details):
        '''Given details on a state, returns action to take given average policy.
        
        Args:
        details: A (player, info_set, legal_actions) tuple containing information about the state
        '''

        self._update_average_policy()

        player, info_set, legal_actions = details

        # Take average policy
        if info_set in self._average_policy:
            probs = [self._average_policy[info_set][x] if x in self._average_policy[info_set] else 0 for x in legal_actions]
            return random.choices(legal_actions, weights=probs)[0]

        # If we have not seen this state before, select random legal action
        else:
            return random.choice(legal_actions)

    def _get_cartpole_action(self, details):
        player, info_set, legal_actions = details

        q_values = np.array([self._cumulative_Q[info_set][action] for action in legal_actions])
        probs = stable_softmax(np.asarray(q_values))
        return random.choices(legal_actions, weights=probs)[0]

    def _update_q_values(self, info_set, action, q_value):
        ''' Updates Q-values given an observation.
        
        Args:
        info_set: info set from which the observation was taken
        action: action taken at the given info set
        q_value: Q-value obtained during simulation.
        '''

        # Update cumulative Q-values
        self._cumulative_Q[info_set][action] += q_value

        # Update average Q-values
        self._action_counts[info_set][action] += 1
        self._average_Q[info_set][action] = self._cumulative_Q[info_set][action]/self._action_counts[info_set][action]

    def _external_sampling_helper(self, player, state):
        ''' External sampling traversal. '''

        # We are done, if we blow out our budget
        if self._nodes_touched > self._MAX_NODES:
            return 0

        player, info_set, legal_actions = self._env.details(state)
        cur_policy = self._get_action_probs((player, info_set, legal_actions))

        child_values = defaultdict(float)
        value = 0

        for idx, action in enumerate(legal_actions):
            self._nodes_touched += 1
            self._env.set_state(state)
            (next_state, reward, done, next_info) = self._env.step(action, self._get_action, player=player)

            if not done:
                child_values[action] = reward + self._external_sampling_helper(player, next_state)
            else:
                child_values[action] = reward

            value += cur_policy[idx] * child_values[action]

        for idx, action in enumerate(legal_actions):
            self._cumulative_regrets[info_set][action] += (child_values[action] - value)
            self._cumulative_Q[info_set][action] += child_values[action]
            self._cumulative_policy[info_set][action] += cur_policy[idx]

        return value

    def _outcome_sampling_helper(self, player, state, reach_prob=1.0):
        ''' Outcome sampling traversal. '''

        # We are done, if we blow out our budget
        if self._nodes_touched > self._MAX_NODES:
            return 0

        player, info_set, legal_actions = self._env.details(state)
        cur_policy = self._get_action_probs((player, info_set, legal_actions))
        outcome_policy = self._outcome_action_probs((player, info_set, legal_actions))

        sampled_action = self._get_action(self._env.details(state), outcome=True)
        sampled_idx = legal_actions.index(sampled_action)

        child_values = defaultdict(float)
        value = 0

        for idx, action in enumerate(legal_actions):
            if action == sampled_action:
                self._nodes_touched += 1
                self._env.set_state(state)
                (next_state, reward, done, next_info) = self._env.step(sampled_action, self._get_action, player=player)

                try:
                    (game_name, _) = next_state
                    if game_name == "cartpole":
                        self._cartpole_count += 1
                    else:
                        self._spiel_count += 1
                except:
                    pass

                if not done:
                    new_reach_prob = reach_prob * outcome_policy[sampled_idx]
                    final_val, forward_prob = self._outcome_sampling_helper(player, next_state, new_reach_prob)
                    child_values[action] = (reward + final_val) * forward_prob
                    forward_prob *= cur_policy[sampled_idx]
                else:
                    term_prob = reach_prob * outcome_policy[sampled_idx]
                    forward_prob = 1.0 * cur_policy[sampled_idx]
                    final_val = reward/term_prob
                    child_values[action] = final_val
            else:
                child_values[action] = 0

            value += cur_policy[idx] * child_values[action]

        for idx, action in enumerate(legal_actions):
            self._cumulative_regrets[info_set][action] += (child_values[action] - value)
            self._cumulative_Q[info_set][action] += child_values[action]
            self._cumulative_policy[info_set][action] += cur_policy[idx]

        return final_val, forward_prob

    def _update_average_policy(self):
        ''' Updates the average policy. '''
        for info_state in self._cumulative_policy:
            prob_sum = sum(self._cumulative_policy[info_state].values())
            for action in self._env.get_legal_actions(info_state):
                if prob_sum == 0:
                    self._average_policy[info_state][action] = 1/len(self._cumulative_policy[info_state])
                else:
                    self._average_policy[info_state][action] = self._cumulative_policy[info_state][action]/prob_sum

    def average_policy(self):
        ''' Returns the average policy over self-play. '''
        self._update_average_policy()
        return copy.deepcopy(self._average_policy)

    def _get_current_policy(self):
        for info_set in self._cumulative_policy:
            legal_actions = self._env.get_legal_actions(info_set)
            q_values = np.array([self._cumulative_Q[info_set][action] for action in legal_actions]) * 1e10
            probs = stable_softmax(np.asarray(q_values))
            for idx, action in enumerate(legal_actions):
                self._current_policy[info_set][action] = probs[idx]
        
        return copy.deepcopy(self._current_policy)

    def run_round(self):
        ''' Runs a single iteration of value iteration. '''

        self._sampled_policy = {}

        for player in self._env.players:
            init_state = self._env.reset(player, self._get_action)[0]

            if self._exp_type == 'external':
                self._external_sampling_helper(player, init_state)

            elif self._exp_type == 'outcome':
                self._outcome_sampling_helper(player, init_state)

        self._iteration += 1

    def eval(self, env_type, render, nash_conv=False):
        ''' Evaluates the current policy on the environment.
        
        Args:
        env_type: Specify either "gym" or "spiel" depending on the type of the environment.
        '''

        if env_type == "spiel":
            assert not render, "Openspiel has no render"
            return self._env.eval(copy.deepcopy(self.average_policy()), copy.deepcopy(self._get_current_policy()), self._get_cartpole_action, nash_conv=nash_conv)
        elif env_type == "gym":
            return self._env.eval(self._get_cartpole_action, render=render)
        elif env_type == "mixed":
            return self._env.eval(copy.deepcopy(self.average_policy()), copy.deepcopy(self._get_current_policy()), self._get_cartpole_action, nash_conv=nash_conv)
        else:
            raise Exception("Only OpenSpiel and Gym environments are currently supported.")
