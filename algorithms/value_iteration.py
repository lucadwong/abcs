import random
from collections import defaultdict, deque
import numpy as np
from scipy.stats import ttest_ind, ks_2samp
import statistics
import copy
from IPython import embed
import math

from .alg_utils import stable_softmax

class ValueIteration():
    def __init__(self, env, exp_type, boot, alpha = 0.05, max_nodes=np.inf, boltzmann=False, temp=1):
        # Store cumulative and average Q-values
        self._cumulative_Q = defaultdict(lambda: defaultdict(float))
        self._average_Q = defaultdict(lambda: defaultdict(float))
        self._action_counts = defaultdict(lambda: defaultdict(int))

        # Store cumulative and average policies
        self._cumulative_policy = defaultdict(lambda: defaultdict(int))
        self._average_policy = defaultdict(lambda: defaultdict(float))
        self._current_policy = defaultdict(lambda: defaultdict(float))

        # Store sampled policy per-round
        self._sampled_policy = {}

        # Initialize environment
        self._env = env
        self._exp_type = exp_type
        self._boot = boot
        self._alpha = alpha
        self._MAX_NODES = max_nodes

        # Keep track of current learning iteration
        self._iteration = 0
        self._nodes_touched = 0

        # Boltzmann Q parameters
        self._boltzmann = boltzmann
        self._temp = temp

        # Stacked environment counts
        self._cartpole_count = 0
        self._spiel_count = 0

    def _temp_schedule(self):
        if self._iteration % 20 == 0:
            self._temp = max(self._temp * 0.995, 0)

    def _get_action_probs(self, details):
        '''Given details on a state, returns softmax action probabilities given current Q-values.
        
        Args:
        details: A (player, info_set, legal_actions) tuple containing information about the state
        '''

        _, info_set, legal_actions = details

        # Make sure that we explore every action if we haven't done so already
        action_counts = np.array([self._action_counts[info_set][action] for action in legal_actions])

        if 0 in action_counts:
            probs = np.array([0 if action_counts[action] != 0 else 1 for action in range(len(legal_actions))])
            probs = probs/sum(probs)
            return probs

        if self._boltzmann:
            q_values = np.array([self._average_Q[info_set][action] for action in legal_actions])
            softmax_factor = max(self._temp, 1/max(action_counts))
            q_values *= (1/softmax_factor)
            probs = stable_softmax(q_values)
            return probs

        # Softmax on cumulative Q-values
        else:
            q_values = np.array(list(self._average_Q[info_set].values())) * min(self._action_counts[info_set].values())
            probs = stable_softmax(np.asarray(q_values))

        return probs

    def _get_action(self, details):
        '''Given details on a state, returns action to take given current Q-values.
        
        Args:
        details: A (player, info_set, legal_actions) tuple containing information about the state
        '''

        _, _, legal_actions = details
        probs = self._get_action_probs(details)

        return random.choices(legal_actions, weights=probs)[0]

    def _get_cartpole_action(self, details):
        player, info_set, legal_actions = details

        action_counts = np.array([self._action_counts[info_set][action] for action in legal_actions])
        q_values = np.array([self._average_Q[info_set][action] for action in legal_actions]) * min(action_counts)
        probs = stable_softmax(np.asarray(q_values))
        return random.choices(legal_actions, weights=probs)[0]


    def _get_eval_action(self, details):
        '''Given details on a state, returns action to take given average policy.
        
        Args:
        details: A (player, info_set, legal_actions) tuple containing information about the state
        '''

        player, info_set, legal_actions = details

        self._update_average_policy()

        # Take average policy
        if info_set in self._average_policy:
            probs = [self._average_policy[info_set][x] if x in self._average_policy[info_set] else 0 for x in legal_actions]
            return random.choices(legal_actions, weights=probs)[0]

        # If we have not seen this state before, select random legal action
        else:
            return random.choice(legal_actions)

    def _update_q_values(self, info_set, action, q_value):
        ''' Updates Q-values given an observation.
        
        Args:
        info_set: info set from which the observation was taken
        action: action taken at the given info set
        q_value: Q-value obtained during simulation.
        '''

        # Update average Q-values
        cumulative_reward = self._action_counts[info_set][action] * self._average_Q[info_set][action]
        cumulative_reward += q_value
        self._action_counts[info_set][action] += 1
        self._average_Q[info_set][action] = cumulative_reward/self._action_counts[info_set][action]

    # This update is done for all traversal methods. The main difference between them
    # is simply which infosets to do this update in.
    def _core_update(self, player, state, boot):

        player, info_set, legal_actions = self._env.details(state)
        next_states, dones = defaultdict(), defaultdict()

        if info_set not in self._sampled_policy or True:
            self._sampled_policy[info_set] = self._get_action(self._env.details(state))

        for action in legal_actions:

            if not boot or action == self._sampled_policy[info_set]:

                self._nodes_touched += 1

                self._env.set_state(state)
                (next_state, reward, done, next_info) = self._env.step(action, self._get_action, player=player)

                try:
                    (game_name, _) = next_state
                    if game_name == "cartpole":
                        self._cartpole_count += 1
                    else:
                        self._spiel_count += 1
                except:
                    pass

                if boot:
                    # Update cumulative Q-values (Bootstrapped)
                    try:
                        # Maybe no next infoset
                        reward += max(self._average_Q[next_info[1]].values())
                    except:
                        pass
                    
                else:
                    # Update cumulative Q-values (Monte-Carlo)
                    if not done:
                        reward += self._env.rollout(next_state, player, self._get_action)
                    
                self._update_q_values(info_set, action, reward)
                next_states[action] = next_state
                dones[action] = done
        
        # Update the cumulative policy whether or not we are on the path of play
        cur_policy = self._get_action_probs(self._env.details(state))
        for action_idx, action in enumerate(legal_actions):
            self._cumulative_policy[info_set][action] += cur_policy[action_idx]
        
        return next_states, dones

    def _outcome_sampling(self, player, state, on_path=True, boot=True):
        ''' Outcome sampling traversal. '''

        # We are done, if we blow out our budget
        if self._nodes_touched > self._MAX_NODES:
            return None

        _, info_set, legal_actions = self._env.details(state)

        next_states, dones = self._core_update(player, state, boot)
        chosen_action = self._sampled_policy[info_set]
        
        done, next_state = dones[chosen_action], next_states[chosen_action]

        if not done and on_path:
            self._outcome_sampling(player, next_state, on_path=True)

    def _full_traverse(self, player, state, on_path=True, boot=True):
        ''' External sampling traversal. '''

        # We are done, if we blow out our budget
        if self._nodes_touched > self._MAX_NODES:
            return None

        player, info_set, legal_actions = self._env.details(state)
        next_states, dones = self._core_update(player, state, on_path, boot)

        for action in legal_actions:
            done, next_state = dones[action], next_states[action]

            # Continue traversal if the game hasn't terminated (branches out everything)
            if not done:
                if on_path and action == self._sampled_policy[info_set]:
                    self._full_traverse(player, next_state, on_path=True)
                else:
                    self._full_traverse(player, next_state, on_path=False)

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
            action_counts = np.array([self._action_counts[info_set][action] for action in legal_actions])
            q_values = np.array([self._average_Q[info_set][action] for action in legal_actions]) * 1e10
            probs = stable_softmax(np.asarray(q_values))
            for idx, action in enumerate(legal_actions):
                self._current_policy[info_set][action] = probs[idx]
        
        return copy.deepcopy(self._current_policy)

    def run_round(self):
        ''' Runs a single iteration of value iteration. '''

        self._sampled_policy = {}

        for player in self._env.players:
            init_state = self._env.reset(player, self._get_action)[0]

            if self._exp_type == 'full':
                self._full_traverse(player, init_state, boot=self._boot)

            elif self._exp_type == 'trajectory':
                self._outcome_sampling(player, init_state, boot=self._boot)

            elif self._exp_type == 'nonstat':
                self._nonstat_traverse(player, init_state, detect_method="q_based")

        self._temp_schedule()
        self._iteration += 1

    def eval(self, env_type, render, nash_conv=False):
        ''' Evaluates the current policy on the environment.
        
        Args:
        env_type: Specify either "gym" or "spiel" depending on the type of the environment.
        '''

        if env_type == "spiel":
            assert not render, "Openspiel has no render"
            return self._env.eval(copy.deepcopy(self.average_policy()), copy.deepcopy(self._get_current_policy()), self._get_cartpole_action, nash_conv)
        elif env_type == "gym":
            return self._env.eval(self._get_cartpole_action, render=render)
        elif env_type == "mixed":
            return self._env.eval(copy.deepcopy(self.average_policy()), copy.deepcopy(self._get_current_policy()), self._get_cartpole_action, nash_conv)
        else:
            raise Exception("Only OpenSpiel and Gym environments are currently supported.")
