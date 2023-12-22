import pyspiel
import random
import numpy as np
import re

from .open_spiel_helpers import exploitability, policy, external_sampling_mccfr

class GymSpiel():
    ''' A wrapper to transform OpenSpiel environments into Gym-like environments.'''
    
    def __init__(self, game):
        self._game_name = game
        self._game = pyspiel.load_game(game)
        self.players = [x for x in range(self._game.num_players())]
        self._state = self._game.new_initial_state()
        self._done = False
        self._legal_actions = {}

        self._info_state_nodes = set()
        self._initialize_info_state_nodes(self._game.new_initial_state())

    def _initialize_info_state_nodes(self, state):
        """Initializes info_state_nodes.

        Create one _InfoStateNode per infoset. We could also initialize the node
        when we try to access it and it does not exist.

        Args:
            state: The current state in the tree walk. This should be the root node
            when we call this function from a CFR solver.
        """
        if state.is_terminal():
            return

        if state.is_chance_node():
            for action, unused_action_prob in state.chance_outcomes():
                self._initialize_info_state_nodes(state.child(action))
            return

        current_player = state.current_player()
        info_state = state.information_state_string(current_player)
        self._info_state_nodes.add(info_state)
        
        if self._game_name == "tic_tac_toe":
            self._legal_actions[self.ttt_convert(info_state)] = state.legal_actions()
        else:
            self._legal_actions[info_state] = state.legal_actions()

        for action in state.legal_actions():
            self._initialize_info_state_nodes(state.child(action))

    def ttt_convert(self, info_state_string):
        '''Converts OpenSpiel info state into merged info states for Tic-Tac-Toe.'''

        new_info_state = [int(s) for s in re.findall(r'\b\d+\b', info_state_string)]
        first, second = sorted(new_info_state[0::2]), sorted(new_info_state[1::2])
        new_info_state = (tuple(first), tuple(second))
        return new_info_state

    def details(self, state):
        '''Given a state, returns current player, info set, and legal actions available.
        
        Args:
        state: The state we want to find details for.
        '''

        if self._game_name == "tic_tac_toe":
            info_state_string = state.information_state_string(state.current_player())
            new_info_state = self.ttt_convert(info_state_string)
            return (state.current_player(), new_info_state, state.legal_actions())

        return (state.current_player(), state.information_state_string(state.current_player()), state.legal_actions())

    def _traverse(self, state, player, policy):
        """Given a state and policy, returns next state seen by the current player.

        Args:
        state: The current state in the tree walk.
        player: The player who we are currently simulating play for.
        policy: The policy taken by other players – this is a function
                f(details) that returns a chosen action.
        """

        self._state = state

        # if state is terminal, then no state follows
        if state.is_terminal():
            self._done = True
            reward = np.asarray(state.returns())[player]
            return (state, reward, True, None)
        
        # if chance node, pick a random chance outcome
        if state.is_chance_node():
            outcomes, probs = zip(*state.chance_outcomes())
            chosen = random.choices(outcomes, weights=probs)[0]
            return self._traverse(state.child(chosen), player, policy)
            
        # if we've returned to the original player's turn, return
        cur_player = state.current_player()
        if cur_player == player:
            return (state, 0., False, self.details(state))

        chosen = policy(self.details(state))

        return self._traverse(state.child(chosen), player, policy)


    def step(self, action, policy, player=None):
        """Given a state, policy, and action, returns next state/reward seen by the current player.

        Args:
        action: The action taken at the current state.
        policy: The policy taken by other players – this is a function
                f(details) that returns a chosen action.
        player: The player who we are currently simulating play for. 
                If None, assumes we are simulating for player in current environment state.
        state: The current state in the tree walk. If None, assumes we are in current state of environment.
        """

        if self._done:
            raise Exception("Play has terminated. Please reset the environment.")

        state = self._state

        if player is None:
            player = state.current_player()

        return self._traverse(state.child(action), player, policy)


    def rollout(self, state, player, policy, trials=1):
        """Gets a sampled reward given initial state and policy.

        Args:
        state: The state we are starting from.
        player: The player who we are currently simulating play for.
        policy: The policy taken by all players – this is a function
                f(details) that returns a chosen action.
        """

        def rollout_helper(state, player, policy):

            # if state is terminal, then no state follows
            if state.is_terminal():
                self._done = True
                reward = np.asarray(state.returns())[player]
                return reward
            
            # if chance node, pick a random chance outcome
            if state.is_chance_node():
                outcomes, probs = zip(*state.chance_outcomes())
                chosen = random.choices(outcomes, weights=probs)[0]
                return rollout_helper(state.child(chosen), player, policy)

            chosen = policy(self.details(state))

            return rollout_helper(state.child(chosen), player, policy)

        total_reward = 0

        for _ in range(trials):
            total_reward += rollout_helper(state, player, policy)

        return total_reward/trials


    def reset(self, player, policy):
        """Resets the environment and returns initial state for given player.

        Args:
        player: The player who we are currently simulating play for.
        policy: The policy taken by other players – this is a function
                f(details) that returns a chosen action.
        """
        self._done = False
        return self._traverse(self._game.new_initial_state(), player, policy)

    def set_state(self, state):
        '''Sets the current state of the environment.
        
        Args:
        state: The state we wish to set the environment to.
        '''

        if state.is_terminal():
            raise Exception("Please set to a non-terminal state.")

        self._done = False
        self._state = state

    def get_state(self):
        '''Returns the current state of the environment. '''
        return self._state
    
    def get_detect_state(self, state):

        if self._game_name == "tic_tac_toe":
            if not state.is_terminal():
                _, info, _ = self.details(state)
                return info
        else:
            return state

    def get_legal_actions(self, info_set):
        '''Returns legal actions given info set. '''
        # print(self._info_state_nodes)
        return self._legal_actions[info_set]

    def play_random(self, policy, iterations=250):

        def random_helper(state, player, policy):

            # if state is terminal, then no state follows
            if state.is_terminal():
                self._done = True
                reward = np.asarray(state.returns())[player]
                return reward
            
            # if chance node, pick a random chance outcome
            if state.is_chance_node():
                outcomes, probs = zip(*state.chance_outcomes())
                chosen = random.choices(outcomes, weights=probs)[0]
                return random_helper(state.child(chosen), player, policy)

            cur_player = state.current_player()

            if cur_player == player:
                chosen = policy(self.details(state))
            else:
                chosen = random.choice(state.legal_actions())

            return random_helper(state.child(chosen), player, policy)

        total_reward = 0
        for test_player in self.players:
            for _ in range(iterations):
                init_state = self.reset(0, policy)[0]
                total_reward += random_helper(init_state, test_player, policy)

        return total_reward/(iterations * 2)


    def eval(self, average_policy, current_policy, policy_fn, nash_conv):
        """Evaluates the average policy (in terms of exploitability).

        Args:
        average_policy: The policy you want to evaluate.
        """


        tabular_avg = policy.TabularPolicy(self._game)

        for info_state in self._info_state_nodes:
            avg_state = tabular_avg.policy_for_key(info_state)

            if self._game_name == "tic_tac_toe":
                new_info_set = self.ttt_convert(info_state)
                legal_actions = self.get_legal_actions(new_info_set)
                for action in legal_actions:
                    if new_info_set in current_policy:
                        avg_state[action] = current_policy[new_info_set][action]
            elif not nash_conv:
                legal_actions = self.get_legal_actions(info_state)
                for action in legal_actions:
                    if info_state in average_policy:
                        avg_state[action] = average_policy[info_state][action]
            else:
                legal_actions = self.get_legal_actions(info_state)
                for action in legal_actions:
                    if info_state in current_policy:
                        avg_state[action] = current_policy[info_state][action]

        if not nash_conv:
            return exploitability.exploitability(self._game, tabular_avg)
        else:
            return exploitability.nash_conv(self._game, tabular_avg, return_only_nash_conv=False).player_improvements
        


    
