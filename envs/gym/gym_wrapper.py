import gym
import random
import numpy as np
from IPython import embed

class WrappedGymEnv():
    ''' A wrapper to transform Gym environments into standardized environments.'''
    
    def __init__(self, game, details, set_state_helper, get_state_helper, env_noise=0.0, no_timer=False, max_steps=None, term_prob=0.0, seed=None):
        self._game_name = game
        self._game = gym.make(game)
        self._game.seed(seed)
        self.players = [0]
        self._done = False
        self.num_actions = self._game.action_space.n

        if no_timer:
            self._game._max_episode_steps = np.inf

        if max_steps:
            self._game._max_episode_steps = max_steps

        self.env_noise = env_noise
        self.term_prob = term_prob
        self.details = details
        self._set_state_helper = set_state_helper
        self._get_state_helper = get_state_helper

    def step(self, action, _policy, player=None, ignore_mods=False):

        if not ignore_mods:

            # If env_noise is specified, pick a random action with probability env_noise.
            # Unlike in espilon exploration, this missed action is NOT revealed to the agent, who thinks it took the action it did.
            if random.random() < self.env_noise:
                action = random.randint(0, self.num_actions - 1)


        """Given a state, policy, and action, returns next state/reward seen by the current player.

        Args:
        action: The action taken at the current state.
        policy: The policy taken by other players – this is a function
                f(details) that returns a chosen action.
        """

        if self._done:
            raise Exception("Play has terminated. Please reset the environment.")

        _obs, reward, self._done, _info = self._game.step(action)
        next_state = self.get_state()

        # Randomly terminates episode with some given probability.
        if not ignore_mods:
            if random.random() < self.term_prob:
                self._done = True
                
        return (next_state, reward, self._done, self.details(next_state))

    def rollout(self, state, _player, policy):
        """Gets a sampled reward given initial state and policy.

        Args:
        state: The state we are starting from.
        policy: The policy taken by all players – this is a function
                f(details) that returns a chosen action.
        """

        self.set_state(state)
        total_reward = 0

        while not self._done:
            state = self.get_state()
            sampled = policy(self.details(state))
            (_obs, reward, _, _) = self.step(sampled, policy, None)
            total_reward += reward

        return total_reward


    def reset(self, _player, _policy):
        """Resets the environment and returns initial state for given player.

        Args:
        player: The player who we are currently simulating play for.
        policy: The policy taken by other players – this is a function
                f(details) that returns a chosen action.
        """

        self._done = False
        _ = self._game.reset()
        state = self.get_state()

        return (state, 0, False, self.details(state))

    def set_state(self, state):
        '''Sets the current state of the environment.'''

        self._done = False
        self._state = state

        self._game = self._set_state_helper(self._game, state)

    def get_state(self):
        '''Returns the current state of the environment. '''
        return self._get_state_helper(self._game)

    def get_detect_state(self, state):
        if self._game_name == "cartpole":
            _, info_set, _ = self.details(state)
            return info_set
        else:
            return state

    def eval(self, policy, iterations=10, render=False, max_steps=200):
        """Evaluates the current policy for a given player.

        Args:
        policy: The policy taken by all players – this is a function
                f(details) that returns a chosen action.
        iterations: The number of trials you want to simulate.
        """

        total_reward = 0

        for _ in range(iterations):
            (_obs, reward, _, _) = self.reset(None, policy)
            iter_reward = 0
            iter_reward += reward

            while not self._done and iter_reward < max_steps:
                state = self.get_state()
                sampled = policy(self.details(state))
                (_obs, reward, _, _) = self.step(sampled, policy, None, ignore_mods = True)
                iter_reward += reward

                if render:
                    self._game.render()

            total_reward += iter_reward
        
        print("Average Score: ", total_reward/iterations)
        return total_reward/iterations

    
