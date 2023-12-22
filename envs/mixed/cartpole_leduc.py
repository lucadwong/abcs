import random
import numpy as np
from ..open_spiel.spiel_wrapper import GymSpiel
from ..gym import cartpole

class MixedEnv():

    def __init__(self, game_name, seed):
            self.players = [0, 1]
            self._state = None

            self._spiel_env = GymSpiel(game_name)
            self._cartpole_env = cartpole.get_new_markov_cartpole_env(term_prob=1/100, max_steps=500, seed=seed)

            self._cartpole_terminated = False

    def set_state(self, state):
        (game_flag, state_data) = state
        if game_flag == "cartpole":
            self._cartpole_env.set_state(state_data)
        else:
            self._spiel_env.set_state(state_data)
        
        self._state = state

    def step(self, action, policy, player):
        (game_flag, state_data) = self._state
        if game_flag == "cartpole":
            (next_state, cartpole_reward, done, info_state) = self._cartpole_env.step(action, policy, player)
            if done and not self._cartpole_terminated:
                # self._cartpole_terminated = True
                (spiel_state, spiel_reward, done, info_state) = self._spiel_env.reset(player, policy)
                return (("spiel", spiel_state), spiel_reward + cartpole_reward, done, info_state)
            else:
                return (("cartpole", next_state), cartpole_reward, done, info_state)
        else:
            (next_state, reward, done, info_state) = self._spiel_env.step(action, policy, player)
            return (("spiel", next_state), reward, done, info_state)
        
    def reset(self, player, policy):
        if player == 0:
            self._cartpole_terminated = False
            (next_state, reward, done, info_state) = self._cartpole_env.reset(player, policy)
            return (("cartpole", next_state), reward, done, info_state)

        elif player == 1:
            (next_state, reward, done, info_state) = self._spiel_env.reset(player, policy)
            return (("spiel", next_state), reward, done, info_state)

    def details(self, state):
        (game_flag, state_data) = state
        if game_flag == "cartpole":
            return self._cartpole_env.details(state_data)
        else:
            return self._spiel_env.details(state_data)

    def get_detect_state(self, state):
        (game_flag, state_data) = state
        if game_flag == "cartpole":
            return self._cartpole_env.get_detect_state(state_data)
        else:
            return self._spiel_env.get_detect_state(state_data)

    def get_legal_actions(self, info_state):
        try:
            return self._spiel_env.get_legal_actions(info_state)
        except:
            return [0, 1]

    def eval(self, average_policy, current_policy, policy_fn, nash_conv=False):
        cartpole_reward = self._cartpole_env.eval(policy_fn)
        spiel_exploitability = self._spiel_env.eval(average_policy, current_policy, policy_fn, nash_conv)
        return {"cartpole": cartpole_reward, "spiel": spiel_exploitability}
                