import random
import numpy as np

class WeightedRPS():

    def __init__(self):
            self.players = [0, 1]
            self._state = ""
            self._done = False
            self.rewards = [[0, -1, 1], #R
                            [1, 0, -2], #P
                            [-1, 2, 0]] #S

    def set_state(self, state):
        self._done = False
        self._state = state

    def step(self, action, policy, player):
        if len(self._state) == 2:
            raise Exception("Play has terminated. Please reset the environment.")
        
        self._state = self._state + str(action)
        
        if player == 0:
            sampled_action = policy(self.details(self._state))
            self._state = self._state + str(sampled_action)

        self._done = True
        first_action, second_action = int(self._state[0]), int(self._state[1])

        if player == 0:
            return (self._state, self.rewards[first_action][second_action], self._done, None)
        else:
            return (self._state, -1 * self.rewards[first_action][second_action], self._done, None)
        
    def reset(self, player, policy):
        self._done = False
        self._state = ""
            
        if player == 1:
            sampled_action = policy(self.details(self._state))
            self._state = self._state + str(sampled_action)

        return (self._state, 0, self._done, self.details(self._state))

    def details(self, state):
        if state == "":
            return (0, "info_one", [0, 1, 2])
        else:
            return (1, "info_two", [0, 1, 2])

    def get_detect_state(self, state):
        return state

    def get_legal_actions(self, info_state):
        return [0, 1, 2]

    def eval(self, average_policy, _current_policy, _policy_fn, nash_conv=False):

        try:
            fra = average_policy['info_one'][2] - average_policy['info_one'][1]
            fpa = average_policy['info_one'][0] - 2 * average_policy['info_one'][1]
            fsa = 2 * average_policy['info_one'][1] - average_policy['info_one'][0]

            sra = average_policy['info_two'][2] - average_policy['info_two'][1]
            spa = average_policy['info_two'][0] - 2 * average_policy['info_two'][1]
            ssa = 2 * average_policy['info_two'][1] - average_policy['info_two'][0]

            return max(fra, fpa, fsa) + max(sra, spa, ssa)
        except:
            return "Average policy incomplete..."
                