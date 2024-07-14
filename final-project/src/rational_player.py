import numpy as np
from game.players import BasePokerPlayer
from .pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate

NB_SIMULATION = 2000

def calculate_raise_amount(win_rate, amount_range, lower_bound):
        """
        Scale the raise amount based on the win rate - lower_bound.
        - If win rate - lower_bound is 0, raise min_raise
        - If win rate - lower_bound is 1, raise max_raise
        Otherwise, interpolate between min_raise and max_raise
        """
        min_raise = amount_range["min"]
        max_raise = amount_range["max"]
        if min_raise == -1 or max_raise == -1:
            return -1

        return min_raise + (max_raise - min_raise) * (win_rate - lower_bound)

def find_stack(id, seats, mine=True):
    for seat in seats:
        if mine and seat["uuid"] == id:
            return seat["stack"]
        if not mine and seat["uuid"] != id:
            return seat["stack"]
    
    return -1

class RationalPlayer(BasePokerPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        community_card = round_state["community_card"]
        win_rate = estimate_hole_card_win_rate(
            nb_simulation=NB_SIMULATION,
            nb_player=self.nb_player,
            hole_card=gen_cards(hole_card),
            community_card=gen_cards(community_card),
        )
        lower_bound = 0.8 / self.nb_player

        if win_rate >= 1.6 / self.nb_player and len(valid_actions) == 3:
            action = valid_actions[2]
            raise_amount = calculate_raise_amount(win_rate, action["amount"], lower_bound)
            if raise_amount == -1:
                action = valid_actions[1] # if raise is invalid, call instead
            else:
                action["amount"] = raise_amount # raise
        elif win_rate >= 0.2 / self.nb_player:
            action = valid_actions[1]   # call
        else:
            action = valid_actions[0]  # fold
        
        return action["action"], action["amount"]

    def receive_game_start_message(self, game_info):
        self.nb_player = game_info['player_num']

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

class SmartRationalPlayer(BasePokerPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        community_card = round_state["community_card"]
        win_rate = estimate_hole_card_win_rate(
            nb_simulation=NB_SIMULATION,
            nb_player=self.nb_player,
            hole_card=gen_cards(hole_card),
            community_card=gen_cards(community_card),
        )
        lower_bound = 1.2 / self.nb_player
        my_stack = find_stack(self.uuid, round_state["seats"], True)
        current_pot = round_state["pot"]["main"]["amount"]
            
        # print(f"================ win rate: {win_rate} ==================")
        # print("valid actions   ", valid_actions)
        # print("hole card         ", hole_card)

        # if I have a much larger score than the opponent, I can fold from now on
        if self.nb_player == 2 and round_state["street"] == "preflop":
            opponent_stack = find_stack(self.uuid, round_state["seats"], False)
            rounds_left = 20 - round_state["round_count"] + 1
            if my_stack - opponent_stack > rounds_left * 10 + 20: #TODO: buffer for precision loss
                action = valid_actions[0]
                return action["action"], action["amount"]

        if win_rate >= 1.6 / self.nb_player and len(valid_actions) == 3:
            action = valid_actions[2]
            raise_amount = calculate_raise_amount(win_rate, action["amount"], lower_bound)
            if raise_amount == -1:
                action = valid_actions[1] # if raise is invalid, call instead
            else:
                action["amount"] = raise_amount # raise
        elif win_rate >= 1.4 / self.nb_player:
            action = valid_actions[2]
            raise_amount = calculate_raise_amount(win_rate, action["amount"], lower_bound)
            if raise_amount == -1:
                if valid_actions[1]["amount"] < 200:
                    action = valid_actions[1] # if raise is invalid and call amount not too big, call instead
                else:
                    action = valid_actions[0] # if call amount is too big, fold
            elif raise_amount > 200 or current_pot > 200 or valid_actions[1]["amount"] == 0: #TODO: make it smaller? 
                action = valid_actions[1] # if raise is too big, just call
            else:
                action["amount"] = raise_amount # raise
        elif win_rate >= 1.2 / self.nb_player:
            action = valid_actions[1]
            if action["amount"] > 200:# or current_pot > 200: #TODO
                action = valid_actions[0] # if call amount is too big, fold
        elif win_rate >= 1.0 / self.nb_player:
            if valid_actions[1]["amount"] == 0:
                action = valid_actions[1]  # call
            elif round_state["street"] == "river": 
                action = valid_actions[1]  # call
            elif current_pot < 200 and valid_actions[1]["amount"] < 100:
                action = valid_actions[1]  # call
            else:
                action = valid_actions[0]  # fold
        elif win_rate >= 0.4 / self.nb_player:
            if valid_actions[1]["amount"] == 0:
                action = valid_actions[1]  # call
            elif (round_state["street"] == "river" or current_pot < 60) \
                and valid_actions[1]["amount"] < 50:
                action = valid_actions[1]  # call
            else:
                action = valid_actions[0]  # fold
        else:
            if valid_actions[1]["amount"] == 0:  
                action = valid_actions[1] # call
            else:   
                action = valid_actions[0]  # fold
        
        return action["action"], action["amount"]

    def receive_game_start_message(self, game_info):
        self.nb_player = game_info['player_num']

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass


class AggressiveRationalPlayer(BasePokerPlayer):

    def declare_action(self, valid_actions, hole_card, round_state):
        community_card = round_state["community_card"]
        win_rate = estimate_hole_card_win_rate(
            nb_simulation=NB_SIMULATION,
            nb_player=self.nb_player,
            hole_card=gen_cards(hole_card),
            community_card=gen_cards(community_card),
        )
        lower_bound = 0.8 / self.nb_player
        my_stack = find_stack(self.uuid, round_state["seats"], True)

        # if I have a much larger score than the opponent, I can fold from now on
        if self.nb_player == 2 and round_state["street"] == "preflop":
            opponent_stack = find_stack(self.uuid, round_state["seats"], False)
            rounds_left = 20 - round_state["round_count"] + 1
            if my_stack - opponent_stack > rounds_left * 10 + 20:
                action = valid_actions[0]
                return action["action"], action["amount"]

        if win_rate >= 1.6 / self.nb_player and len(valid_actions) == 3:
            action = valid_actions[2]
            raise_amount = calculate_raise_amount(win_rate, action["amount"], lower_bound)
            if raise_amount == -1:
                action = valid_actions[1] # if raise is invalid, call instead
            else:
                action["amount"] = raise_amount # raise
        elif win_rate >= 1.2 / self.nb_player:
            action = valid_actions[2]
            raise_amount = calculate_raise_amount(win_rate, action["amount"], lower_bound)
            if raise_amount == -1:
                action = valid_actions[1] # if raise is invalid, call instead
            else:
                action["amount"] = raise_amount # raise
        elif win_rate >= 1.0 / self.nb_player and valid_actions[1]["amount"] < 100:
            action = valid_actions[1]  # fetch CALL action info
        elif win_rate >= 0.2 / self.nb_player and valid_actions[1]["amount"] < 100:
            action = valid_actions[1]  # fetch CALL action info
        else:
            action = valid_actions[0]  # fetch FOLD action info
        
        return action["action"], action["amount"]

    def receive_game_start_message(self, game_info):
        self.nb_player = game_info['player_num']

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

class ConservativeRationalPlayer(BasePokerPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        community_card = round_state["community_card"]
        win_rate = estimate_hole_card_win_rate(
            nb_simulation=NB_SIMULATION,
            nb_player=self.nb_player,
            hole_card=gen_cards(hole_card),
            community_card=gen_cards(community_card),
        )
        lower_bound = 1.0 / self.nb_player
        my_stack = find_stack(self.uuid, round_state["seats"], True)

        # if I have a much larger score than the opponent, I can fold from now on
        if self.nb_player == 2 and round_state["street"] == "preflop":
            opponent_stack = find_stack(self.uuid, round_state["seats"], False)
            rounds_left = 20 - round_state["round_count"] + 1
            if my_stack - opponent_stack > rounds_left * 10 + 20:
                action = valid_actions[0]
                return action["action"], action["amount"]

        if win_rate >= 1.8 / self.nb_player:
            action = valid_actions[2]  # fetch RAISE action info
            raise_amount = calculate_raise_amount(win_rate, action["amount"], lower_bound)
            if raise_amount == -1:
                action = valid_actions[1] # if raise is invalid, call instead
            else:
                action["amount"] = raise_amount # raise
        elif win_rate >= 1.0 / self.nb_player and len(valid_actions) == 3:
            roll = np.random.randint(2)
            if roll == 1:
                action = valid_actions[2]  # fetch RAISE action info
                raise_amount = calculate_raise_amount(win_rate, action["amount"], lower_bound)
                if raise_amount == -1:
                    action = valid_actions[1] # if raise is invalid, call instead
                else:
                    action["amount"] = raise_amount # raise
            else:
                action = valid_actions[1]   # fetch CALL action info
        elif win_rate >= 0.4 / self.nb_player:
            action = valid_actions[1]   # fetch CALL action info
        else:
            action = valid_actions[0]  # fetch FOLD action info

        return action["action"], action["amount"]

    def receive_game_start_message(self, game_info):
        self.nb_player = game_info['player_num']

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

def setup_ai():
    return SmartRationalPlayer()
    # return AggressiveRationalPlayer()
    # return ConservativeRationalPlayer()
    