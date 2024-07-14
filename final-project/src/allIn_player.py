from game.players import BasePokerPlayer
import random as rand


class AllInPlayer(BasePokerPlayer):
    def __init__(self):
        rand.seed(729)

    def declare_action(self, valid_actions, hole_card, round_state):
        # Always choose the "raise" action with the maximum amount
        for action in valid_actions:
            if action["action"] == "raise":
                return "raise", action["amount"]["max"]
            elif action["action"] == "call":
                return action["action"], action["amount"]
        # If no valid actions are found (which should not happen), return "fold"
        return "fold", 0

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, new_action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass


def setup_ai():
    return AllInPlayer()
