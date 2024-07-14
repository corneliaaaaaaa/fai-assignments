import json
from game.game import setup_config, start_poker
from src.call_player import setup_ai as call_ai
from src.random_player import setup_ai as random_ai
from src.allIn_player import setup_ai as allin_ai
from src.console_player import setup_ai as console_ai
from src.rational_player import setup_ai as rational_ai
from src.agent import setup_ai as final_ai
from src.ml_player import setup_ai as dl_ai

from baseline0 import setup_ai as baseline0_ai
from baseline1 import setup_ai as baseline1_ai
from baseline2 import setup_ai as baseline2_ai
from baseline3 import setup_ai as baseline3_ai
from baseline4 import setup_ai as baseline4_ai
from baseline5 import setup_ai as baseline5_ai
from baseline6 import setup_ai as baseline6_ai
from baseline7 import setup_ai as baseline7_ai


def run_game_with_algorithm(baseline_ai, my_ai):
    config = setup_config(max_round=20, initial_stack=1000, small_blind_amount=5)
    config.register_player(name="p1", algorithm=baseline_ai)
    config.register_player(name="p2", algorithm=my_ai)
    # config.register_player(name="p3", algorithm=baseline3_ai())
    # config.register_player(name="p4", algorithm=baseline4_ai())
    # config.register_player(name="p5", algorithm=baseline5_ai())
    # config.register_player(name="p6", algorithm=baseline2_ai())
    game_result = start_poker(config, verbose=1)
    print(json.dumps(game_result, indent=4))


baseline_algorithms = [
    baseline0_ai,
    baseline1_ai,
    baseline2_ai,
    baseline3_ai,
    baseline4_ai,
    baseline5_ai,
    baseline6_ai,
    baseline7_ai,
]
my_ai = final_ai   #dl_ai #rational_ai
modes = ["allBaselines", "oneBaseline", "interactive"]
selectedModeID = 0
current = modes[selectedModeID]
selectedBaselineID = 7
current_baseline = baseline_algorithms[selectedBaselineID]


if current == "allBaselines":
    for i, baseline_ai in enumerate(baseline_algorithms):
        print(f"Running game with {baseline_ai.__name__}")
        run_game_with_algorithm(baseline_ai(), my_ai())
        print(f"Baseline {i} is done testing!")
        input("Press Enter to continue to the next game...")
    print("All games have been run.")
elif current == "oneBaseline":
    config = setup_config(max_round=20, initial_stack=1000, small_blind_amount=5)
    config.register_player(name="p1", algorithm=current_baseline())
    config.register_player(name="p2", algorithm=my_ai())
    game_result = start_poker(config, verbose=1)
    print(f"Baseline {selectedBaselineID} is done testing!")

    print(json.dumps(game_result, indent=4))
elif current == "interactive":
    config = setup_config(max_round=20, initial_stack=1000, small_blind_amount=5)
    config.register_player(name="p1", algorithm=current_baseline())
    config.register_player(name="me", algorithm=console_ai())
    game_result = start_poker(config, verbose=1)

    print(json.dumps(game_result, indent=4))
