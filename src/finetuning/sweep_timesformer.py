# sweep_timesformer.py
import itertools

from fine_tune import run_experiment  

BALANCE_MODES = ["none", "min", "2min_flip", "max_full"]
FRAME_SAMPLINGS = ["first", "center", "even", "random"]
NUM_UNFROZEN_LAYERS_LIST = [2, 4, 6]

if __name__ == "__main__":
    for balance_mode, frame_sampling, num_unfrozen_layers in itertools.product(
        BALANCE_MODES,
        FRAME_SAMPLINGS,
        NUM_UNFROZEN_LAYERS_LIST,
    ):
        print(
            f"\n\n========== New run ==========\n"
            f"BALANCE_MODE           = {balance_mode}\n"
            f"FRAME_SAMPLING         = {frame_sampling}\n"
            f"NUM_UNFROZEN_LAYERS = {num_unfrozen_layers}\n"
        )

        run_experiment(
            balance_mode=balance_mode,
            frame_sampling=frame_sampling,
            num_unfrozen_layers=num_unfrozen_layers,
        )
