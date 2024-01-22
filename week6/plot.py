import os

from transformers import TrainerState
from transformers.trainer import TRAINER_STATE_NAME

from libs.plot import plot_values
from week6.config import config


def plot():
    state = TrainerState.load_from_json(
        os.path.join(config.my_checkpoint_path, TRAINER_STATE_NAME)
    )

    losses_per_step = [history["loss"] for history in state.log_history]
    plot_values(loss=losses_per_step)


if __name__ == "__main__":
    plot()
