from json import load

from libs.plot import plot_values


def plot():
    losses: list[float] = []

    with open("week4/data/loss.v1.1.json", "r") as f:
        losses += load(f)

    with open("week4/data/loss.json", "r") as f:
        losses += load(f)

    plot_values(train=losses)


if __name__ == "__main__":
    plot()
