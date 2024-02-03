import json
from typing import Any

import plotly.express as px

from week8.config import config


def plot_2d_values(
    x: list[str], y: list[str], values: list[list[float]], **kwargs: Any
):
    if len(values) == 0:
        raise ValueError("No data to plot")
    fig = px.imshow(values, x=x, y=y, **kwargs)
    fig.show()


def plot_results(result_path: str):
    with open(result_path, "r") as f:
        values = json.load(f)
    plot_2d_values(
        x=list(map(str, range(len(values)))),
        y=list(map(str, range(len(values))))[::-1],
        values=values[::-1],
        labels={"x": "LayerIndexFrom", "y": "LayerIndexTo", "color": "Accuracy"},
    )


if __name__ == "__main__":
    plot_results(config.antonyms_hypothesis_result_path)
    # plot_results(config.en_es_hypothesis_result_path)
    # plot_results(config.location_country_hypothesis_result_path)
