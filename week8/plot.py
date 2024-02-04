import json
from typing import Any

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from week8.config import config


def plot_2d_values(
    x: list[str], y: list[str], values: list[list[float]], **kwargs: Any
):
    if len(values) == 0:
        raise ValueError("No data to plot")
    fig = px.imshow(values, x=x, y=y, **kwargs)
    fig.show()


def plot_hypothesis_results(result_path: str):
    with open(result_path, "r") as f:
        values = json.load(f)
    plot_2d_values(
        x=list(map(str, range(len(values)))),
        y=list(map(str, range(len(values))))[::-1],
        values=values[::-1],
        labels={"x": "LayerIndexFrom", "y": "LayerIndexTo", "color": "Accuracy"},
    )


def plot_overall_results():
    x = ["Antonyms", "EnEs", "LocationCountry"]
    antonyms_baseline_acc = 0.3710691823899371
    with open(config.antonyms_hypothesis_result_path, "r") as f:
        antonyms_hypothesis_results = json.load(f)
        antonyms_hypothesis_acc = np.amax(antonyms_hypothesis_results)
    antonyms_regular_acc = 0.9559748427672956

    en_es_baseline_acc = 0.30726256983240224
    with open(config.en_es_hypothesis_result_path, "r") as f:
        en_es_hypothesis_results = json.load(f)
        en_es_hypothesis_acc = np.amax(en_es_hypothesis_results)
    en_es_regular_acc = 0.6662011173184358

    location_country_baseline_acc = 0.1308016877637131
    with open(config.location_country_hypothesis_result_path, "r") as f:
        location_country_hypothesis_results = json.load(f)
        location_country_hypothesis_acc = np.amax(location_country_hypothesis_results)
    location_country_regular_acc = 0.7651195499296765

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            histfunc="sum",
            y=[
                antonyms_baseline_acc,
                en_es_baseline_acc,
                location_country_baseline_acc,
            ],
            x=x,
            name="Baseline",
        )
    )
    fig.add_trace(
        go.Histogram(
            histfunc="sum",
            y=[
                antonyms_hypothesis_acc,
                en_es_hypothesis_acc,
                location_country_hypothesis_acc,
            ],
            x=x,
            name="Hypothesis",
        )
    )
    fig.add_trace(
        go.Histogram(
            histfunc="sum",
            y=[antonyms_regular_acc, en_es_regular_acc, location_country_regular_acc],
            x=x,
            name="Regular",
        )
    )

    fig.update_layout(
        title_text="Overall Results",
        xaxis_title_text="Task",
        yaxis_title_text="Accuracy",
        bargap=0.2,
        bargroupgap=0.1,
    )

    fig.show()


if __name__ == "__main__":
    plot_hypothesis_results(config.antonyms_hypothesis_result_path)
    plot_hypothesis_results(config.en_es_hypothesis_result_path)
    plot_hypothesis_results(config.location_country_hypothesis_result_path)

    plot_overall_results()
