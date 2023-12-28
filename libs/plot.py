import pandas as pd
import plotly.express as px


def plot_values(**kwargs: list[float]):
    if len(kwargs) == 0:
        raise ValueError("No data to plot")
    length = len(list(kwargs.values())[0])
    for _i, value in enumerate(kwargs.values()):
        assert isinstance(value, list)
        assert len(value) == length
    df = pd.DataFrame(kwargs)
    fig = px.line(df, title="Loss")
    fig.show()
