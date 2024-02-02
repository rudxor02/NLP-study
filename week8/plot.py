import plotly.express as px


def plot_2d_values(x: list[str], y: list[str], values: list[list[float]]):
    if len(values) == 0:
        raise ValueError("No data to plot")
    fig = px.imshow(values, x=x, y=y)
    fig.show()


if __name__ == "__main__":
    values = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    plot_2d_values(
        x=list(map(str, range(len(values)))),
        y=list(map(str, range(len(values))))[::-1],
        values=values,
    )
