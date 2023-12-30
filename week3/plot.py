from libs.plot import plot_values


def plot():
    # from train.py...
    train_loss = [
        4.434931715133874,
        3.464883699623043,
        3.3355369702711766,
        3.2708003271957136,
        3.229737990746484,
        3.200551265650186,
        3.1781104092204244,
        3.1605218935797055,
        3.145285995434957,
        3.132373110656905,
    ]
    plot_values(train_loss=train_loss)


if __name__ == "__main__":
    plot()
