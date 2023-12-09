import os
from pickle import load

from torch import nn, optim

from week1.process_train_data import TRAIN_DATA_PATH
from week1.vocab import DATA_DIR_PATH, VOCAB_FILE_PATH

EMBEDDING_DIM = 100
MODEL_FILE_PATH = os.path.join(DATA_DIR_PATH, "model.pt")


def one_hot(index, size):
    vector = [0] * size
    vector[index] = 1
    return vector


def main():
    vocab = load(open(VOCAB_FILE_PATH, "rb"))

    print("vocab loaded")

    class Model(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.embedding = nn.Embedding(len(vocab), EMBEDDING_DIM)
            self.linear = nn.Linear(EMBEDDING_DIM, len(vocab))
            self.softmax = nn.Softmax()

        def forward(self, input) -> None:
            return self.softmax(self.linear(self.embedding(input)))

    model = Model()
    criterion = nn.BCELoss(reduction="mean")
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    x_data, y_data = load(open(TRAIN_DATA_PATH, "rb"))

    print(f"train data loaded : x_data: {len(x_data)}, y_data: {len(y_data)}")

    y_data = [one_hot(index, len(vocab)) for index in y_data]

    print("training...")
    for epoch in range(100):
        for x, y in zip(x_data, y_data):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        print(f"epoch: {epoch}, loss: {loss}")

    torch.save(model.state_dict(), MODEL_FILE_PATH)


if __name__ == "__main__":
    main()
