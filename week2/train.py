import pandas as pd
import plotly.express as px
import torch
from torch import cuda, nn, optim
from torch.utils.data import DataLoader

from week2.vocab import AGNewsDataset, load_vocab

device = "cuda:0" if cuda.is_available() else "cpu"


class TextRNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_classes: int,
        hidden_size: int,
        num_layers: int,
        dropout_p: float,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # print(x.shape)
        x = self.embedding(x)
        # print(x.shape)
        x, _ = self.rnn(x)
        # print(x.shape)
        x = x[:, -1, :]
        # print(x.shape)
        x = self.dropout(x)
        # print(x.shape)
        x = self.linear(x)
        # print(x.shape)
        # raise Exception
        return x


def train():
    vocab = load_vocab(min=10, max=10000)
    model = TextRNN(
        vocab_size=len(vocab),
        embed_dim=300,
        num_classes=4,
        hidden_size=128,
        num_layers=2,
        dropout_p=0.2,
    )

    model.to(device)

    dataset = AGNewsDataset(vocab, sentence_len=50)
    train_loader = DataLoader(
        dataset=dataset, batch_size=32, shuffle=True, num_workers=0
    )

    test_dataset = AGNewsDataset(vocab, sentence_len=50, is_train=False)
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=32, shuffle=False, num_workers=0
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    train_losses = []
    test_losses = []

    for epoch in range(30):
        model.train()
        total_loss = 0
        total_correct = 0
        for i, data in enumerate(train_loader):
            x, y = data
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += (outputs.argmax(1) == y.argmax(1)).sum().item()

            # if i % 100 == 0:
            #     print(f"loss: {loss.item()}")
        train_loss_mean = total_loss / len(train_loader.dataset)
        train_losses.append(train_loss_mean)
        print(f"=====epoch: {epoch}=====")
        print(
            f"train accuracy: {total_correct / len(train_loader.dataset)}, train loss: {train_loss_mean}"
        )
        model.eval()
        test_loss = 0
        test_correct = 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                test_loss += loss.item()
                test_correct += (outputs.argmax(1) == y.argmax(1)).sum().item()
        test_loss_mean = test_loss / len(test_loader.dataset)
        test_losses.append(test_loss_mean)
        print(
            f"test accuracy: {test_correct / len(test_loader.dataset)}, test loss: {test_loss_mean}"
        )
        print("==========")

    plot_loss(train_losses, test_losses)


def plot_loss(train_losses, test_losses):
    df = pd.DataFrame({"train": train_losses, "test": test_losses})
    fig = px.line(df, title="Loss")
    fig.show()


if __name__ == "__main__":
    train()
