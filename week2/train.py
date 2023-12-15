import time
from typing import Optional

import pandas as pd
import plotly.express as px
import torch
from torch import Tensor, cuda, nn, optim
from torch.utils.data import DataLoader

from libs.trainer import ClassificationTrainer
from week2.vocab import AGNewsDataset, load_vocab

device = "cuda:0" if cuda.is_available() else "cpu"


class RNNCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, out: bool = False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.w_xh = nn.Linear(input_size, hidden_size, bias=False)
        self.w_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out = out
        if self.out:
            self.w_hy = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bias = nn.Parameter(torch.randn(hidden_size))
        self.activation = nn.ReLU()
        # self.activation = nn.Tanh()

    def forward(self, x: Tensor, h: Tensor) -> tuple[Optional[Tensor], Tensor]:
        h = self.activation(self.w_xh(x) + self.w_hh(h) + self.bias)
        if self.out:
            return self.w_hy(h) + self.bias, h
        return None, h


class LSTMCell(nn.Module):
    """
    Not implemented for deep LSTM
    """

    def __init__(self, intput_size: int, hidden_size: int):
        super().__init__()
        self.input_size = intput_size
        self.hidden_size = hidden_size

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.w_xi = nn.Linear(intput_size, hidden_size, bias=False)
        self.w_hi = nn.Linear(hidden_size, hidden_size, bias=False)
        self.b_i = nn.Parameter(torch.randn(hidden_size))

        self.w_xf = nn.Linear(intput_size, hidden_size, bias=False)
        self.w_hf = nn.Linear(hidden_size, hidden_size, bias=False)
        self.b_f = nn.Parameter(torch.randn(hidden_size))

        self.w_xo = nn.Linear(intput_size, hidden_size, bias=False)
        self.w_ho = nn.Linear(hidden_size, hidden_size, bias=False)
        self.b_o = nn.Parameter(torch.randn(hidden_size))

        self.w_xc = nn.Linear(intput_size, hidden_size, bias=False)
        self.w_hc = nn.Linear(hidden_size, hidden_size, bias=False)
        self.b_c = nn.Parameter(torch.randn(hidden_size))

    def forward(self, x: Tensor, h: Tensor, c: Tensor) -> tuple[Tensor, Tensor]:
        # x size: (batch, input_size)
        # h size: (batch, hidden_size)
        f = self.sigmoid(self.w_xf(x) + self.w_hf(h) + self.b_f)
        i = self.sigmoid(self.w_xi(x) + self.w_hi(h) + self.b_i)
        o = self.sigmoid(self.w_xo(x) + self.w_ho(h) + self.b_o)

        c_tilde = self.tanh(self.w_xc(x) + self.w_hc(h) + self.b_c)

        c = (f * c) + (i * c_tilde)
        h = o * self.tanh(c)

        return h, c


class RNN(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int, sequence_size: int
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_size = sequence_size
        module_list: list[RNNCell] = []
        for idx in range(self.num_layers):
            kwargs = {"hidden_size": hidden_size}
            if idx == 0:
                kwargs["input_size"] = input_size
            else:
                kwargs["input_size"] = hidden_size

            if idx == self.num_layers - 1:
                kwargs["out"] = True

            module_list.append(RNNCell(**kwargs))
        self.layers = nn.ModuleList(module_list)
        self._device = "cpu"

    def to(self, device: str):
        self._device = device
        return super().to(device)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        batch_size = x.shape[0]

        if x.shape[1] != self.sequence_size:
            raise ValueError(
                "input vector should be sequence of vectors, which length is same as cells"
            )
        h_list: list[Tensor] = [None for _ in range(self.sequence_size)]
        for layer_idx, layer in enumerate(self.layers):
            for sequence_idx in range(self.sequence_size):
                cell = layer
                if sequence_idx == 0:
                    cell_h = torch.zeros(
                        batch_size, self.hidden_size, device=self._device
                    )
                if layer_idx == 0:
                    # shape of x: (batch_size, sequence_size, input_size)
                    cell_x = x[:, sequence_idx]
                else:
                    cell_x = h_list[sequence_idx]
                # print(layer_idx, sequence_idx)

                cell_y, cell_h = cell(cell_x, cell_h)

                h_list[sequence_idx] = cell_h

        # shape of call_y: (batch_size, hidden_size)
        # shape of h_lists: (batch_size, hidden_size)

        return cell_y, cell_h

    # def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
    #     y_list: list[Tensor] = []
    #     h_lists: list[Tensor] = []
    #     batch_size = x.shape[0]
    #     if x.shape[1] != self.sequence_size:
    #         raise ValueError(
    #             "input vector should be sequence of vectors, which length is same as cells"
    #         )
    #     for layer_idx, layer in enumerate(self.layers):
    #         h_list: list[Tensor] = []
    #         for sequence_idx in range(self.sequence_size):
    #             cell = layer
    #             if sequence_idx == 0:
    #                 cell_h = torch.zeros(batch_size, self.hidden_size)
    #                 if getattr(self, "_device", None) is not None:
    #                     cell_h = cell_h.to(self._device)
    #             if layer_idx == 0:
    #                 # shape of x: (batch_size, sequence_size, input_size)
    #                 cell_x = x[:, sequence_idx]
    #             else:
    #                 cell_x = h_lists[layer_idx - 1][sequence_idx]
    #             # print(layer_idx, sequence_idx)
    #             if layer_idx == self.sequence_size - 1:
    #                 cell_y, cell_h = cell(cell_x, cell_h, out=True)
    #                 y_list.append(cell_y)

    #             else:
    #                 _, cell_h = cell(cell_x, cell_h)
    #             h_list.append(cell_h)
    #         h_lists.append(torch.stack(h_list, 0))
    #     # shape of y_list: (sequence_size, batch_size, hidden_size)
    #     # shape of h_lists: (num_layers, sequence_size, batch_size, hidden_size)
    #     return torch.stack(y_list, 0).transpose(0, 1), torch.stack(h_lists, 0)[
    #         :, -1
    #     ].transpose(0, 1)


class LSTM(nn.Module):
    """
    Not implemented for deep LSTM, just for forwarding
    """

    def __init__(self, input_size: int, hidden_size: int, sequence_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sequence_size = sequence_size
        self.lstm = LSTMCell(input_size, hidden_size)
        self._device = "cpu"

    def to(self, device: str):
        self._device = device
        return super().to(device)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        batch_size = x.shape[0]
        if x.shape[1] != self.sequence_size:
            raise ValueError(
                "input vector should be sequence of vectors, which length is same as cells"
            )
        for sequence_idx in range(self.sequence_size):
            if sequence_idx == 0:
                cell_h = torch.zeros(batch_size, self.hidden_size, device=self._device)
                cell_c = torch.zeros(batch_size, self.hidden_size, device=self._device)
            cell_x = x[:, sequence_idx]
            cell_y, cell_c = self.lstm(cell_x, cell_h, cell_c)

        return cell_y, cell_c


class TextRNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_classes: int,
        hidden_size: int,
        num_layers: int,
        dropout_p: float,
        sequence_size: int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = RNN(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            sequence_size=sequence_size,
        )
        self.dropout = nn.Dropout(dropout_p)
        self.linear = nn.Linear(hidden_size, num_classes)

    def _rnn_to(self, device: str):
        return self.rnn.to(device)

    def to(self, device: str):
        self._rnn_to(device)
        return super().to(device)

    def forward(self, x: Tensor) -> Tensor:
        # print(x.shape)
        x = self.embedding(x)
        # print(x.shape)
        x, _ = self.rnn(x)
        # print(x.shape)
        # x = x[:, -1, :]
        # print(x.shape)
        x = self.dropout(x)
        # print(x.shape)
        x = self.linear(x)
        # print(x.shape)
        # raise Exception
        return x


class TextLSTM(TextRNN):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_classes: int,
        hidden_size: int,
        num_layers: int,
        dropout_p: float,
        sequence_size: int,
    ):
        super().__init__(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_classes=num_classes,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout_p=dropout_p,
            sequence_size=sequence_size,
        )
        self.rnn = LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            sequence_size=sequence_size,
        )


def save_model(model: nn.Module, path: str):
    torch.save(model.state_dict(), path)


def train_rnn():
    vocab = load_vocab(min=10, max=10000)
    model = TextRNN(
        vocab_size=len(vocab),
        embed_dim=300,
        num_classes=4,
        hidden_size=128,
        num_layers=1,
        dropout_p=0.2,
        sequence_size=50,
    )
    model_path = "week2/data/rnn_model"

    model.to(device)
    print(model)

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

    trainer = ClassificationTrainer(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
    )

    train_losses, test_losses = trainer.run(
        num_epoch=30, device=device, model_save_path=model_path, verbose=True
    )

    plot_loss(train_losses, test_losses)


def train_lstm():
    vocab = load_vocab(min=10, max=10000)
    model = TextLSTM(
        vocab_size=len(vocab),
        embed_dim=300,
        num_classes=4,
        hidden_size=128,
        num_layers=1,
        dropout_p=0.2,
        sequence_size=50,
    )
    model_path = "week2/data/lstm_model"

    model.to(device)
    print(model)

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

    trainer = ClassificationTrainer(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
    )

    train_losses, test_losses = trainer.run(
        num_epoch=30, device=device, model_save_path=model_path, verbose=False
    )

    plot_loss(train_losses, test_losses)


def plot_loss(train_losses: list[float], test_losses: list[float]):
    df = pd.DataFrame({"train": train_losses, "test": test_losses})
    fig = px.line(df, title="Loss")
    fig.show()


if __name__ == "__main__":
    # train_rnn()
    train_lstm()
