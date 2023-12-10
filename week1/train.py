import os
from functools import partial
from pickle import load

import torch
from torch import cuda, nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchtext.datasets import WikiText103

from week1.process import BATCH_SIZE, one_hot_to_idx, process
from week1.vocab import DATA_DIR_PATH, VOCAB_FILE_PATH, VOCAB_FREQ_PATH

EMBEDDING_DIM = 1000
MODEL_FILE_PATH = os.path.join(DATA_DIR_PATH, "model")

device = "cuda" if cuda.is_available() else "cpu"


class Model(nn.Module):
    def __init__(self, len_vocab: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.embedding = nn.Embedding(len_vocab, EMBEDDING_DIM, max_norm=1.0)
        self.linear = nn.Linear(EMBEDDING_DIM, len_vocab)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input) -> None:
        # print(input[:3])
        # print(input.shape)
        x = self.embedding(input)

        x = torch.mean(x, dim=1)

        x = x.reshape(-1, x.shape[2])

        # print(x.shape)
        x = self.linear(x)
        # print(x.shape)

        result = self.softmax(x)
        # print(result.shape)
        # print(result.shape)
        # print(result[:3])
        # sum of all
        # print(result[0:3].sum(dim=2))
        # print([one_hot_to_idx(result[0]) for result in result[:3]])
        return result


def main():
    vocab = load(open(VOCAB_FILE_PATH[3], "rb"))
    voca_with_freq = load(open(VOCAB_FREQ_PATH, "rb"))

    print("vocab loaded")

    model = Model(len_vocab=len(vocab))
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e4)

    print("model initialized")

    train, test = WikiText103(root=DATA_DIR_PATH, split=("train", "test"))

    print("dataset loaded")

    train_loader = DataLoader(
        dataset=list(iter(train)),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=partial(process, vocab=vocab, voca_with_freq=voca_with_freq),
    )

    print("training...")
    # model.load_state_dict(torch.load(MODEL_FILE_PATH + "_2_8800"))
    for epoch in range(100):
        # for idx, (x, y) in enumerate(train_loader):
        model_path = MODEL_FILE_PATH + f"_{epoch}"
        for idx, (x, y) in enumerate(train_loader):
            model_batch_path = model_path + f"_{idx}"
            x, y = Variable(torch.from_numpy(x)).to(torch.int).to(device), Variable(
                torch.from_numpy(y)
            ).to(torch.float).to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            print("===========================")
            print([one_hot_to_idx(y_) for y_ in y[:3]])
            print([one_hot_to_idx(output_) for output_ in output[:3]])
            print(
                sum(
                    [
                        y__ == output__
                        for y__, output__ in zip(
                            [one_hot_to_idx(y_) for y_ in y],
                            [one_hot_to_idx(output_) for output_ in output],
                        )
                    ]
                )
            )
            print(y.shape)
            print("===========================")
            # print(x[:3])
            # print(
            #     [
            #         idx
            #         for idx, value in enumerate(model.embedding.weight.grad)
            #         if value.sum() != 0
            #     ]
            # )
            # print(model.linear.weight.grad)
            # print("===========================")
            # print(model.embedding.weight[-1])
            # print(model.embedding.weight.shape)
            # print(model.embedding.weight.grad)
            # print("===========================")
            optimizer.step()
            print(f"batch idx: {idx} epoch: {epoch}, loss: {loss}")
            if idx % 100 == 0:
                torch.save(model.state_dict(), model_batch_path)
                print(f"batch idx: {idx} epoch: {epoch}, saved model")
        torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    main()
