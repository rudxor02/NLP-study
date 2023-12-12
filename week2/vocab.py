import os
from pickle import dump, load

import pandas as pd
import plotly.express as px
from torch import IntTensor, Tensor
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import AG_NEWS

DATA_DIR_PATH = os.path.join(os.path.dirname(__file__), "data")
VOCAB_FREQ_PATH = os.path.join(DATA_DIR_PATH, "vocab.pkl")


def load_data() -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    """
    Load AG_NEWS dataset, which is a dataset for topic classification.

    Each sample is a tuple of (label, text).

    Label is an integer between 1 and 4 : (“World”, “Sports”, “Business”, “Sci/Tech”)

    Returns train(120000), test(7600)
    """
    train, test = AG_NEWS(root=DATA_DIR_PATH, split=("train", "test"))
    return list(train), list(test)


def index_to_word(vocab: dict[str, int], idx: int) -> str:
    return list(vocab.keys())[list(vocab.values()).index(idx)]


def word_to_index(vocab: dict[str, int], word: str) -> int:
    try:
        return vocab[word]
    except KeyError:
        return vocab["<unk>"]


class AGNewsDataset(Dataset):
    def __init__(self, vocab: dict[str, int], sentence_len: int, is_train: bool = True):
        super().__init__()
        train, test = load_data()
        tokenizer = get_tokenizer("basic_english")
        data_list: list[tuple[int, str]] = train if is_train else test
        self.pad_idx = word_to_index(vocab, "<pad>")
        self.sentence_len = sentence_len
        self.len = len(data_list)
        self.y_data: list[int] = [data[0] for data in data_list]
        tokens_list: list[list[str]] = [tokenizer(data[1]) for data in data_list]
        self.x_data: list[list[int]] = [
            [word_to_index(vocab, token) for token in tokens] for tokens in tokens_list
        ]
        self.y_onehot: list[Tensor] = [
            Tensor([1 if i == y else 0 for i in range(4)]) for y in self.y_data
        ]

    #     self._preprocess()

    # def _preprocess(self):
    #     for i in range(self.len):
    #         if len(self.x_data[i]) < self.sentence_len:
    #             self.x_data[i] += [self.pad_idx] * (
    #                 self.sentence_len - len(self.x_data[i])
    #             )
    #         if len(self.x_data[i]) > self.sentence_len:
    #             self.x_data[i] = self.x_data[i][: self.sentence_len]

    def __getitem__(self, index: int) -> tuple[IntTensor, Tensor]:
        x_data = self.x_data[index]

        if len(x_data) < self.sentence_len:
            x_data = [self.pad_idx] * (self.sentence_len - len(x_data)) + x_data
        if len(x_data) > self.sentence_len:
            x_data = x_data[: self.sentence_len]
        return IntTensor(x_data), self.y_onehot[index]

    def __len__(self):
        return self.len


def analyze(data_list: list[tuple[int, str]]):
    """
    ```text
    =====test data analyze=====
    label_count: [1900, 1900, 1900, 1900]
    max_len: 161
    min_len: 14
    vocab with freq saved 25272
    vocab_0_10000 saved 25269
    vocab_10_10000 saved 3488
    vocab_30_10000 saved 1348
    vocab_50_10000 saved 808
    =====train data analyze=====
    label_count: [30000, 30000, 30000, 30000]
    max_len: 207
    min_len: 12
    vocab with freq saved 95812
    vocab_0_10000 saved 95771
    vocab_10_10000 saved 19466
    vocab_30_10000 saved 10600
    vocab_50_10000 saved 7796
    ```
    """
    tokenizer = get_tokenizer("basic_english")
    label_count = [0, 0, 0, 0]
    max_len = 0
    min_len = 1000000
    count_under_50 = 0
    count_under_100 = 0
    vocab_with_freq: dict[str, list[int]] = {"<unk>": [0, 50], "<pad>": [1, 50]}
    for data in data_list:
        max_len = max(max_len, len(tokenizer(data[1])))
        min_len = min(min_len, len(tokenizer(data[1])))
        if len(tokenizer(data[1])) < 50:
            count_under_50 += 1
        if len(tokenizer(data[1])) < 100:
            count_under_100 += 1
        label_count[data[0] - 1] += 1
        for token in tokenizer(data[1]):
            if token not in vocab_with_freq.keys():
                vocab_with_freq[token] = [len(vocab_with_freq), 0]
            else:
                vocab_with_freq[token][1] += 1
    print(f"label_count: {label_count}")
    print(f"max_len: {max_len}")
    print(f"min_len: {min_len}")
    print(f"count_under_50: {count_under_50}")
    print(f"count_under_100: {count_under_100}")
    with open(VOCAB_FREQ_PATH, "wb") as f:
        dump(vocab_with_freq, f)
    print(f"vocab with freq saved {len(vocab_with_freq)}")

    for min_, max_ in [(0, 10000), (10, 10000), (30, 10000), (50, 10000)]:
        vocab = {
            k: v[0]
            for k, v in vocab_with_freq.items()
            if (min_ <= v[1] <= max_) or k in ["<unk>", "<pad>"]
        }

        vocab = {k: i for i, k in enumerate(vocab.keys())}

        with open(os.path.join(DATA_DIR_PATH, f"vocab_{min_}_{max_}.pkl"), "wb") as f:
            dump(vocab, f)
        print(f"vocab_{min_}_{max_} saved {len(vocab)}")


def plot_freq(vocab_with_freq: dict[str, list[int]]):
    idx_with_freq = vocab_with_freq.values()
    idx_with_freq = sorted(idx_with_freq, key=lambda x: x[1], reverse=True)
    for i in range(len(idx_with_freq)):
        idx_with_freq[i][0] = i

    df = pd.DataFrame(idx_with_freq, columns=["idx", "freq"])
    fig = px.line(df, x="idx", y="freq", title="Frequency of words")
    fig.show()


def load_vocab(min: int, max: int) -> dict[str, int]:
    with open(os.path.join(DATA_DIR_PATH, f"vocab_{min}_{max}.pkl"), "rb") as f:
        vocab = load(f)
    return vocab


def load_vocab_with_freq() -> dict[str, list[int]]:
    with open(VOCAB_FREQ_PATH, "rb") as f:
        vocab_with_freq = load(f)
    return vocab_with_freq


if __name__ == "__main__":
    # train, test = load_data()
    # print("=====test data analyze=====")
    # analyze(test)
    # print("=====train data analyze=====")
    # analyze(train)
    vocab_with_freq = load_vocab_with_freq()
    plot_freq(vocab_with_freq)
