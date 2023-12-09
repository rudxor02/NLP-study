import os
from pickle import dump

from torchtext.data.utils import get_tokenizer
from torchtext.datasets import WikiText103

DATA_DIR_PATH = os.path.join(os.path.dirname(__file__), "data")
VOCAB_FILE_PATH = os.path.join(DATA_DIR_PATH, "vocab.pkl")


def main():
    train, test = WikiText103(root=DATA_DIR_PATH, split=("train", "test"))

    print("data loaded")

    tokenizer = get_tokenizer("basic_english")
    vocab = {"<unk>": 0, "<pad>": 1}
    for data in iter(train):
        for token in tokenizer(data):
            if token not in vocab:
                vocab[token] = len(vocab)

    with open(VOCAB_FILE_PATH, "wb") as f:
        dump(vocab, f)

    print("vocab saved")


if __name__ == "__main__":
    main()
