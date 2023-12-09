import os
from pickle import dump, load

from torchtext.data.utils import get_tokenizer
from torchtext.datasets import WikiText103

from week1.vocab import DATA_DIR_PATH, VOCAB_FILE_PATH

TRAIN_DATA_PATH = os.path.join(DATA_DIR_PATH, "train.pickle")
CBOW_WINDOW_SIZE = 5

tokenizer = get_tokenizer("basic_english")


def word_to_token(vocab, text):
    try:
        return vocab[text]
    except KeyError:
        return vocab["<unk>"]


def main():
    train, test = WikiText103(root=DATA_DIR_PATH, split=("train", "test"))

    print("data loaded")

    vocab = load(open(VOCAB_FILE_PATH, "rb"))

    batch_train_input, batch_train_target = [], []

    for data in iter(train):
        tokens = tokenizer(data)

        if len(tokens) < CBOW_WINDOW_SIZE * 2 + 1:
            continue
        for i in range(CBOW_WINDOW_SIZE, len(tokens) - CBOW_WINDOW_SIZE):
            train_input = [
                word_to_token(vocab, token)
                for token in tokens[i - CBOW_WINDOW_SIZE : i]
            ] + [
                word_to_token(vocab, token)
                for token in tokens[i + 1 : i + CBOW_WINDOW_SIZE + 1]
            ]
            batch_train_input.append(train_input)
            batch_train_target.append(word_to_token(vocab, tokens[i]))

    with open(TRAIN_DATA_PATH, "wb") as f:
        dump((batch_train_input, batch_train_target), f)

    print("data processed")


if __name__ == "__main__":
    main()
