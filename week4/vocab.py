import os

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.implementations import ByteLevelBPETokenizer

from week4.config import config

DATA_DIR_PATH = os.path.join(os.path.dirname(__file__), "data")
special_tokens = ["<unk>", "<pad>", "<sos>", "<eos>", "<mask>"]


def load_data():
    dataset = load_dataset(
        "Skylion007/openwebtext",
        cache_dir=os.path.join(DATA_DIR_PATH, "train"),
        # streaming=True,
    )
    print(dataset["train"])

    for data in dataset:
        print(data)
        break


def train_tokenizer():
    tokenizer = ByteLevelBPETokenizer(lowercase=True)
    dataset = load_dataset(
        "Skylion007/openwebtext",
        cache_dir=os.path.join(DATA_DIR_PATH, "train"),
    )["train"]

    def iter_dataset():
        for data in dataset:
            yield data["text"]

    tokenizer.train_from_iterator(
        iterator=iter_dataset(),
        vocab_size=config.vocab_size,
        min_frequency=1,
        special_tokens=special_tokens,
    )
    tokenizer.save(os.path.join(DATA_DIR_PATH, "tokenizer.json"))


def load_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer.from_file(os.path.join(DATA_DIR_PATH, "tokenizer.json"))
    return tokenizer


if __name__ == "__main__":
    load_data()
    # train_tokenizer()
    # tokenizer = load_tokenizer()

    # print(tokenizer.encode("Hello, my name is Skylion.").ids)
