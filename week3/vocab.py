import os

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.implementations import CharBPETokenizer
from torch import IntTensor, LongTensor, Tensor
from torch.utils.data import Dataset

DATA_DIR_PATH = os.path.join(os.path.dirname(__file__), "data")
VOCAB_FREQ_PATH = os.path.join(DATA_DIR_PATH, "vocab")
special_tokens = ["<unk>", "<pad>", "<sos>", "<eos>", "<mask>"]


def load_data(train: bool, num_data: int | None = None) -> tuple[list[str], list[str]]:
    """
    Load WMT'14 English-German dataset, raw data is already downloaded from https://nlp.stanford.edu/projects/nmt/

    :param train: If True, load train dataset. If False, load test dataset.

    :return: tuple of list(english_sentences, german_sentences) - length is 45000 if train is True.
    """
    if train:
        f_en = open(os.path.join(DATA_DIR_PATH, "train.en"))
        f_de = open(os.path.join(DATA_DIR_PATH, "train.de"))
    else:
        f_en = open(os.path.join(DATA_DIR_PATH, "newstest2014.en"))
        f_de = open(os.path.join(DATA_DIR_PATH, "newstest2014.de"))
    en = f_en.readlines()
    if num_data is not None:
        en = en[:num_data]
    de = f_de.readlines()
    if num_data is not None:
        de = de[:num_data]
    f_en.close()
    f_de.close()
    en = [sentence.strip() for sentence in en]
    de = [sentence.strip() for sentence in de]
    return en, de


def index_to_word(vocab: dict[str, int], idx: int) -> str:
    return list(vocab.keys())[list(vocab.values()).index(idx)]


def word_to_index(vocab: dict[str, int], word: str) -> int:
    try:
        return vocab[word]
    except KeyError:
        return vocab["<unk>"]


def train_tokenizer():
    tokenizer = CharBPETokenizer(lowercase=True, suffix="")
    tokenizer.train(
        files=[
            os.path.join(DATA_DIR_PATH, "train.de"),
            os.path.join(DATA_DIR_PATH, "train.en"),
        ],
        vocab_size=37000,
        min_frequency=1,
        special_tokens=special_tokens,
        suffix="",
    )
    tokenizer.save(os.path.join(DATA_DIR_PATH, "tokenizer.json"))


def load_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer.from_file(os.path.join(DATA_DIR_PATH, "tokenizer.json"))
    return tokenizer


def analyze(data: list[str]):
    """
    Analyze the data and print the result.
    """
    tokenizer = load_tokenizer()
    data = [tokenizer.encode(sentence).ids for sentence in data]
    print(data[:5])
    data = [len(sentence) for sentence in data]
    print(max(data))


class WMT14Dataset(Dataset):
    def __init__(
        self,
        tokenizer: Tokenizer,
        train_en: list[str],
        train_de: list[str],
        seq_length: int = 100,
    ):
        super(WMT14Dataset, self).__init__()
        self.tokenizer = tokenizer
        self.en = train_en
        self.de = train_de
        self.seq_length = seq_length
        self.cached_en = [None for _ in range(len(self.en))]
        self.cached_de = [None for _ in range(len(self.de))]
        self.cached_label = [None for _ in range(len(self.de))]
        self.vocab_size = self.tokenizer.get_vocab_size()

        self.unk = self.tokenizer.token_to_id("<unk>")
        self.pad = self.tokenizer.token_to_id("<pad>")
        self.sos = self.tokenizer.token_to_id("<sos>")
        self.eos = self.tokenizer.token_to_id("<eos>")
        self.mask = self.tokenizer.token_to_id("<mask>")

        if len(self.en) != len(self.de):
            raise ValueError("train_en and train_de must have same length.")

        self.pad_one_hot = F.one_hot(
            LongTensor([self.pad]), num_classes=self.vocab_size
        )

    def __getitem__(self, index: int) -> tuple[IntTensor, IntTensor, LongTensor]:
        if self.cached_en[index] is not None:
            return (
                self.cached_en[index],
                self.cached_de[index],
                self.cached_label[index],
            )

        en_sentence = self.en[index]
        de_sentence = self.de[index]
        en_sentence = self.tokenizer.encode(en_sentence).ids
        de_sentence = self.tokenizer.encode(de_sentence).ids
        en_sentence = en_sentence[: self.seq_length - 2]
        de_sentence = de_sentence[: self.seq_length - 2]
        en_sentence = [self.sos] + en_sentence + [self.eos]
        de_sentence = [self.sos] + de_sentence + [self.eos]
        if len(en_sentence) < self.seq_length:
            en_sentence += [self.pad] * (self.seq_length - len(en_sentence))
        if len(de_sentence) < self.seq_length:
            de_sentence += [self.pad] * (self.seq_length - len(de_sentence))

        en_sentence = IntTensor(en_sentence)
        de_sentence = IntTensor(de_sentence)
        label = de_sentence[1:].to(torch.int64)
        label = torch.cat([label, LongTensor([self.pad])], dim=0)
        self.cached_en[index] = en_sentence
        self.cached_de[index] = de_sentence
        self.cached_label[index] = label

        return en_sentence, de_sentence, label

    def __len__(self):
        return len(self.en)


if __name__ == "__main__":
    train_en, train_de = load_data(train=True)
    dataset = WMT14Dataset(load_tokenizer(), train_en, train_de)
    print(dataset[0])
    print(dataset[100])
