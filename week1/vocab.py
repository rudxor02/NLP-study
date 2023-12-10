import os
from pickle import dump, load

import numpy as np
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import WikiText103

DATA_DIR_PATH = os.path.join(os.path.dirname(__file__), "data")
VOCAB_FILE_PATH = [
    os.path.join(DATA_DIR_PATH, "vocab_0.pkl"),
    os.path.join(DATA_DIR_PATH, "vocab_10.pkl"),
    os.path.join(DATA_DIR_PATH, "vocab_30.pkl"),
    os.path.join(DATA_DIR_PATH, "vocab_50.pkl"),
]

VOCAB_FREQ_PATH = os.path.join(DATA_DIR_PATH, "vocab.pkl")
WORDS_COUNT = 101317627


def index_to_word(vocab: dict[str, int], idx: int):
    return list(vocab.keys())[list(vocab.values()).index(idx)]


def main():
    train, test = WikiText103(root=DATA_DIR_PATH, split=("train", "test"))

    print("data loaded")

    tokenizer = get_tokenizer("basic_english")
    voca_with_freq = {"<unk>": [0, 50], "<pad>": [1, 50]}
    for data in iter(train):
        for token in tokenizer(data):
            if token not in voca_with_freq.keys():
                voca_with_freq[token] = [len(voca_with_freq), 0]
            else:
                voca_with_freq[token][1] += 1
    with open(VOCAB_FREQ_PATH, "wb") as f:
        dump(voca_with_freq, f)
    # 0: 226799
    # 10: 109616
    # 30: 64089
    # 50: 49092
    for loop_idx, freq in enumerate([0, 10, 30, 50]):
        tmp_vocab = {k: v[0] for k, v in voca_with_freq.items() if v[1] >= freq}

        tmp_vocab = {k: i for i, k in enumerate(tmp_vocab.keys())}

        with open(VOCAB_FILE_PATH[loop_idx], "wb") as f:
            dump(tmp_vocab, f)

        print(f"vocab saved: {len(tmp_vocab)}")


def analyze():
    with open(VOCAB_FREQ_PATH, "rb") as f:
        voca_with_freq = load(f)
    with open(VOCAB_FILE_PATH[3], "rb") as f:
        vocab = load(f)

    print(index_to_word(vocab, 1747))
    print(index_to_word(vocab, 823))
    print(index_to_word(vocab, 13))
    print(voca_with_freq[index_to_word(vocab, 14)])
    # print((1 - np.sqrt(1e-5 * 101317627 / voca_with_freq[index_to_word(vocab, 14)][1])))
    # print(
    #     (1 - np.sqrt(1e-5 * 101317627 / voca_with_freq[index_to_word(vocab, 40000)][1]))
    # )
    # print(voca_with_freq[index_to_word(vocab, 14)][1] / 101317627)
    # for _ in range(10):
    #     print(
    #         np.random.uniform()
    #         > (
    #             1
    #             - np.sqrt(
    #                 1e-5 * 101317627 / voca_with_freq[index_to_word(vocab, 14)][1]
    #             )
    #         )
    #     )
    # count = 0
    # for k, v in voca_with_freq.items():
    #     count += v[1]
    # print(count)
    # word count = 101317627
    # vocab_list.sort(key=lambda x: x[1][1], reverse=True)
    # print(vocab_list[:100])


if __name__ == "__main__":
    # main()
    analyze()
