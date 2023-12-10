import os

import numpy as np
import torch
from torchtext.data.utils import get_tokenizer

from week1.vocab import DATA_DIR_PATH, WORDS_COUNT, index_to_word

TRAIN_INPUT_DATA_PATH = os.path.join(DATA_DIR_PATH, "train.npy")
TRAIN_TARGET_DATA_PATH = os.path.join(DATA_DIR_PATH, "target.npy")
CBOW_WINDOW_SIZE = 4
BATCH_SIZE = 1000


tokenizer = get_tokenizer("basic_english")


def one_hot_to_idx(one_hot) -> int:
    if isinstance(one_hot, np.ndarray):
        return np.argmax(one_hot, axis=0)
    return torch.argmax(one_hot, dim=0)


def idx_to_one_hot(idx: int, vocab_size: int) -> list[int]:
    tmp = np.zeros(vocab_size)
    tmp[idx] = 1
    return tmp


def word_to_token(vocab: dict[str, int], text: str) -> int:
    try:
        return vocab[text]
    except KeyError:
        return vocab["<unk>"]


def negative_sample(vocab: dict[str, int]) -> tuple[np.array, np.array]:
    batch_inputs = np.random.randint(
        0, len(vocab), size=(BATCH_SIZE * 10, CBOW_WINDOW_SIZE * 2, 1)
    )
    # batch_targets = np.zeros((BATCH_SIZE * 10, len(vocab)))
    # for i in range(BATCH_SIZE * 10):
    #     batch_targets[i][np.random.randint(0, len(vocab))] = 0.01
    batch_targets = np.full((BATCH_SIZE * 10, len(vocab)), 1 / len(vocab))
    return batch_inputs, batch_targets


def subsample(
    tokens: list[str],
    vocab_with_freq: dict[str, tuple[int, int]],
    vocab: dict[str, int],
) -> list[str]:
    # return [
    #     token
    #     for token in tokens
    #     if word_to_token(vocab, token) > 1000
    #     and np.random.uniform()
    #     > (1 - np.sqrt(1e-5 * WORDS_COUNT / voca_with_freq[token][1]))
    # ]
    return [
        token
        for token in tokens
        if token in vocab.keys()
        and (
            np.random.uniform()
            > (1 - np.sqrt(1e-5 * WORDS_COUNT / vocab_with_freq[token][1]))
        )
    ]


def process(
    data_arr: list[str],
    vocab: dict[str, int],
    vocab_with_freq: dict[str, tuple[int, int]],
):
    # print(len(data_arr))
    # if np.random.uniform() > 0.5:
    #     return negative_sample(vocab)

    batch_train_input: list[list[int]] = []
    batch_train_target: list[list[float]] = []
    one_hot_vector_dim = len(vocab)
    for idx, data in enumerate(data_arr):
        # print(data)
        tokens: list[str] = tokenizer(data)
        tokens = subsample(tokens, vocab_with_freq, vocab)

        if len(tokens) < CBOW_WINDOW_SIZE * 2 + 1:
            continue

        for i in range(CBOW_WINDOW_SIZE, len(tokens) - CBOW_WINDOW_SIZE):
            if tokens[i] not in vocab.keys():
                continue
            train_input = [
                word_to_token(vocab, token)
                for token in tokens[i - CBOW_WINDOW_SIZE : i]
            ] + [
                word_to_token(vocab, token)
                for token in tokens[i + 1 : i + CBOW_WINDOW_SIZE + 1]
            ]

            batch_train_input.append(train_input)
            batch_train_target.append(
                idx_to_one_hot(word_to_token(vocab, tokens[i]), one_hot_vector_dim)
            )
    batch_train_input, batch_train_target = np.array(batch_train_input), np.array(
        batch_train_target
    )
    batch_train_input = batch_train_input.reshape(-1, CBOW_WINDOW_SIZE * 2, 1)
    batch_train_target = batch_train_target.reshape(-1, one_hot_vector_dim)
    return batch_train_input, batch_train_target
