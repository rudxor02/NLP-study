import itertools
import multiprocessing as mp
import os
from functools import partial
from pickle import load

import numpy as np
import torch
from torchtext.data.utils import get_tokenizer

from week1.vocab import DATA_DIR_PATH, WORDS_COUNT, index_to_word

TRAIN_INPUT_DATA_PATH = os.path.join(DATA_DIR_PATH, "train.npy")
TRAIN_TARGET_DATA_PATH = os.path.join(DATA_DIR_PATH, "target.npy")
CBOW_WINDOW_SIZE = 5
BATCH_SIZE = 100


tokenizer = get_tokenizer("basic_english")


def one_hot_to_idx(one_hot):
    if isinstance(one_hot, np.ndarray):
        return np.argmax(one_hot, axis=0)
    return torch.argmax(one_hot, dim=0)


def idx_to_one_hot(idx: int, vocab_size: int):
    tmp = np.zeros(vocab_size)
    tmp[idx] = 1
    return tmp


def word_to_token(vocab: dict[str, int], text: str):
    try:
        return vocab[text]
    except KeyError:
        return vocab["<unk>"]


def negative_sample(vocab: dict[str, int]):
    return [
        index_to_word(vocab, idx)
        for idx in np.random.randint(0, len(vocab), CBOW_WINDOW_SIZE * 2)
    ]


def subsample(
    tokens: list[str], voca_with_freq: dict[str, tuple[int, int]], vocab: dict[str, int]
):
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
        if word_to_token(vocab, token) != 0
        and (
            np.random.uniform()
            > (1 - np.sqrt(1e-5 * WORDS_COUNT / voca_with_freq[token][1]))
        )
    ]


def process(
    data_arr: list[str],
    vocab: dict[str, int],
    voca_with_freq: dict[str, tuple[int, int]],
):
    # print(len(data_arr))
    batch_train_input: list[list[int]] = []
    batch_train_target: list[list[float]] = []
    one_hot_vector_dim = len(vocab)
    for idx, data in enumerate(data_arr):
        # print(data)
        tokens: list[str] = tokenizer(data)
        tokens = (
            subsample(tokens, voca_with_freq, vocab)
            if np.random.uniform() > 0.5
            else negative_sample(vocab)
        )

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
            batch_train_target.append(
                idx_to_one_hot(word_to_token(vocab, tokens[i]), one_hot_vector_dim)
            )
    batch_train_input, batch_train_target = np.array(batch_train_input), np.array(
        batch_train_target
    )
    # print(batch_train_input.shape)
    # print(batch_train_target.shape)
    batch_train_input = batch_train_input.reshape(-1, CBOW_WINDOW_SIZE * 2, 1)
    batch_train_target = batch_train_target.reshape(-1, one_hot_vector_dim)
    # print(batch_train_input.shape)
    # print([one_hot_to_idx(e) for e in batch_train_target[:3]])
    # print(data_arr[:3])
    # print(batch_train_target.shape)
    return batch_train_input, batch_train_target
