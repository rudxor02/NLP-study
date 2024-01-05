import math
from functools import partial
from random import randint

import torch
from tokenizers import Tokenizer
from torch import IntTensor, LongTensor, nn
from torch.utils.data import DataLoader
from torchtext.datasets import WikiText103

from week4.config import config
from week4.model import GPT
from week4.vocab import DATA_DIR_PATH, load_tokenizer


def collate_fn(batch: list[str], tokenizer: Tokenizer) -> tuple[IntTensor, LongTensor]:
    batch_tokens = tokenizer.encode_batch(batch)

    batch_inputs, batch_labels = [], []

    for tokens in batch_tokens:
        tokens = tokens.ids
        if len(tokens) <= config.block_size + 1:
            tokens += [tokenizer.token_to_id("<pad>")] * (
                config.block_size + 1 - len(tokens)
            )
            batch_inputs.append(tokens[:-1])
            batch_labels.append(tokens[1:])
            continue
        idx = randint(0, len(tokens) - config.block_size - 1)
        batch_inputs.append(tokens[idx : idx + config.block_size])
        batch_labels.append(tokens[idx + 1 : idx + config.block_size + 1])

    return IntTensor(batch_inputs), LongTensor(batch_labels)


def test():
    _train, test = WikiText103(root=DATA_DIR_PATH, split=("train", "test"))

    tokenizer = load_tokenizer()

    dataloader = DataLoader(
        list(test),
        batch_size=config.batch_size,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
        num_workers=4,
        shuffle=True,
    )

    model = GPT(config)
    model.load_state_dict(torch.load("week4/data/model.v2.epoch_0.step_300"))

    criterion = nn.CrossEntropyLoss(ignore_index=config.padding_idx)

    model.to(config.device)
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for data, label in dataloader:
            data = data.to(config.device)
            label = label.to(config.device)
            pred = model(data)
            pred = torch.transpose(pred, 1, 2)
            loss = criterion(pred, label)
            losses.append(loss.item())

    print(f"cross entropy: {sum(losses) / (len(losses))}")
    print(f"perplexity: {math.exp(sum(losses) / (len(losses)))}")


if __name__ == "__main__":
    test()
