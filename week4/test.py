import math

import torch
from tokenizers import Tokenizer
from torch import IntTensor, nn
from torch.utils.data import DataLoader, Dataset
from torchtext.datasets import WikiText103

from week4.config import config
from week4.model import GPT
from week4.vocab import DATA_DIR_PATH, load_tokenizer


class WikiTextDataset(Dataset):
    def __init__(self, tokenizer: Tokenizer, stride: int = 512):
        self.stride = stride
        _train, test = WikiText103(root=DATA_DIR_PATH, split=("train", "test"))
        self.encodings = IntTensor(tokenizer.encode("\n\n".join(test)).ids)

    def __getitem__(self, index: int):
        return (
            self.encodings[index * self.stride : (index + 1) * self.stride],
            self.encodings[
                index * self.stride + 1 : (index + 1) * self.stride + 1
            ].type(torch.LongTensor),
        )

    def __len__(self):
        return (len(self.encodings) // self.stride) - 1


def test():
    tokenizer = load_tokenizer()

    dataset = WikiTextDataset(tokenizer=tokenizer, stride=config.block_size)

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        # collate_fn=partial(collate_fn, tokenizer=tokenizer),
        # num_workers=4,
        # shuffle=True,
    )

    model = GPT(config)
    model.load_state_dict(torch.load("week4/data/model.v2.epoch_0.step_19500"))

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
            print(loss)

    print(f"cross entropy: {sum(losses) / (len(losses))}")
    print(f"perplexity: {math.exp(sum(losses) / (len(losses)))}")


if __name__ == "__main__":
    test()
