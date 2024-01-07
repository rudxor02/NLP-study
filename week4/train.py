import os
from functools import partial
from random import randint

from datasets import load_dataset
from tokenizers import Tokenizer
from torch import IntTensor, LongTensor, nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from libs.scheduling_trainer import LRStepSchedulingTrainer
from week4.config import config
from week4.model import GPT
from week4.vocab import DATA_DIR_PATH, load_tokenizer


class GPTScheduler(CosineAnnealingLR):
    def __init__(
        self,
        optimizer: Optimizer,
        T_max: int,
        warmup_steps: int = 0,
        eta_min: float = 0,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        self._warmup_steps = warmup_steps
        super().__init__(optimizer, T_max, eta_min, last_epoch, verbose)

    def _get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        elif self.last_epoch < self._warmup_steps:
            print(self.last_epoch)
            print(self._warmup_steps)
            return [
                self.last_epoch * base_lr / self._warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            return super().get_lr()

    def get_lr(self):
        return self._get_lr()


def collate_fn(
    batch: list[dict[str, str]], tokenizer: Tokenizer
) -> tuple[IntTensor, LongTensor]:
    batch_texts = [data["text"] for data in batch]
    batch_tokens = tokenizer.encode_batch(batch_texts)

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


def train():
    tokenizer = load_tokenizer()
    dataset = load_dataset(
        "Skylion007/openwebtext",
        cache_dir=os.path.join(DATA_DIR_PATH, "train"),
        # streaming=True,
    )

    dataloader = DataLoader(
        dataset["train"],
        batch_size=config.batch_size,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
        num_workers=4,
        shuffle=True,
    )

    model = GPT(config)

    criterion = nn.CrossEntropyLoss(ignore_index=config.padding_idx)

    optimizer = model.get_optimizer()

    trainer = LRStepSchedulingTrainer(
        model=model,
        train_dataloader=dataloader,
        batch_size=config.batch_size,
        criterion=criterion,
        optimizer=optimizer,
    )

    trainer.run(
        num_epoch=config.n_epochs,
        device=config.device,
        model_save_path=os.path.join(DATA_DIR_PATH, "model"),
        loss_save_path=os.path.join(DATA_DIR_PATH, "loss"),
        model_version="v2",
        verbose=True,
    )


if __name__ == "__main__":
    train()
