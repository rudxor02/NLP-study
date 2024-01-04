import os
from random import randint

from datasets import load_dataset
from torch import IntTensor, Tensor, nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from libs.scheduling_trainer import LRStepSchedulingTrainer
from week4.config import config
from week4.model import GPT
from week4.vocab import DATA_DIR_PATH, load_tokenizer


class GPTScheduler(CosineAnnealingLR):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, verbose=False):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        """
        lr updates per step, not per epoch
        """
        if self.last_epoch == 0:
            return self.base_lrs
        elif self.last_epoch < self.T_max:
            return [self.last_epoch / self.T_max for _ in self.base_lrs]
        else:
            return super().get_lr()


def train():
    tokenizer = load_tokenizer()
    dataset = load_dataset(
        "Skylion007/openwebtext",
        cache_dir=os.path.join(DATA_DIR_PATH, "train"),
        # streaming=True,
    )

    def collate_fn(batch: list[dict[str, str]]) -> tuple[IntTensor, Tensor]:
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

        return IntTensor(batch_inputs), Tensor(batch_labels)

    dataloader = DataLoader(
        dataset["train"],
        batch_size=config.batch_size,
        collate_fn=collate_fn,
    )

    model = GPT(config)

    criterion = nn.CrossEntropyLoss(ignore_index=config.padding_idx)

    optimizer = Adam(
        model.parameters(), betas=(config.adam_beta1, config.adam_beta2), lr=2.5e-4
    )

    scheduler = GPTScheduler(optimizer, config.warmup_steps, verbose=True)

    print(model)

    trainer = LRStepSchedulingTrainer(
        model=model,
        train_dataloader=dataloader,
        batch_size=config.batch_size,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=scheduler,
    )

    trainer.run(
        num_epoch=config.n_epochs,
        device=config.device,
        model_save_path=os.path.join(DATA_DIR_PATH, "model"),
        model_version="v1",
        verbose=True,
    )


if __name__ == "__main__":
    train()
