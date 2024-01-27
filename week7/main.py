# reference: https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html

import os
from functools import partial
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datasets import DatasetDict
from torch import nn
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from week6.config import config
from week6.lora import (
    load_dataset_,
    load_pretrained_model,
    load_pretrained_tokenizer,
    preprocess_dataset,
)
from week6.my_lora import MyLoraConfig, MyLoraModelWrapper


def setup(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def my_lora_auto_wrap_policy(
    module: nn.Module,
    recurse: bool,
    nonwrapped_numel: int,
) -> bool:
    if hasattr(module, "is_my_lora_layer") and module.is_my_lora_layer:
        print(module)
        return False
    return True


def tokenize_dataset(dataset: DatasetDict, tokenizer: AutoTokenizer) -> DatasetDict:
    def tokenize_function(example: dict[str, Any]):
        outputs = tokenizer(
            example["question"],
            add_special_tokens=True,
            truncation=True,
            padding=False,
            max_length=config.max_seq_length,
            return_overflowing_tokens=False,
            return_length=False,
        )

        return {
            "input_ids": outputs["input_ids"],
            "attention_mask": outputs["attention_mask"],
        }

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=2,
        remove_columns=dataset.column_names,
        batch_size=500,
    )
    return tokenized_dataset


def train(
    model: nn.Module,
    rank: int,
    train_loader: DataLoader,
    optimizer: Optimizer,
    epoch: int,
    version: str,
    sampler: Optional[DistributedSampler] = None,
):
    model.train()
    ddp_loss = torch.zeros(2).to(rank)
    if sampler:
        sampler.set_epoch(epoch)
    for batch_idx, inputs in enumerate(train_loader):
        inputs = inputs.to(rank)
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        loss.backward()
        optimizer.step()
        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(inputs)
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        if rank == 0:
            print(
                f"Train step: {batch_idx} / {len(train_loader)} \tLoss: {ddp_loss[0]/ ddp_loss[1]}"
            )

        if batch_idx % 100 == 0:
            with FSDP.state_dict_type(
                model, StateDictType.FULL_STATE_DICT, save_policy
            ):
                cpu_state = model.state_dict()
            if rank == 0:
                torch.save(cpu_state, f"week7/data/cpu_state_{version}.pt")
                print("saved cpu state")
            dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)


def fsdp_main(rank: int, world_size: int, epoch: int, version: str):
    setup(rank, world_size)
    tokenizer = load_pretrained_tokenizer()
    model = load_pretrained_model()
    train_dataset, _val_dataset = load_dataset_()

    train_dataset = preprocess_dataset(train_dataset)
    train_dataset = tokenize_dataset(train_dataset, tokenizer)

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=config.max_seq_length,
        return_tensors="pt",
    )

    torch.cuda.set_device(rank)

    dataloader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collator
    )

    my_lora_config = MyLoraConfig(
        r=config.lora_r, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout
    )
    wrapper_model = MyLoraModelWrapper(model, my_lora_config)

    sharded_module = FSDP(
        wrapper_model,
        auto_wrap_policy=partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda x: (hasattr(x, "is_my_lora_layer") and x.is_my_lora_layer),
        ),
    )

    sharded_module = sharded_module.to(rank)
    optim = torch.optim.Adam(sharded_module.parameters(), lr=0.0001)

    for _epoch in range(epoch):
        train(
            model=sharded_module,
            rank=rank,
            train_loader=dataloader,
            optimizer=optim,
            epoch=_epoch,
            version=version,
        )

    cleanup()


if __name__ == "__main__":
    torch.manual_seed(123456)

    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(fsdp_main, args=(WORLD_SIZE, 1, "v0"), nprocs=WORLD_SIZE, join=True)
