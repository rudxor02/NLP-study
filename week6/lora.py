import os
from typing import Any

from datasets import DatasetDict, load_dataset
from dotenv import load_dotenv
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    LlamaModel,
    LlamaTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer

from week6.config import config
from week6.table_linearize import IndexedRowTableLinearize

load_dotenv()

token = os.getenv("HUGGINGFACE_TOKEN")

model_name = "meta-llama/Llama-2-7b-hf"


def load_pretrained_tokenizer() -> LlamaTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, token=token, add_eos_token=True
    )
    # llama model has no pad token
    tokenizer.pad_token_id = tokenizer.unk_token_id
    print("tokenizer loaded")
    return tokenizer


def load_pretrained_model() -> LlamaModel:
    model = AutoModelForCausalLM.from_pretrained(
        model_name, token=token, cache_dir="/data/hub"
    )
    print("model loaded")
    return model


def load_dataset_() -> tuple[DatasetDict, DatasetDict]:
    train_dataset, val_dataset = load_dataset(
        "wikisql", cache_dir="week6/data/datasets", split=["train", "validation"]
    )
    print("dataset loaded")
    return train_dataset, val_dataset


def preprocess_dataset(dataset: DatasetDict) -> DatasetDict:
    linearizer = IndexedRowTableLinearize()

    def format_dataset(example: dict[str, Any]):
        table_dict = {
            "header": example["table"]["header"],
            "rows": example["table"]["rows"][: config.table_max_rows],
        }
        linear_table = linearizer.process_table(table_dict)

        return {
            "question": "### Table\n"
            + linear_table
            + "\n### Question\n"
            + example["question"]
            + "\n### SQL\n"
            + example["sql"]["human_readable"]
        }

    # reference: https://github.com/mrm8488/shared_colab_notebooks/blob/master/T5_wikiSQL_with_HF_transformers.ipynb
    dataset = dataset.map(format_dataset, remove_columns=dataset.column_names)

    return dataset


def train_with_lora_lib(
    model: LlamaModel, tokenizer: LlamaTokenizer, train_dataset: DatasetDict
):
    from peft import get_peft_model
    from peft.tuners.lora.config import LoraConfig
    from peft.utils.peft_types import TaskType

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=512, return_tensors="pt"
    )

    def collate_fn(examples):
        """
        Just for checking inputs
        """
        print(examples[0]["input_ids"])
        for example in examples:
            print(tokenizer.decode(example["input_ids"]))
            print(len(example["input_ids"]))
        print(len(examples))
        raise

    train_args = TrainingArguments(
        "week6/data/sft/",
        per_device_eval_batch_size=config.batch_size,
        per_device_train_batch_size=config.batch_size,
        num_train_epochs=1,
        logging_steps=1000,
        save_steps=1,
        save_total_limit=1,
        # evaluation_strategy="steps",
        # eval_steps=1,
        logging_dir="week6/data/sft/",
    )

    trainer = SFTTrainer(
        model=model,
        args=train_args,
        tokenizer=tokenizer,
        data_collator=collator,
        train_dataset=train_dataset,
        dataset_text_field="question",
        max_seq_length=512,
        # data_collator=collate_fn,
        # compute_metrics=compute_metrics,
        # eval_dataset=val_dataset,
    )

    print("training...")
    trainer.train()


if __name__ == "__main__":
    tokenizer = load_pretrained_tokenizer()
    model = load_pretrained_model()
    train_dataset, val_dataset = load_dataset_()

    train_dataset = preprocess_dataset(train_dataset)
    # cannot evaluate because of memory issue
    # val_dataset = preprocess_dataset(tokenizer, val_dataset)

    train_with_lora_lib(model, tokenizer, train_dataset)
