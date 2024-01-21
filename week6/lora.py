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
        "wikisql",
        cache_dir="week6/data/datasets",
        split=["train", "validation"],
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
    print("dataset preprocessed")
    return dataset


def collate_fn(examples: list[dict[str, Any]], tokenizer: LlamaTokenizer):
    """
    Just for checking inputs
    """
    print(examples[0]["input_ids"])
    for example in examples:
        print(tokenizer.decode(example["input_ids"]))
        print(len(example["input_ids"]))
    print(len(examples))
    raise


def train_with_peft_lib(
    model: LlamaModel, tokenizer: LlamaTokenizer, train_dataset: DatasetDict
):
    from peft.mapping import get_peft_model
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
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=config.max_seq_length,
        return_tensors="pt",
    )

    train_args = TrainingArguments(
        config.lib_output_dir,
        per_device_eval_batch_size=config.batch_size,
        per_device_train_batch_size=config.batch_size,
        num_train_epochs=config.num_train_epochs,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        # evaluation_strategy="steps",
        # eval_steps=1,
        logging_dir=config.lib_output_dir,
    )

    trainer = SFTTrainer(
        model=model,
        args=train_args,
        tokenizer=tokenizer,
        data_collator=collator,
        train_dataset=train_dataset,
        dataset_text_field="question",
        max_seq_length=config.max_seq_length,
        # data_collator=partial(collate_fn, tokenizer=tokenizer),
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

    train_with_peft_lib(model, tokenizer, train_dataset)
