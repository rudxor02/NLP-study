from typing import Any

import torch
from datasets import DatasetDict, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaModel,
    LlamaTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

from week6.config import config
from week6.table_linearize import IndexedRowTableLinearize


def load_dataset_():
    test_dataset = load_dataset(
        "wikisql", cache_dir="week6/data/datasets", split="test"
    )
    print("dataset loaded")
    return test_dataset


def load_tokenizer(local_path: str) -> LlamaTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
    tokenizer.add_eos_token = False
    print("tokenizer loaded")
    return tokenizer


def preprocess_dataset(dataset: DatasetDict) -> DatasetDict:
    linearizer = IndexedRowTableLinearize()

    def format_dataset(example: dict[str, Any]):
        table_dict = {
            "header": example["table"]["header"],
            "rows": example["table"]["rows"][: config.table_max_rows],
        }
        linear_table = linearizer.process_table(table_dict)

        return {
            "prompt": "### Table\n"
            + linear_table
            + "\n### Question\n"
            + example["question"]
            + "\n### SQL\n",
            "label": example["sql"]["human_readable"],
            "question": example["question"],
        }

    # reference: https://github.com/mrm8488/shared_colab_notebooks/blob/master/T5_wikiSQL_with_HF_transformers.ipynb
    dataset = dataset.map(format_dataset, remove_columns=dataset.column_names)

    return dataset


def load_model(local_path: str) -> LlamaModel:
    model = AutoModelForCausalLM.from_pretrained(
        local_path,
        local_files_only=True,
        cache_dir="/data/hub",
    )
    print("model loaded")
    return model


class StopOnTokens(StoppingCriteria):
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        stop_ids = [0, 2]
        for stop_id in stop_ids:
            if input_ids[0][-1].item() == stop_id:
                return True
        return False


def test(tokenizer: LlamaTokenizer, model: LlamaModel, examples: list[dict[str, Any]]):
    model.to("cuda:5")
    model.eval()

    correct_count = 0

    for example_idx, example in enumerate(examples):
        prompt = example["prompt"]
        prompt_encodings = tokenizer(prompt, return_tensors="pt")

        prompt_encodings = prompt_encodings.to("cuda:5")

        print("=" * 50)
        print("predicting...")
        prompt_output = model.generate(
            **prompt_encodings,
            max_length=512,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.9,
            stopping_criteria=StoppingCriteriaList([StopOnTokens()]),
        )

        output = tokenizer.decode(prompt_output[0])

        label_query = example["label"]

        _, _, predicted_query = output.partition("### SQL\n")
        predicted_query = predicted_query.replace("</s>", "").replace("\n", "")

        if predicted_query.lower() == label_query.replace("'", "").lower():
            print("correct")
            correct_count += 1
        else:
            print(f"question: {example['question']}")
            print(f"label query: {label_query}")
            print(f"predicted query: {predicted_query}")
            print("wrong")
        print(f"accuracy: {correct_count / (example_idx + 1)}")
        print("=" * 50)


if __name__ == "__main__":
    local_path = config.lib_checkpoint_path
    tokenizer = load_tokenizer(local_path)
    model = load_model(local_path)
    test_dataset = load_dataset_()
    test_dataset = test_dataset.shuffle()
    test_dataset = preprocess_dataset(test_dataset)
    examples = test_dataset.select(range(100))
    test(tokenizer, model, examples)
