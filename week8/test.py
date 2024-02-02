import json
from typing import Optional

from torch import Tensor
from transformers import LlamaForCausalLM, LlamaTokenizer

from week8.config import config
from week8.dataset import SimpleDataset
from week8.model import add_patch_task_vector_hook, extract_task_vector
from week8.utils import (
    LangEnum,
    compare_synonyms,
    construct_baseline_query_template,
    construct_demonstration,
    construct_regular_query_template,
)


def measure_accuracy(
    model: LlamaForCausalLM,
    tokenizer: LlamaTokenizer,
    dataset: SimpleDataset,
    task_vector: Optional[Tensor],
    layer_idx: Optional[int],
    query_template: str,
    lang: LangEnum,
) -> float:
    if str(model.device) != config.device:
        model.to(config.device)
    if model.training:
        model.eval()
    if task_vector is not None:
        if layer_idx is None:
            raise ValueError("layer_idx must be provided if task_vector is provided")
        add_patch_task_vector_hook(
            model=model, task_vector=task_vector, layer_idx=layer_idx
        )
    examples = dataset.items[:]
    total_correct = 0
    for example in examples:
        query = query_template.format(query=example[0])
        label = example[1]
        prompt_encodings = tokenizer(
            query, return_tensors="pt", add_special_tokens=False
        ).to(config.device)
        prompt_output = model.generate(
            **prompt_encodings,
            max_length=1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.9,
        )
        output = tokenizer.decode(prompt_output[0], skip_special_tokens=False)
        output = output.split("→")[-1]
        if compare_synonyms(word1=output, word2=label, lang=lang):
            total_correct += 1
    return total_correct / len(examples)


def test_regular(
    model: LlamaForCausalLM,
    tokenizer: LlamaTokenizer,
    dataset: SimpleDataset,
    demonstration_dict: dict[str, str],
    lang: LangEnum,
):
    query_template = construct_regular_query_template(demonstration_dict)

    print(
        measure_accuracy(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            task_vector=None,
            layer_idx=None,
            query_template=query_template,
            lang=lang,
        )
    )


def test_hypothesis(
    model: LlamaForCausalLM,
    tokenizer: LlamaTokenizer,
    dataset: SimpleDataset,
    demonstration_dict: dict[str, str],
    result_path: str,
    lang: LangEnum,
):

    demonstration = construct_demonstration(demonstration_dict)

    accuracies = [
        [0.0 for _ in range(config.num_layers)] for _ in range(config.num_layers)
    ]

    for i in range(config.num_layers):
        for j in range(config.num_layers):
            layer_index_from = i
            layer_index_to = j
            print("=" * 20)
            print(
                f"layer_index_from: {layer_index_from}, layer_index_to: {layer_index_to}"
            )

            task_vector = extract_task_vector(
                model=model,
                tokenizer=tokenizer,
                demonstration=demonstration,
                layer_idx=layer_index_from,
            )

            accuracy = measure_accuracy(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                task_vector=task_vector,
                layer_idx=layer_index_to,
                query_template=construct_baseline_query_template(),
                lang=lang,
            )

            accuracies[i][j] = accuracy

            print(f"accuracy: {accuracy}")

    with open(result_path, "w") as f:
        json.dump(accuracies, f)


def test_baseline(
    model: LlamaForCausalLM,
    tokenizer: LlamaTokenizer,
    dataset: SimpleDataset,
    demonstration_dict: dict[str, str],
    lang: LangEnum,
):
    query_template = construct_baseline_query_template()
    print(
        measure_accuracy(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            task_vector=None,
            layer_idx=None,
            query_template=query_template,
            lang=lang,
        )
    )
