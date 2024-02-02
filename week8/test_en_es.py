from transformers import LlamaForCausalLM, LlamaTokenizer

from week8.config import config
from week8.dataset import SimpleDataset
from week8.model import load_pretrained_model, load_pretrained_tokenizer
from week8.test import test_baseline, test_hypothesis, test_regular
from week8.utils import LangEnum


def load_en_es_demonstration_and_dataset() -> tuple[dict[str, str], SimpleDataset]:
    demonstration_dict = {
        "be": "ser",
        "and": "y",
        "a": "a",
    }
    return demonstration_dict, SimpleDataset(
        config.en_es_data_path, list(demonstration_dict.keys())
    )


def test_en_es_baseline(
    model: LlamaForCausalLM,
    tokenizer: LlamaTokenizer,
    dataset: SimpleDataset,
    demonstration_dict: dict[str, str],
    lang: LangEnum,
):
    print(
        test_baseline(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            demonstration_dict=demonstration_dict,
            lang=lang,
        )
    )


def test_en_es_regular(
    model: LlamaForCausalLM,
    tokenizer: LlamaTokenizer,
    dataset: SimpleDataset,
    demonstration_dict: dict[str, str],
    lang: LangEnum,
):
    print(
        test_regular(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            demonstration_dict=demonstration_dict,
            lang=lang,
        )
    )


def test_en_es_hypothesis(
    model: LlamaForCausalLM,
    tokenizer: LlamaTokenizer,
    dataset: SimpleDataset,
    demonstration_dict: dict[str, str],
    lang: LangEnum,
):
    print(
        test_hypothesis(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            demonstration_dict=demonstration_dict,
            lang=lang,
            result_path=config.en_es_hypothesis_result_path,
        )
    )


if __name__ == "__main__":
    model = load_pretrained_model()
    tokenizer = load_pretrained_tokenizer()
    demonstration_dict, dataset = load_en_es_demonstration_and_dataset()
    lang = LangEnum.ES
    test_en_es_baseline(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        demonstration_dict=demonstration_dict,
        lang=lang,
    )
    test_en_es_regular(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        demonstration_dict=demonstration_dict,
        lang=lang,
    )
    test_en_es_hypothesis(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        demonstration_dict=demonstration_dict,
        lang=lang,
    )
