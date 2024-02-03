from transformers import LlamaForCausalLM, LlamaTokenizer

from week8.config import config
from week8.dataset import SimpleDataset
from week8.model import load_pretrained_model, load_pretrained_tokenizer
from week8.test import test_baseline, test_hypothesis, test_regular
from week8.utils import LangEnum


def load_antonyms_demonstration_dict_and_dataset() -> (
    tuple[dict[str, str], SimpleDataset]
):
    demonstration_dict = {
        "high": "low",
        "big": "small",
        "deep": "shallow",
    }
    return demonstration_dict, SimpleDataset(
        config.antonyms_data_path, list(demonstration_dict.keys())
    )


def test_antonyms_baseline(
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


def test_antonyms_regular(
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


def test_antonyms_hypothesis(
    model: LlamaForCausalLM,
    tokenizer: LlamaTokenizer,
    dataset: SimpleDataset,
    demonstration_dict: dict[str, str],
    lang: LangEnum,
):
    test_hypothesis(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        demonstration_dict=demonstration_dict,
        result_path=config.antonyms_hypothesis_result_path,
        lang=lang,
    )


if __name__ == "__main__":
    tokenizer = load_pretrained_tokenizer()
    model = load_pretrained_model()
    demonstration_dict, dataset = load_antonyms_demonstration_dict_and_dataset()
    lang = LangEnum.EN
    test_antonyms_baseline(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        demonstration_dict=demonstration_dict,
        lang=lang,
    )
    test_antonyms_regular(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        demonstration_dict=demonstration_dict,
        lang=lang,
    )
