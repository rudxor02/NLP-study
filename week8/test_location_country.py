from transformers import LlamaForCausalLM, LlamaTokenizer

from week8.config import config
from week8.dataset import SimpleDataset
from week8.model import load_pretrained_model, load_pretrained_tokenizer
from week8.test import test_baseline, test_hypothesis, test_regular
from week8.utils import LangEnum


def load_location_country_demonstration_and__dataset() -> (
    tuple[dict[str, str], SimpleDataset]
):
    demonstration_dict = {
        "Galata": "Istanbul",
        "Oliver Ames High School": "Massachusetts",
        "Ostankinsky District": "Moscow",
    }
    return demonstration_dict, SimpleDataset(
        config.location_country_data_path, list(demonstration_dict.keys())
    )


def test_location_country_baseline(
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


def test_location_country_regular(
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


def test_location_country_hypothesis(
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
            result_path=config.location_country_hypothesis_result_path,
        )
    )


if __name__ == "__main__":
    model = load_pretrained_model()
    tokenizer = load_pretrained_tokenizer()
    demonstration_dict, dataset = load_location_country_demonstration_and__dataset()
    lang = LangEnum.EN
    test_location_country_baseline(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        demonstration_dict=demonstration_dict,
        lang=lang,
    )
    test_location_country_regular(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        demonstration_dict=demonstration_dict,
        lang=lang,
    )
    test_location_country_hypothesis(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        demonstration_dict=demonstration_dict,
        lang=lang,
    )
