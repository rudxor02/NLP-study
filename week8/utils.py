from enum import Enum

from nltk.corpus import wordnet as wn


class LangEnum(Enum):
    EN = "eng"
    ES = "spa"


def get_synonyms(word: str, lang: LangEnum) -> set[str]:
    synsets = wn.synsets(word, lang=lang)
    synonyms: set[str] = set(word)
    for synset in synsets:
        for lemma in synset.lemmas(lang=lang.value):
            synonyms.add(lemma.name())
    return synonyms


def construct_demonstration(demonstration_dict: dict[str, str]) -> str:
    items = list(demonstration_dict.items())
    return (
        f"<s>{items[0][0]}→{items[0][1]}, {items[1][0]}→{items[1][1]}, {items[2][0]}→"
    )


def construct_regular_query_template(demonstration_dict: dict[str, str]) -> str:
    items = list(demonstration_dict.items())
    return f"<s>{items[0][0]}→{items[0][1]}, {items[1][0]}→{items[1][1]}, " + "{query}→"


def construct_baseline_query_template() -> str:
    return "<s>{query}→"


def compare_synonyms(word1: str, word2: str, lang: LangEnum) -> bool:
    word1_synonyms = get_synonyms(word1, lang)
    word2_synonyms = get_synonyms(word2, lang)

    return len(word1_synonyms & word2_synonyms) > 0
