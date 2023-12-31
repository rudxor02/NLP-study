from pickle import load
from typing import Union

import torch
from scipy import spatial

from week1.process import word_to_token
from week1.train import MODEL_FILE_PATH, Model
from week1.vocab import VOCAB_FILE_PATH, index_to_word


def similarity(arr1, arr2):
    return 1 - spatial.distance.cosine(arr1, arr2)


def cos_search(
    embedding_matrix: list[list[float]],
    word_or_vector: Union[str, list[float]],
    vocab: dict[str, int],
    top_n: int = 10,
) -> list[tuple[float, str]]:
    if isinstance(word_or_vector, str):
        word_idx = word_to_token(vocab, word_or_vector)
        word_embedding = embedding_matrix[word_idx]
    else:
        word_embedding = word_or_vector

    similarity_list = []
    for idx, embedding in enumerate(embedding_matrix):
        similarity_list.append(
            (
                idx,
                similarity(word_embedding.detach().numpy(), embedding.detach().numpy()),
            )
        )

    similarity_list.sort(key=lambda x: x[1], reverse=True)

    similarity_list = similarity_list[1 : top_n + 1]

    return [(sim, index_to_word(vocab, idx)) for idx, sim in similarity_list]


def main():
    vocab = load(open(VOCAB_FILE_PATH[7], "rb"))
    model = Model(len_vocab=len(vocab))
    model.load_state_dict(torch.load(MODEL_FILE_PATH + "_checkpoint"))
    search_keywords = ["happy", "tree", "pencil", "king", "cloud"]
    for keyword in search_keywords:
        result = cos_search(model.embedding.weight, keyword, vocab)
        print(f"Search result for {keyword}")
        print(result)

    king = model.embedding.weight[vocab["king"]]
    man = model.embedding.weight[vocab["man"]]
    woman = model.embedding.weight[vocab["woman"]]
    print(f"Search result for king - man + woman")
    print(cos_search(model.embedding.weight, king - man + woman, vocab))

    bigger = model.embedding.weight[vocab["bigger"]]
    big = model.embedding.weight[vocab["big"]]
    small = model.embedding.weight[vocab["small"]]
    print("Search result for bigger - big + small")
    print(cos_search(model.embedding.weight, bigger - big + small, vocab))

    paris = model.embedding.weight[vocab["paris"]]
    france = model.embedding.weight[vocab["france"]]
    germany = model.embedding.weight[vocab["germany"]]
    print("Search result for paris - france + germany")
    print(cos_search(model.embedding.weight, paris - france + germany, vocab))


def check_equal():
    vocab = load(open(VOCAB_FILE_PATH[3], "rb"))
    model = Model(len_vocab=len(vocab))
    model_2 = Model(len_vocab=len(vocab))
    model.load_state_dict(torch.load(MODEL_FILE_PATH + "_0_1000"))
    model_2.load_state_dict(torch.load(MODEL_FILE_PATH + "_0_1100"))

    print(torch.equal(model.embedding.weight, model_2.embedding.weight))


if __name__ == "__main__":
    main()
    # check_equal()
