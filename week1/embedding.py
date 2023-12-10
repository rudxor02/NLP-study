from pickle import load

import torch
from scipy import spatial

from week1.train import MODEL_FILE_PATH, Model
from week1.vocab import VOCAB_FILE_PATH, VOCAB_FREQ_PATH, index_to_word


def similarity(arr1, arr2):
    return 1 - spatial.distance.cosine(arr1, arr2)


def cos_search(embedding_matrix, word_or_vector, vocab: dict, top_n=10):
    if isinstance(word_or_vector, str):
        word_idx = vocab[word_or_vector]
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
    vocab = load(open(VOCAB_FILE_PATH[3], "rb"))
    model = Model(len_vocab=len(vocab))
    model.load_state_dict(torch.load(MODEL_FILE_PATH + "_2_8800"))
    search_keywords = [
        "happy",
        "tree",
        "pencil",
        "the"
        # "cloud",
        # "king",
        # "man",
        # "woman",
        # "bigger",
        # "big",
        # "small",
        # "paris",
        # "france",
        # "germany",
    ]
    # for keyword in search_keywords:
    #     result = cos_search(model.embedding.weight, keyword, vocab)
    #     print(result)
    man = model.embedding.weight[vocab["man"]]
    woman = model.embedding.weight[vocab["woman"]]
    queen = model.embedding.weight[vocab["queen"]]
    print(cos_search(model.embedding.weight, man - woman + queen, vocab))


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
