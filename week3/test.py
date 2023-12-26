import nltk.translate.bleu_score as bleu
import torch
from tokenizers import Tokenizer
from torch import IntTensor, nn
from torch.utils.data import DataLoader

from week3.model import Transformer
from week3.train import config
from week3.vocab import WMT14Dataset, load_data, load_tokenizer


class WMT14TestDataset(WMT14Dataset):
    def __getitem__(self, index: int) -> tuple[str, str]:
        return self.en[index], self.de[index]


def predict(model: nn.Module, tokenizer: Tokenizer, sentence: str, seq_len: int) -> str:
    model.eval()

    tokens = tokenizer.encode(sentence).ids

    tokens = tokens[: seq_len - 2]
    tokens = (
        [tokenizer.token_to_id("<sos>")] + tokens + [tokenizer.token_to_id("<eos>")]
    )
    if len(tokens) < seq_len:
        tokens += [tokenizer.token_to_id("<pad>")] * (seq_len - len(tokens))

    encoder_input = IntTensor([tokens])
    decoder_input = IntTensor(
        [
            [tokenizer.token_to_id("<sos>")]
            + [tokenizer.token_to_id("<pad>")] * (seq_len - 1)
        ]
    )

    # print("*********************************")
    for i in range(seq_len):
        with torch.no_grad():
            outputs = model(encoder_input, decoder_input)
            predicted_tokens = outputs[0].argmax(dim=-1)
            predicted_token = predicted_tokens[i].item()

            # print(predicted_tokens)
            # print(predicted_token)
            # print(decoder_input)
            # print("=================================")

            if predicted_token == tokenizer.token_to_id("<eos>"):
                # if i > 0:
                #     decoder_input[:, 1:i] = predicted_tokens[: i - 1]
                break
            decoder_input[:, i + 1] = predicted_token

    decoder_output = decoder_input[0][1:].tolist()

    return " ".join(
        tokenizer.id_to_token(token)
        for token in decoder_output
        if token != tokenizer.token_to_id("<pad>")
        and token != tokenizer.token_to_id("<eos>")
    )


def test():
    tokenizer = load_tokenizer()
    test_en, test_de = load_data(train=False)
    dataset = WMT14TestDataset(
        tokenizer,
        test_en,
        test_de,
        seq_length=100,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
    )

    padding_idx = dataset.pad

    model = Transformer(
        padding_idx=padding_idx,
        vocab_size=tokenizer.get_vocab_size(),
        seq_len=config.seq_len,
        d_model=config.d_model,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        p_dropout=config.p_dropout,
        num_layers=config.num_layers,
    )

    model.load_state_dict(torch.load("week3/data/transformer_model.v2.5"))

    total_bleu = 0.0

    for i, (en, de) in enumerate(dataloader):
        en = list(en)
        de = list(de)
        outputs = [predict(model, tokenizer, en_, seq_len=config.seq_len) for en_ in en]
        for en_, de_, output_ in zip(en, de, outputs):
            print(f"en: {en_}")
            print(f"de: {de_}")
            print(f"output: {output_}")
            print("%.2f" % bleu.sentence_bleu([de_], output_))
        raise Exception
        bleu_score = 0.0
        for hypothesis, reference in zip(outputs, de):
            # print(f"hypothesis: {' '.join(hypothesis)}")
            hypothesis = tokenizer.encode(hypothesis).tokens
            reference = tokenizer.encode(reference).tokens
            print(f"hypothesis: {' '.join(hypothesis)}")
            print(f"reference: {' '.join(reference)}")
            bleu_score += bleu.sentence_bleu([reference], hypothesis)
        total_bleu += bleu_score

    print(f"BLEU: {total_bleu / len(dataloader)}")


def a():
    print(
        bleu.sentence_bleu(
            [
                "Kürzlich stellte das Unternehmen eine Anzeige vor , die der Beginn eines neuen Anzeigenkriegs sind könnte und Geldgebern ein Bild zeigte , in dem drei Personen zusammengequetscht in einem Restaurant sitzen , und dazu der Titel : „ Würden Sie das akzeptieren ? “"
            ],
            "das ist der fall .",
        )
    )


if __name__ == "__main__":
    test()
    # a()
