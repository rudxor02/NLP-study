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

    for i in range(seq_len):
        with torch.no_grad():
            outputs = model(encoder_input, decoder_input)
            predicted_tokens = outputs[0].argmax(dim=-1)
            predicted_token = predicted_tokens[i].item()

            if predicted_token == tokenizer.token_to_id("<eos>"):
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

    model.load_state_dict(torch.load("week3/data/transformer_model.v3.9"))

    total_bleu = 0.0
    count = 0

    for i, (en, de) in enumerate(dataloader):
        en = list(en)
        de = list(de)
        for en_, de_ in zip(en, de):
            output = predict(model, tokenizer, en_, seq_len=config.seq_len)
            total_bleu += bleu.sentence_bleu([de_], output)
            count += 1
        print(f"BLEU: {total_bleu / count} {i} / {len(dataloader)}")

    print(f"BLEU: {total_bleu / count}")


def generate_examples():
    en = [
        "The quick brown fox jumps over the lazy dog.",
        "Every morning, I enjoy a cup of coffee while watching the sunrise.",
        "Technology is rapidly advancing, transforming how we live and work.",
    ]

    tokenizer = load_tokenizer()
    padding_idx = tokenizer.token_to_id("<pad>")
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
    model.load_state_dict(torch.load("week3/data/transformer_model.v3.9"))

    for _en in en:
        print("=================================")
        print(_en)
        print(predict(model, tokenizer, _en, seq_len=config.seq_len))


if __name__ == "__main__":
    test()
    # generate_examples()
