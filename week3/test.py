import nltk.translate.bleu_score as bleu
import torch
from tokenizers import Tokenizer
from torch import IntTensor, Tensor, nn
from torch.utils.data import DataLoader

from week3.model import Transformer
from week3.train import config
from week3.vocab import WMT14Dataset, load_data, load_tokenizer


class WMT14TestDataset(WMT14Dataset):
    def __getitem__(self, index: int) -> tuple[str, str]:
        return self.en[index], self.de[index]


def predict(model: nn.Module, tokenizer: Tokenizer, sentences: list[str], seq_len: int):
    model.eval()
    processed_sentences: list[Tensor] = []

    for sentence in sentences:
        sentence = tokenizer.encode(sentence).ids
        sentence = sentence[: seq_len - 2]
        sentence = (
            [tokenizer.token_to_id("<sos>")]
            + sentence
            + [tokenizer.token_to_id("<eos>")]
        )
        if len(sentence) < seq_len:
            sentence += [tokenizer.token_to_id("<pad>")] * (seq_len - len(sentence))
        sentence = IntTensor(sentence)
        processed_sentences.append(sentence)

    encoder_inputs = torch.stack(processed_sentences, dim=0)
    decoder_input = torch.full_like(encoder_inputs[0], tokenizer.token_to_id("<pad>"))
    decoder_input[0] = tokenizer.token_to_id("<sos>")
    decoder_inputs = decoder_input.unsqueeze(0)
    decoder_inputs = decoder_input.repeat(len(encoder_inputs), 1)

    with torch.no_grad():
        outputs = model(encoder_inputs, decoder_inputs)
        outputs = outputs.argmax(dim=-1)

    outputs: list[list[int]] = outputs.tolist()
    outputs: list[str] = tokenizer.decode_batch(outputs)
    outputs = [output.replace("<sos>", "").replace("<eos>", "") for output in outputs]
    return outputs


def test():
    tokenizer = load_tokenizer()
    test_en, test_de = load_data(train=False)
    dataset = WMT14Dataset(
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

    model.load_state_dict(torch.load("model.6"))

    total_bleu = 0.0

    for i, (en, de) in enumerate(dataloader):
        en = en.tolist()
        de = de.tolist()
        outputs = predict(model, tokenizer, en, seq_len=100)
        bleu_score = 0.0
        for hypothesis, reference in zip(outputs, de):
            hypothesis = tokenizer.encode(hypothesis).tokens
            reference = tokenizer.encode(reference).tokens
            bleu_score += bleu.sentence_bleu([reference], hypothesis)

    print(f"BLEU: {total_bleu / len(dataloader)}")


if __name__ == "__main__":
    test()
