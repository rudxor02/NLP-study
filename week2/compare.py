import torch
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer

from week2.train import TextLSTM, TextRNN
from week2.vocab import (
    AGNewsDataset,
    index_to_word,
    load_data,
    load_vocab,
    word_to_index,
)

label_list = ("World", "Sports", "Business", "Sci/Tech")


def word_to_label(word: str) -> int:
    return label_list.index(word) + 1


class MyDataset(Dataset):
    def __init__(self, vocab: dict[str, int], sentence_len: int):
        super().__init__()
        self.sentences_with_label = [
            (
                word_to_label("Business"),
                "ABC company is going to launch a new product. ABC company is a tech company",
            ),
            (
                word_to_label("Sports"),
                "In football season with many athletes, the Lakers defeated the Warriors.",
            ),
            (
                word_to_label("Sci/Tech"),
                "Research data has been presented in the conference. It is about the performance of athletes which is sponsored by ABC company",
            ),
        ]

    def __len__(self):
        return len(self.sentences_with_label)

    def __getitem__(self, idx):
        return self.sentences_with_label[idx]


def sentence_to_tensor(
    sentence: str, vocab: dict[str, int], sentence_len: int
) -> torch.IntTensor:
    tokenizer = get_tokenizer("basic_english")
    tokens = tokenizer(sentence)
    x_data = [word_to_index(vocab, token) for token in tokens]
    if len(x_data) < sentence_len:
        x_data = [vocab["<pad>"]] * (sentence_len - len(x_data)) + x_data
    if len(x_data) > sentence_len:
        x_data = x_data[:sentence_len]
    return torch.IntTensor(x_data)


def compare():
    rnn_model_path = "week2/data/rnn_model_29"
    lstm_model_path = "week2/data/lstm_model.29"

    vocab = load_vocab(min=10, max=10000)

    rnn_model = TextRNN(
        vocab_size=len(vocab),
        embed_dim=300,
        num_classes=4,
        hidden_size=128,
        num_layers=1,
        dropout_p=0.2,
        sequence_size=50,
    )

    lstm_model = TextLSTM(
        vocab_size=len(vocab),
        embed_dim=300,
        num_classes=4,
        hidden_size=128,
        num_layers=1,
        dropout_p=0.2,
        sequence_size=50,
    )

    rnn_model.load_state_dict(torch.load(rnn_model_path))
    lstm_model.load_state_dict(torch.load(lstm_model_path))

    rnn_model.eval()
    lstm_model.eval()
    dataset = AGNewsDataset(vocab, sentence_len=50)
    my_dataset = MyDataset(vocab, sentence_len=50)

    # sentences = []

    # x_data = [sentence_to_tensor(sentence, vocab, 50) for sentence in sentences]
    # x_data = torch.stack(x_data, 0)
    train, _test = load_data()
    for i in range(30):
        import random

        # label, sentence = train[random.randint(0, len(train))]
        if i == len(my_dataset):
            break
        label, sentence = my_dataset[i]

        x_data = sentence_to_tensor(sentence, vocab, 50)

        x_data = x_data.unsqueeze(0)
        y_rnn = rnn_model(x_data).argmax(1)
        y_lstm = lstm_model(x_data).argmax(1)

        print("sentence: ", sentence)
        print("RNN model prediction: ", label_list[y_rnn.item()])
        print("LSTM model prediction: ", label_list[y_lstm.item()])
        print(label_list[label - 1])


# sentence:  Stocks Fall on Oil, Dow Ends Below 10,000 (Reuters) Reuters - The blue-chip Dow Jones average closed\below 10,000 for the first time in about six weeks on Monday as\a spike in oil prices to nearly  #36;50 a barrel renewed concerns\about corporate profits while analysts cutting recommendations\hurt tech stocks.
# RNN model prediction:  Sci/Tech
# LSTM model prediction:  Business
# Business

# sentence:  Diageo says Cheerio to US stake Diageo, the world's biggest spirits company, is selling most of its 20 stake in US food company General Mills to ease its 4.5bn (\$8bn) debt burden.
# RNN model prediction:  Sci/Tech
# LSTM model prediction:  Business
# Business

# sentence:  The Gainesville Sun Ron Zook has been relived of his duties as the Florida football head coach effective at the end of the season, Florida athletic director Jeremy Foley confirmed Monday.
# RNN model prediction:  Business
# LSTM model prediction:  Sports
# Sports

# sentence:  Cemex buying UK #39;s RMC Group Mexico #39;s Cemex, one of the world #39;s largest makers of concrete will pay \$4.1 billion for British rival RMC Group, the Wall Street Journal reported Monday.
# RNN model prediction:  Sci/Tech
# LSTM model prediction:  Business
# Business

# sentence:  Tokyo Stocks Open Higher, Exporters Lead  TOKYO (Reuters) - The Nikkei average opened 0.69 percent  higher on Wednesday as gains on Wall Street eased uncertainty  ahead of the Nov. 2 U.S. presidential election, prompting  buying of Toyota Motor Corp. and other exporters.
# RNN model prediction:  Sci/Tech
# LSTM model prediction:  Business
# Business

if __name__ == "__main__":
    compare()
