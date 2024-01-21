from week6.config import config
from week6.evaluate import load_dataset_, load_tokenizer, preprocess_dataset, test
from week6.my_lora import MyLoraWrapper


def load_model(local_path: str):
    model = MyLoraWrapper.from_pretrained(local_path)

    print("model loaded")
    return model


if __name__ == "__main__":
    local_path = config.my_checkpoint_path
    tokenizer = load_tokenizer(local_path)
    model = load_model(local_path)
    test_dataset = load_dataset_()
    test_dataset = test_dataset.shuffle()
    test_dataset = preprocess_dataset(test_dataset)
    examples = test_dataset.select(range(100))
    test(tokenizer, model, examples)
