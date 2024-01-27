import torch

from week6.config import config
from week6.evaluate import load_test_dataset_, preprocess_test_dataset, test
from week6.lora import load_pretrained_model, load_pretrained_tokenizer
from week6.my_lora import MyLoraConfig, MyLoraModelWrapper


def load_from_cpu_state_dict(cpu_state_path: str):
    model = load_pretrained_model()

    my_lora_config = MyLoraConfig(
        r=config.lora_r, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout
    )
    wrapper_model = MyLoraModelWrapper(model, my_lora_config)

    wrapper_model.load_state_dict(torch.load(cpu_state_path))

    print("loaded from cpu state dict")

    return wrapper_model


if __name__ == "__main__":
    local_path = config.lib_checkpoint_path
    tokenizer = load_pretrained_tokenizer()
    # model = load_pretrained_model()
    model = load_from_cpu_state_dict("week7/data/cpu_state.pt")
    test_dataset = load_test_dataset_()
    test_dataset = test_dataset.shuffle()
    test_dataset = preprocess_test_dataset(test_dataset)
    examples = test_dataset.select(range(100))
    test(tokenizer, model, examples)
