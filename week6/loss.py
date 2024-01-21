from transformers import TrainingArguments
from trl import SFTTrainer

from week6.config import config
from week6.lora import load_pretrained_model, load_pretrained_tokenizer


def loss():
    tokenizer = load_pretrained_tokenizer()
    model = load_pretrained_model()
    train_args = TrainingArguments(output_dir=config.lib_output_dir)

    trainer = SFTTrainer(
        model=model,
        args=train_args,
        tokenizer=tokenizer,
        dataset_text_field="question",
    )

    print(trainer.state.log_history)


if __name__ == "__main__":
    loss()
