from pydantic import BaseModel


class FineTuningConfig(BaseModel):
    table_max_rows: int = 3
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    batch_size: int = 2
    max_seq_length: int = 512
    num_train_epochs: int = 1
    logging_steps: int = 1000
    save_steps: int = 1
    save_total_limit: int = 1
    lora_lib_output_dir: str = "week6/data/sft/"


config = FineTuningConfig()
