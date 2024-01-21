from pydantic import BaseModel


class FineTuningConfig(BaseModel):
    table_max_rows: int = 3
    lora_r: int = 8
    lora_alpha: int = 8
    lora_dropout: float = 0.1
    batch_size: int = 2
    max_seq_length: int = 512
    num_train_epochs: int = 1
    logging_steps: int = 1
    save_steps: int = 100
    save_total_limit: int = 1
    logging_strategy: str = "steps"
    lib_output_dir: str = "week6/data/lib/sft/"
    lib_logging_dir: str = "week6/log/lib/sft/"
    lib_checkpoint_path: str = "./week6/data/lib/sft_v2/checkpoint-3523/"
    my_output_dir: str = "week6/data/my/sft/"
    my_logging_dir: str = "week6/log/my/sft/"
    # accuracy: 0.5
    my_checkpoint_path: str = "./week6/data/my/sft_v0/checkpoint-3500/"


config = FineTuningConfig()
