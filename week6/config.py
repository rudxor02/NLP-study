from pydantic import BaseModel


class FineTuningConfig(BaseModel):
    sep_token: str = "[SEP]"
    table_max_rows: int = 3
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    batch_size: int = 2


config = FineTuningConfig()
