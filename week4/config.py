from pydantic import BaseModel
from torch import cuda


class GPTConfig(BaseModel):
    device: str = "cuda:2" if cuda.is_available() else "cpu"
    n_epochs: int = 3
    vocab_size: int = 50257
    block_size: int = 512
    # batch_size: int = 512
    batch_size: int = 64
    n_layer: int = 12
    n_head: int = 12
    embedding_dim: int = 768
    prob_attention_dropout: float = 0.1
    prob_residual_dropout: float = 0.1
    prob_embedding_dropout: float = 0.1
    # warmup_steps: int = 2000
    warmup_steps: int = 16000
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    padding_idx: int = 1


config = GPTConfig()
