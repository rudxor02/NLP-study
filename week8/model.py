import os

from dotenv import load_dotenv
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
)

from week8.config import config

load_dotenv()

token = os.getenv("HUGGINGFACE_TOKEN")

model_name = "meta-llama/Llama-2-13b-chat-hf"


def load_pretrained_tokenizer() -> LlamaTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, token=token, add_eos_token=True
    )
    # llama model has no pad token
    tokenizer.pad_token_id = tokenizer.unk_token_id
    print("tokenizer loaded")
    import warnings

    warnings.filterwarnings("ignore")
    return tokenizer


def load_pretrained_model() -> LlamaForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        model_name, token=token, cache_dir="/data/hub"
    )
    print("model loaded")
    return model


def add_patch_task_vector_hook(
    model: LlamaForCausalLM, task_vector: Tensor, layer_idx: int
) -> RemovableHandle:
    # task_vector shape: [hidden_size]
    def hook(module: nn.Module, input: tuple[Tensor], output: tuple[Tensor] | Tensor):
        if isinstance(output, Tensor):
            output[0, -1] = task_vector
            return output
        output[0][0, -1] = task_vector
        return output

    return model.model.layers[layer_idx].register_forward_hook(hook)


def extract_task_vector(
    model: LlamaForCausalLM,
    tokenizer: LlamaTokenizer,
    demonstration: str,
    layer_idx: int,
) -> Tensor:
    if str(model.device) != config.device:
        model.to(config.device)
    if model.training:
        model.eval()
    demonstration_encodings = tokenizer(
        demonstration, return_tensors="pt", add_special_tokens=False
    ).to(config.device)
    demonstration_output = model(**demonstration_encodings, output_hidden_states=True)
    task_vector = demonstration_output.hidden_states[layer_idx + 1]
    # shape: [hidden_size]
    return task_vector[:, -1, :].squeeze(0)


if __name__ == "__main__":
    tokenizer = load_pretrained_tokenizer()
    model = load_pretrained_model()
