from typing import Any, Optional, OrderedDict

import numpy as np
import torch
from datasets import DatasetDict
from pydantic import BaseModel
from torch import Tensor, nn
from transformers import (
    DataCollatorForLanguageModeling,
    LlamaConfig,
    LlamaModel,
    LlamaTokenizer,
    PreTrainedModel,
    TrainingArguments,
)
from trl import SFTTrainer

from week6.config import config
from week6.lora import (
    load_dataset_,
    load_pretrained_model,
    load_pretrained_tokenizer,
    preprocess_dataset,
)


class MyLoraConfig(BaseModel):
    r: int
    lora_alpha: float
    lora_dropout: float


class MyLoraLinear(nn.Module):
    def __init__(self, layer: nn.Linear, config: MyLoraConfig):
        super().__init__()
        self._my_lora_inner_layer = layer
        self._my_lora_config = config
        self._my_lora_init_wrapper_layer()
        self._my_lora_freeze_gradient()
        self._my_lora_dropout = nn.Dropout(config.lora_dropout)
        self._my_lora_scaling = self._my_lora_config.lora_alpha / self._my_lora_config.r

    def _my_lora_init_wrapper_layer(self):
        self._my_lora_wrapper_layer_A = nn.Linear(
            self._my_lora_inner_layer.in_features, self._my_lora_config.r, bias=False
        )

        self._my_lora_wrapper_layer_B = nn.Linear(
            self._my_lora_config.r, self._my_lora_inner_layer.out_features, bias=False
        )

    def _my_lora_freeze_gradient(self):
        self._my_lora_inner_layer.requires_grad_(False)
        self._my_lora_wrapper_layer_A.requires_grad_(True)
        self._my_lora_wrapper_layer_B.requires_grad_(True)

    def forward(self, input: Tensor, *args: Any, **kwargs: Any):
        if self.training:
            ret = self._my_lora_inner_layer(input)
            ret = self._my_lora_dropout(ret)
            ret = (
                self._my_lora_wrapper_layer_B(self._my_lora_wrapper_layer_A(ret))
                * self._my_lora_scaling
            ) + ret
        else:
            raise NotImplementedError
            ret = self._my_lora_inner_layer(input)

        return ret

    def _my_lora_merged_weight(self):
        return (
            self._my_lora_wrapper_layer_B.weight @ self._my_lora_wrapper_layer_A.weight
            + self._my_lora_inner_layer.weight
        )

    def _save_to_state_dict(
        self, destination: OrderedDict[str, Any], prefix: str, keep_vars: bool
    ):
        """
        override nn.Module._save_to_state_dict
        """
        destination[prefix.replace("_my_lora_inner_layer", "") + "weight"] = (
            self._my_lora_merged_weight()
            if keep_vars
            else self._my_lora_merged_weight().detach()
        )

    def state_dict(
        self,
        *args: Any,
        destination: Optional[OrderedDict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ):
        """
        override nn.Module.state_dict
        """
        if destination is None:
            destination = OrderedDict()
        self._save_to_state_dict(destination, prefix, keep_vars)
        return destination


class MyLoraWrapper(PreTrainedModel):
    def __init__(self, model: nn.Module, config: MyLoraConfig):
        super().__init__(config=LlamaConfig())
        self._my_lora_inner_model = model
        self._my_lora_inner_model.requires_grad_(False)
        self._my_lora_config = config
        self._my_lora_init_wrapper_model()

    def _my_lora_is_replacable(self, name: str, module: nn.Module) -> bool:
        return isinstance(module, nn.Linear) and "_proj" in name and "attn" in name

    def _my_lora_is_replaced(self, name: str, module: nn.Module) -> bool:
        return isinstance(module, MyLoraLinear) and "_proj" in name and "attn" in name

    def _my_lora_replace_module(self, name: str, module: nn.Module):
        module_path = name.split(".")[:-1]
        module_name = name.split(".")[-1]

        current_module = self._my_lora_inner_model
        for path in module_path:
            current_module = getattr(current_module, path)
        setattr(current_module, module_name, module)

    def _my_lora_init_wrapper_model(self):
        modules_to_replace = {}
        for name, module in self._my_lora_inner_model.named_modules():
            if self._my_lora_is_replacable(name, module):
                modules_to_replace[name] = MyLoraLinear(module, self._my_lora_config)

        for name, module in modules_to_replace.items():
            self._my_lora_replace_module(name, module)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return self._my_lora_inner_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def state_dict(
        self,
        *args: Any,
        destination: Optional[OrderedDict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ):
        """
        override nn.Module.state_dict
        """
        original_state_dict = self._my_lora_inner_model.state_dict(
            *args,
            destination=destination,
            prefix=prefix,
            keep_vars=keep_vars,
        )

        return original_state_dict

    def print_trainable_parameters(self):
        total_prams = 0
        trainable_params = 0
        for _name, module in self._my_lora_inner_model.named_modules():
            for param in module.parameters():
                total_prams += np.prod(param.size())
                if param.requires_grad:
                    trainable_params += np.prod(param.size())
        print(
            f"trainable params: {trainable_params} || total params: {total_prams} || trainable ratio: {trainable_params / total_prams}"
        )


def train_with_my_lora(
    model: LlamaModel, tokenizer: LlamaTokenizer, train_dataset: DatasetDict
):
    my_lora_config = MyLoraConfig(
        r=config.lora_r, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout
    )
    wrapper_model = MyLoraWrapper(model, my_lora_config)

    wrapper_model.print_trainable_parameters()

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=config.max_seq_length,
        return_tensors="pt",
    )

    train_args = TrainingArguments(
        config.my_output_dir,
        per_device_eval_batch_size=config.batch_size,
        per_device_train_batch_size=config.batch_size,
        num_train_epochs=config.num_train_epochs,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        logging_dir=config.my_logging_dir,
        logging_strategy=config.logging_strategy,
    )

    trainer = SFTTrainer(
        model=wrapper_model,
        args=train_args,
        tokenizer=tokenizer,
        data_collator=collator,
        train_dataset=train_dataset,
        dataset_text_field="question",
        max_seq_length=config.max_seq_length,
    )

    print("training...")
    # trainer.train(resume_from_checkpoint=True)
    trainer.train()


if __name__ == "__main__":
    tokenizer = load_pretrained_tokenizer()
    model = load_pretrained_model()
    train_dataset, val_dataset = load_dataset_()

    train_dataset = preprocess_dataset(train_dataset)
    train_with_my_lora(model, tokenizer, train_dataset)
