import math
from math import sqrt
from typing import Generator

import torch
from torch import Tensor, nn

from libs.modules import RecursiveDeviceModule, StrNumOfParamsModule
from week3.model import LookAheadMask, PaddingMask
from week4.config import GPTConfig


class Attention(RecursiveDeviceModule):
    """
    Only difference from week3 `Attention` module is applying dropout to attention score
    """

    def __init__(self, d_model: int, num_heads: int, p_dropout: float):
        super(Attention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.d_n = d_model // num_heads

        self.w_q = nn.Linear(d_model, self.d_n)
        self.w_k = nn.Linear(d_model, self.d_n)
        self.w_v = nn.Linear(d_model, self.d_n)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor):
        # q, k, v: (batch_size, seq_len, d_model)
        # mask: (batch_size, seq_len, seq_len)

        q = self.w_q(q)  # (batch_size, seq_len, d_n)
        k = self.w_k(k)  # (batch_size, seq_len, d_n)
        v = self.w_v(v)  # (batch_size, seq_len, d_n)

        attention_score = (
            torch.matmul(q, torch.transpose(k, 1, 2)) / sqrt(self.d_n)
        ) + mask * -1e9  # (batch_size, seq_len, seq_len)
        attention_score = self.softmax(attention_score)
        attention_score = self.dropout(attention_score)
        return torch.matmul(attention_score, v)  # (batch_size, seq_len, d_n)


class MultiHeadAttention(RecursiveDeviceModule):
    """
    Only difference from week3 `MultiHeadAttention` module is applying dropout to output
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        prob_attention_dropout: float,
        prob_residual_dropout: float,
    ):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.d_n = d_model // num_heads

        self.attentions = nn.ModuleList(
            [
                Attention(d_model, num_heads, prob_attention_dropout)
                for _ in range(num_heads)
            ]
        )

        self.residual_dropout = nn.Dropout(p=prob_residual_dropout)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor, mask: Tensor):
        # x, encoder_output: (batch_size, seq_len, d_model)
        # mask: (batch_size, seq_len, seq_len)

        # splitted_x_list = torch.split(x, self.num_heads, dim=-2)
        attention_outputs: list[Tensor] = []
        for attention in self.attentions:
            attention_outputs.append(attention(x, x, x, mask))

        attention_outputs = torch.cat(attention_outputs, dim=-1)

        return self.residual_dropout(
            self.proj(attention_outputs)
        )  # (batch_size, seq_len, d_model)


class DecoderLayer(RecursiveDeviceModule):
    def __init__(self, config: GPTConfig):
        super(DecoderLayer, self).__init__()
        self.config = config

        self.attention = MultiHeadAttention(
            d_model=config.embedding_dim,
            num_heads=config.n_head,
            prob_attention_dropout=config.prob_attention_dropout,
            prob_residual_dropout=config.prob_residual_dropout,
        )

        self.layer_norm_1 = nn.LayerNorm(config.embedding_dim)
        self.layer_norm_2 = nn.LayerNorm(config.embedding_dim)
        self.fc_1 = nn.Linear(config.embedding_dim, 4 * config.embedding_dim)
        self.proj = nn.Linear(4 * config.embedding_dim, config.embedding_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.prob_residual_dropout)

    def forward(self, x: Tensor, mask: Tensor):
        # x: (batch_size, seq_len, d_model)
        # mask: (batch_size, seq_len, seq_len)

        x = x + self.attention(self.layer_norm_1(x), mask)
        x = x + self.dropout(self.proj(self.gelu(self.fc_1(self.layer_norm_2(x)))))
        return x


class GPT(StrNumOfParamsModule, RecursiveDeviceModule):
    def __init__(self, config: GPTConfig):
        super(GPT, self).__init__()
        self.config = config

        self.word_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(config.block_size, config.embedding_dim)
        # self.positional_encoding = PositionalEncoding(
        #     config.block_size, config.embedding_dim
        # )
        self.embedding_dropout = nn.Dropout(config.prob_embedding_dropout)

        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.n_layer)]
        )

        self.layer_norm = nn.LayerNorm(config.embedding_dim)
        self.fc_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)

        self.padding_mask = PaddingMask(config.padding_idx)
        self.look_ahead_mask = LookAheadMask()

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("proj.weight"):
                torch.nn.init.normal_(
                    p,
                    mean=0.0,
                    std=self.config.weight_std / math.sqrt(2 * config.n_layer),
                )

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.weight_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.weight_std)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def get_optimizer(self):
        # from https://github.com/karpathy/minGPT/blob/37baab71b9abea1b76ab957409a1cc2fbfba8a26/mingpt/model.py#L215
        decay_set: set[str] = set()
        no_decay_set: set[str] = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, _p in m.named_parameters():
                fpn = ".".join([mn, pn]) if mn else pn
                if pn.endswith("bias"):
                    no_decay_set.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay_set.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay_set.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay_set & no_decay_set
        union_params = decay_set | no_decay_set
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay_set))],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay_set))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optim_groups,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            lr=self.config.lr,
        )
        return optimizer

    def forward(self, x: Tensor):
        """
        seq_len is variable
        """
        # x: (batch_size, seq_len)

        mask = self.look_ahead_mask(
            self.padding_mask(x)
        )  # (batch_size, seq_len, seq_len)

        embeddings = self.word_embedding(x)  # (batch_size, seq_len, d_model)

        positions = torch.arange(embeddings.size(1), device=self._device).expand(
            embeddings.size(0), embeddings.size(1)
        )

        positions = self.position_embedding(positions)  # (batch_size, seq_len, d_model)

        x = self.embedding_dropout(
            embeddings + positions
        )  # (batch_size, seq_len, d_model)

        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, mask)
        x = self.layer_norm(x)
        return self.fc_head(x)  # (batch_size, seq_len, vocab_size)

    def _generate_next_token(
        self, x: Tensor, temperature: float, top_k: int | None
    ) -> Tensor:
        # x: (batch_size, seq_len)

        logits = self.forward(x)  # (batch_size, seq_len, vocab_size)
        logits = logits[:, -1, :] / temperature  # (batch_size, vocab_size)

        if top_k is not None:
            logits_topk, _indices = logits.topk(top_k, dim=-1)
            logits[logits < torch.min(logits_topk)] = -1e9
            probs = torch.softmax(logits, dim=-1)

            return torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        next_token = torch.argmax(logits, dim=-1)
        return next_token.unsqueeze(-1)

    @torch.no_grad()
    def generate(
        self,
        x: Tensor,
        max_iter: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> Tensor:
        # x: (1, seq_len)
        if x.size(0) != 1:
            raise ValueError("batch size should be 1")

        if str(x.device) != self._device:
            x = x.to(self._device)

        for _ in range(max_iter):
            x = (
                x
                if x.size(1) <= self.config.block_size
                else x[:, -self.config.block_size :]
            )
            next_token = self._generate_next_token(x, temperature, top_k)
            x = torch.cat([x, next_token], dim=-1)

        return x[:, -max_iter:]

    @torch.no_grad()
    def stream(
        self,
        x: Tensor,
        max_iter: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> Generator[Tensor, None, None]:
        # x: (1, seq_len)
        if x.size(0) != 1:
            raise ValueError("batch size should be 1")

        if str(x.device) != self._device:
            x = x.to(self._device)

        for _ in range(max_iter):
            x = (
                x
                if x.size(1) <= self.config.block_size
                else x[:, -self.config.block_size :]
            )
            next_token = self._generate_next_token(x, temperature, top_k)
            yield next_token
            x = torch.cat([x, next_token], dim=-1)


if __name__ == "__main__":
    config = GPTConfig()
    model = GPT(config)
    print(model)
