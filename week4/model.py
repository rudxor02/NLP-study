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
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor, mask: Tensor):
        # x, encoder_output: (batch_size, seq_len, d_model)
        # mask: (batch_size, seq_len, seq_len)

        # splitted_x_list = torch.split(x, self.num_heads, dim=-2)
        attention_outputs: list[Tensor] = []
        for attention in self.attentions:
            attention_outputs.append(attention(x, x, x, mask))

        attention_outputs = torch.cat(attention_outputs, dim=-1)

        return self.residual_dropout(
            self.w_o(attention_outputs)
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

        self.layer_norm = nn.LayerNorm(config.embedding_dim)
        self.fc_1 = nn.Linear(config.embedding_dim, 4 * config.embedding_dim)
        self.fc_2 = nn.Linear(4 * config.embedding_dim, config.embedding_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.prob_residual_dropout)

    def forward(self, x: Tensor, mask: Tensor):
        # x: (batch_size, seq_len, d_model)
        # mask: (batch_size, seq_len, seq_len)

        x = x + self.attention(self.layer_norm(x), mask)
        x = x + self.dropout(self.fc_2(self.gelu(self.fc_1(self.layer_norm(x)))))
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

        self.fc_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)

        self.padding_mask = PaddingMask(config.padding_idx)
        self.look_ahead_mask = LookAheadMask()

    def forward(self, x: Tensor):
        """
        seq_len is variable
        """
        # x: (batch_size, seq_len)

        mask = self.look_ahead_mask(
            self.padding_mask(x)
        )  # (batch_size, seq_len, seq_len)

        embeddings = self.word_embedding(x)  # (batch_size, seq_len, d_model)
        embeddings = self.embedding_dropout(embeddings)

        positions = torch.arange(embeddings.size(1), device=self._device).expand(
            embeddings.size(0), embeddings.size(1)
        )

        positions = self.position_embedding(positions)  # (batch_size, seq_len, d_model)

        x = embeddings + positions  # (batch_size, seq_len, d_model)

        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, mask)

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
