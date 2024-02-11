import numpy as np
import torch
import torch.nn as nn
import collections.abc
import math

class TextEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.bert_vocab_size, config.bert_hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.bert_max_position_embeddings, config.bert_hidden_size)
        self.token_type_embeddings = nn.Embedding(config.bert_type_vocab_size, config.bert_hidden_size)
        self.LayerNorm = nn.LayerNorm(config.bert_hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.bert_hidden_dropout_prob)
        self.register_buffer(
            "position_ids", torch.arange(config.bert_max_position_embeddings).expand((1, -1)), persistent=False
        )
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        input_shape = input_ids.size()
        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings