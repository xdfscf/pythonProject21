import numpy as np
import torch
import torch.nn as nn
import collections.abc
import math


class VITEmbeddings(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.config.vit_hidden_size))
        self.patch_embed = VITPatchEmbeddings(config)
        self.pos_embed = nn.Parameter(torch.randn(1, self.patch_embed.num_patch+1, self.config.vit_hidden_size))
        self.dropout = nn.Dropout(self.config.vit_hidden_dropout_prob)

    def forward(self, inputs):
        patch_embeds = self.patch_embed(inputs)
        batch, seq, hidden = patch_embeds.shape
        cls_embeds = self.cls_token.repeat(batch, 1, 1)
        embeds = torch.cat((cls_embeds, patch_embeds), dim=1)
        embeds = embeds + self.pos_embed
        return embeds


class VITPatchEmbeddings(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size, self.patch_size, self.num_patch = self.calculate_patch_num()
        self.projection = nn.Conv2d(self.config.img_channels, self.config.vit_hidden_size,
                                    kernel_size=self.patch_size, stride=self.patch_size)

    def calculate_patch_num(self):
        image_size, patch_size = self.config.image_size, self.config.patch_size

        if isinstance(image_size, collections.abc.Iterable):
            assert len(image_size) == 2, "image tuple size is expected to be 2"
        else:
            image_size = (image_size, image_size)

        if isinstance(patch_size, collections.abc.Iterable):
            assert len(patch_size) == 2, "patch tuple size is expected to be 2"
        else:
            patch_size = (patch_size, patch_size)

        num_patch = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        return image_size, patch_size, num_patch

    def forward(self, inputs):
        batch_size, img_channels, height, width = inputs.shape
        assert img_channels == self.config.img_channels, f"image channel should be 3"
        assert (height, width) == self.image_size, f"image size should be ({self.image_size[0]}, {self.image_size[1]})"
        patch_embed = self.projection(inputs).flatten(2).transpose(1, 2)
        return patch_embed

class ViTAttention(nn.Module):
    def __init__(self, config) :
        super().__init__()
        self.config = config
        self.attention_heads= self.config.vit_attention_heads
        self.hidden_size = self.config.vit_hidden_size

        self.Q_attn = nn.Linear(self.hidden_size, self.hidden_size, bias=self.config.qkv_bias)
        self.K_attn = nn.Linear(self.hidden_size, self.hidden_size, bias=self.config.qkv_bias)
        self.V_attn = nn.Linear(self.hidden_size, self.hidden_size, bias=self.config.qkv_bias)
        self.dropout = nn.Dropout(self.config.vit_attention_dropout_prob)

    def forward(self, input):

        batch, seq, hidden = input.shape
        # Q, K, v each with shape batch, seq, hidden
        Q = self.Q_attn(input)
        K = self.K_attn(input)
        V = self.V_attn(input)
        Q.view(batch, seq, self.attention_heads, self.hidden_size//self.attention_heads).transpose(1,2)
        K.view(batch, seq, self.attention_heads, self.hidden_size // self.attention_heads).transpose(1, 2)
        V.view(batch, seq, self.attention_heads, self.hidden_size // self.attention_heads).transpose(1, 2)
        # (batch, head, Q_seq, hidden) @ (batch, head, hidden, K_seq) = (batch, head, Q_seq, K_seq)
        attention = (Q @ K.transpose(-1, -2)) * (1.0 / math.sqrt(K.size(-1)))
        attention = nn.Softmax(dim=-1)(attention)
        attention_probs = self.dropout(attention)
        attn = attention_probs @ V
        attn = attn.transpose(1, 2).reshape((batch, seq, hidden))
        return attn



class ViTIntermediate(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.dense = nn.Linear(config.vit_hidden_size, config.vit_intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states

class ViTOutput(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.dense = nn.Linear(config.vit_intermediate_size, config.vit_hidden_size)
        self.dropout = nn.Dropout(config.vit_hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states + input_tensor

        return hidden_states

class ViTLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embed =VITEmbeddings(self.config)
        self.seq_len_dim = 1
        self.attention = ViTAttention(self.config)
        self.intermediate = ViTIntermediate(self.config)
        self.output = ViTOutput(self.config)
        self.layernorm1= nn.LayerNorm(self.config.vit_hidden_size, eps=self.config.layer_norm_eps)
        self.layernorm2 = nn.LayerNorm(self.config.vit_hidden_size, eps=self.config.layer_norm_eps)

    def forward(self, input) :
        input = self.patch_embed(input)
        self_attention_outputs = self.attention(self.layernorm1(input))

        attention_output = self_attention_outputs

        hidden_states = attention_output + input

        layer_output = self.layernorm2(hidden_states)

        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, input)

        return layer_output
'''
def training_process():
    VILT_configure = VILTConfigure()
    ViTLayers = ViTLayer(VILT_configure)
    random_tensor = torch.rand((2, 3, 224, 224))
    ViTLayers(random_tensor)
training_process()
'''
