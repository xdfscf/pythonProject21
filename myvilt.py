from typing import Optional

import torch
import torch.nn as nn
import collections.abc
import math

from torch import meshgrid
from torch.nn import CrossEntropyLoss

from vitEmbed import VITEmbeddings
from bertEmbed import TextEmbeddings
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings


class ViltEmbeddings(nn.Module):
    """
    Construct the text and patch embeddings.

    Text embeddings are equivalent to BERT embeddings.

    Patch embeddings are equivalent to ViT embeddings.
    """

    def __init__(self, config):
        super().__init__()

        # text embeddings
        self.text_embeddings = TextEmbeddings(config)
        # visual embeddings
        self.patch_embeddings = VITEmbeddings(config)

        # modality type (text/patch) embeddings
        self.token_type_embeddings = nn.Embedding(config.modality_type_vocab_size, config.vilt_hidden_size)
        self.dropout = nn.Dropout(config.vilt_hidden_dropout_prob)
        self.config = config

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        pixel_values,
        pixel_mask,
        inputs_embeds,
        image_embeds,
        image_token_type_idx=1,
    ):
        # PART 1: text embeddings
        text_embeds = self.text_embeddings(
            input_ids=input_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        # PART 2: patch embeddings (with interpolated position encodings)
        if image_embeds is None:
            image_embeds = self.patch_embeddings(pixel_values)

            image_masks = torch.full(image_embeds.shape[:2], 1,  device=image_embeds.device)


        # PART 3: add modality type embeddings
        # 0 indicates text, 1 indicates image, 2 is optionally used when a second image is provided (NLVR2)
        if image_token_type_idx is None:
            image_token_type_idx = 1
        text_embeds = text_embeds + self.token_type_embeddings(
            torch.zeros_like(attention_mask, dtype=torch.long, device=text_embeds.device)
        )
        image_embeds = image_embeds + self.token_type_embeddings(
            torch.full_like(image_masks, image_token_type_idx, dtype=torch.long, device=image_embeds.device)
        )

        # PART 4: concatenate
        embeddings = torch.cat([text_embeds, image_embeds], dim=1)
        masks = torch.cat([attention_mask, image_masks], dim=1)


        return embeddings, masks

class ViltSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        self.attention_heads = self.config.vilt_attention_heads
        self.hidden_size = self.config.vilt_hidden_size

        self.Q_attn = nn.Linear(self.hidden_size, self.hidden_size, bias=self.config.qkv_bias)
        self.K_attn = nn.Linear(self.hidden_size, self.hidden_size, bias=self.config.qkv_bias)
        self.V_attn = nn.Linear(self.hidden_size, self.hidden_size, bias=self.config.qkv_bias)
        self.dropout = nn.Dropout(self.config.vilt_attention_dropout_prob)

    def forward(self, input, attention_mask=None, head_mask=None, output_attentions=False):
        batch, seq, hidden = input.shape

        # Q, K, v each with shape batch, seq, hidden
        Q = self.Q_attn(input)
        K = self.K_attn(input)
        V = self.V_attn(input)

        Q=Q.view(batch, seq, self.attention_heads, self.hidden_size // self.attention_heads).transpose(1, 2)
        K=K.view(batch, seq, self.attention_heads, self.hidden_size // self.attention_heads).transpose(1, 2)
        V=V.view(batch, seq, self.attention_heads, self.hidden_size // self.attention_heads).transpose(1, 2)

        # (batch, head, Q_seq, hidden) @ (batch, head, hidden, K_seq) = (batch, head, Q_seq, K_seq)
        attention_scores = (Q @ K.transpose(-1, -2)) * (1.0 / math.sqrt(K.size(-1)))


        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = attention_probs @ V
        context_layer = context_layer.permute(0, 2, 1, 3)

        new_context_layer_shape = context_layer.size()[:-2] + (hidden,)
        context_layer = context_layer.reshape(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs

class ViltSelfOutput(nn.Module):
    """
    The residual connection is defined in ViltLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config) :
        super().__init__()
        self.dense = nn.Linear(config.vilt_hidden_size, config.vilt_hidden_size)
        self.dropout = nn.Dropout(config.vilt_hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states

class ViltAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = ViltSelfAttention(config)
        self.output = ViltSelfOutput(config)


    def forward(self, input, attention_mask=None, head_mask=None, output_attentions=False):
        self_outputs = self.attention(input, attention_mask, head_mask, output_attentions)

        attention_output = self.output(self_outputs[0], input)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class ViltIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.vilt_hidden_size, config.vilt_intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTOutput with ViT->Vilt
class ViltOutput(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.dense = nn.Linear(config.vilt_intermediate_size, config.vilt_hidden_size)
        self.dropout = nn.Dropout(config.vilt_hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states + input_tensor

        return hidden_states


class ViltLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config):
        super().__init__()
        self.attention = ViltAttention(config)
        self.intermediate = ViltIntermediate(config)
        self.output = ViltOutput(config)
        self.layernorm_before = nn.LayerNorm(config.vilt_hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.vilt_hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in ViLT, layernorm is applied before self-attention
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states.to(attention_output.device)

        # in ViLT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs

class ViltEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([ViltLayer(config) for _ in range(config.num_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return (hidden_states, all_hidden_states,  all_self_attentions)

class ViltModel(nn.Module):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__()
        self.config = config

        self.embeddings = ViltEmbeddings(config)
        self.encoder = ViltEncoder(config)

        self.layernorm = nn.LayerNorm(config.vilt_hidden_size, eps=config.layer_norm_eps)
        self.pooler = ViltPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self._init_weights

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_input_embeddings(self):
        return self.embeddings.text_embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.text_embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def get_extended_attention_mask(
            self, attention_mask, input_shape, device: torch.device = None,
            dtype: torch.float = None
    ) :
        if dtype is None:
            dtype = torch.float

        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and the dtype's smallest value for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask

    def get_head_mask(
            self, head_mask, num_hidden_layers: int, is_attention_chunked: bool = False
    ):

        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        image_embeds: Optional[torch.FloatTensor] = None,
        image_token_type_idx: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):

        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else False
        )
        return_dict = return_dict if return_dict is not None else False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        text_batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(((text_batch_size, seq_length)), device=device)

        if pixel_values is not None and image_embeds is not None:
            raise ValueError("You cannot specify both pixel_values and image_embeds at the same time")
        elif pixel_values is None and image_embeds is None:
            raise ValueError("You have to specify either pixel_values or image_embeds")

        image_batch_size = pixel_values.shape[0] if pixel_values is not None else image_embeds.shape[0]
        if image_batch_size != text_batch_size:
            raise ValueError("The text inputs and image inputs need to have the same batch size")
        if pixel_mask is None:
            if isinstance(self.config.image_size, collections.abc.Iterable):
                pixel_mask = torch.ones((image_batch_size, self.config.image_size[0], self.config.image_size[1]), device=device)
            else:
                pixel_mask = torch.ones((image_batch_size, self.config.image_size, self.config.image_size),
                                        device=device)


        embedding_output, attention_mask = self.embeddings(
            input_ids,
            attention_mask,
            token_type_ids,
            pixel_values,
            pixel_mask,
            inputs_embeds,
            image_embeds,
            image_token_type_idx=image_token_type_idx,
        )

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return (
            sequence_output,
            pooled_output,
            encoder_outputs.hidden_states,
            encoder_outputs.attentions,
        )


class ViltPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.vilt_hidden_size, config.vilt_hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ViltForMaskedLM(nn.Module):
    _tied_weights_keys = ["mlm_score.decoder.weight", "mlm_score.decoder.bias"]

    def __init__(self, config):
        super().__init__()
        self.config=config
        self.vilt = ViltModel(config)
        self.mlm_score = ViltMLMHead(config)

        # Initialize weights and apply final processing
        self._init_weights

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_output_embeddings(self):
        return self.mlm_score.decoder

    def set_output_embeddings(self, new_embeddings):
        self.mlm_score.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        image_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):


        outputs = self.vilt(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            image_embeds=image_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output, pooled_output = outputs[:2]
        # split up final hidden states into text and image features
        text_seq_len = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        text_features, _ = (sequence_output[:, :text_seq_len], sequence_output[:, text_seq_len:])

        mlm_logits = self.mlm_score(text_features)

        masked_lm_loss = None

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            labels = labels.to(mlm_logits.device)
            masked_lm_loss = loss_fct(mlm_logits.view(-1, self.config.bert_vocab_size), labels.view(-1))

        if not return_dict:
            output = (mlm_logits,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return (masked_lm_loss, mlm_logits, outputs.hidden_states, outputs.attentions)


class ViltPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.bert_hidden_size, config.bert_hidden_size)
        self.transform_act_fn = nn.GELU()
        self.LayerNorm = nn.LayerNorm(config.bert_hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class ViltMLMHead(nn.Module):
    def __init__(self, config, weight=None):
        super().__init__()
        self.config = config
        self.transform = ViltPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.bert_hidden_size, config.bert_vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.bert_vocab_size))
        if weight is not None:
            self.decoder.weight = weight

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x)
        return x


class ViltForPrediction(nn.Module):
    _tied_weights_keys = ["mlm_score.decoder.weight", "mlm_score.decoder.bias"]

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vilt = ViltModel(config)
        self.prediction_score = ViltPredictionHead(config)

        # Initialize weights and apply final processing
        self._init_weights

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_output_embeddings(self):
        return self.mlm_score.decoder

    def set_output_embeddings(self, new_embeddings):
        self.mlm_score.decoder = new_embeddings

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            pixel_values: Optional[torch.FloatTensor] = None,
            pixel_mask: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            image_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):

        outputs = self.vilt(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            image_embeds=image_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output, pooled_output = outputs[:2]
        # split up final hidden states into text and image features
        text_seq_len = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        cls_feature = sequence_output[:,0]

        mlm_logits = self.prediction_score(cls_feature)

        masked_lm_loss = None

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            labels = labels.to(mlm_logits.device)
            masked_lm_loss = loss_fct(mlm_logits.view(-1, self.config.label_size), labels.view(-1))

        if not return_dict:
            output = (mlm_logits,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return (masked_lm_loss, mlm_logits, outputs.hidden_states, outputs.attentions)


class ViltPredictionHead(nn.Module):
    def __init__(self, config, weight=None):
        super().__init__()
        self.config = config
        self.transform = ViltPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.bert_hidden_size, config.label_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.label_size))
        if weight is not None:
            self.decoder.weight = weight

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x)
        return x
