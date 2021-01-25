from typing import Optional
from .BertVisioLinguisticEmbeddings import BertVisioLinguisticEmbeddings
from transformers.models.bert.modeling_bert import BertPooler, BertEncoder, BertPredictionHeadTransform
import torch.nn as nn
import torch
from torch import Tensor


class VisualBertModel(nn.Module):
    """ Explanation."""

    def __init__(self, config, visual_embedding_dim):
        super().__init__()

        # Attributes
        self.config = config
        self.config.visual_embedding_dim = visual_embedding_dim
        self.num_labels = config.num_labels

        # Build Bert
        self.embeddings = BertVisioLinguisticEmbeddings(self.config)
        self.encoder = BertEncoder(self.config)
        self.pooler = BertPooler(self.config)

        # Add classification head
        # Added sigmoid activation to smooth the output
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Sequential(
            BertPredictionHeadTransform(self.config),
            nn.Linear(self.config.hidden_size, self.num_labels),
            nn.Sigmoid()
        )

        self.init_weights()

    def init_weights(self):
        """
        Initializes and prunes weights if needed.
        """
        # Initialize weights
        # https://huggingface.co/transformers/_modules/transformers/modeling_bert.html
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(
            self,
            input_ids,
            text_mask,
            visual_embeddings,
            token_type_ids: Optional[Tensor] = None,
            visual_embeddings_type: Optional[Tensor] = None,
            image_text_alignment: Optional[Tensor] = None,
    ):

        image_mask = torch.arange(
            visual_embeddings.size(-2), device=visual_embeddings.device
        ).expand(visual_embeddings.size()[:-1])

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        if visual_embeddings_type is None:
            visual_embeddings_type = torch.zeros_like(image_mask)

        attention_mask = torch.cat([text_mask, image_mask], dim=-1)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility

        ## it seems that we get better result after commenting this line
        # extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Textual and visual input embedding
        embedding_output = self.embeddings(
            input_ids,
            token_type_ids,
            visual_embeddings=visual_embeddings,
            visual_embeddings_type=visual_embeddings_type,
            image_text_alignment=image_text_alignment,
        )

        # Only keep last layer hidden states (no output attentions)
        # Add attention map output by setting output_attentions=True, and return "attentionMap" in output_dic
        encoded_layers = self.encoder(hidden_states=embedding_output,
                                      attention_mask=extended_attention_mask,
                                      output_attentions=True
                                      )
        sequence_output = encoded_layers[0]
        out_attention = encoded_layers[1]
        # Bert Pooling: take hidden state of the first token of sequence_output
        pooled_output = self.pooler(sequence_output)

        output_dic = {}

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.contiguous().view(-1, self.num_labels)

        output_dic["scores"] = reshaped_logits
        output_dic["attentionMap"] = out_attention
        return output_dic
