import torch
from torch import nn
from torch import Tensor
from transformers.models.bert.modeling_bert import BertEmbeddings
from copy import deepcopy
from typing import Optional


class BertVisioLinguisticEmbeddings(BertEmbeddings):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        self.token_type_embeddings_visual = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        self.position_embeddings_visual = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )

        self.projection = nn.Linear(config.visual_embedding_dim, config.hidden_size)

    def initialize_visual_from_pretrained(self):
        self.token_type_embeddings_visual.weight = nn.Parameter(
            deepcopy(self.token_type_embeddings.weight.data), requires_grad=True
        )
        self.position_embeddings_visual.weight = nn.Parameter(
            deepcopy(self.position_embeddings.weight.data), requires_grad=True
        )

    def encode_text(
        self, input_ids: Tensor, token_type_ids: Optional[Tensor] = None
    ) -> Tensor:
        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        return embeddings

    def encode_image(
        self,
        visual_embeddings: Tensor,
        visual_embeddings_type: Tensor,
        image_text_alignment: Optional[Tensor] = None,
    ) -> Tensor:

        visual_embeddings = self.projection(visual_embeddings)
        token_type_embeddings_visual = self.token_type_embeddings_visual(
            visual_embeddings_type
        )

        # get position_embeddings
        # this depends on image_text_alignment
        position_embeddings_visual = self.get_position_embeddings_visual(
            visual_embeddings, image_text_alignment=image_text_alignment
        )

        # calculate visual embeddings
        v_embeddings = (
            visual_embeddings
            + position_embeddings_visual
            + token_type_embeddings_visual
        )
        return v_embeddings

    def get_position_embeddings_visual(
        self, visual_embeddings: Tensor, image_text_alignment: Optional[Tensor] = None
    ) -> Tensor:

        if image_text_alignment is not None:
            # image_text_alignment = Batch x image_length x alignment_number.
            # Each element denotes the position of the word corresponding to the
            # image feature. -1 is the padding value.
            image_text_alignment_mask = (
                (image_text_alignment != -1).long().to(image_text_alignment.device)
            )
            # Get rid of the -1.
            image_text_alignment = image_text_alignment_mask * image_text_alignment

            # position_embeddings_visual
            # = Batch x image_length x alignment length x dim
            position_embeddings_visual = self.position_embeddings(
                image_text_alignment
            ) * image_text_alignment_mask.unsqueeze(-1)
            position_embeddings_visual = position_embeddings_visual.sum(2)

            # We want to averge along the alignment_number dimension.
            image_text_alignment_mask = image_text_alignment_mask.sum(2)
            image_text_alignment_mask[image_text_alignment_mask == 0] = torch.tensor(
                [1], dtype=torch.long
            )  # Avoid devide by zero error
            position_embeddings_visual = (
                position_embeddings_visual / image_text_alignment_mask.unsqueeze(-1)
            )

            position_ids_visual = torch.zeros(
                visual_embeddings.size()[:-1],
                dtype=torch.long,
                device=visual_embeddings.device,
            )

            position_embeddings_visual = (
                position_embeddings_visual
                + self.position_embeddings_visual(position_ids_visual)
            )
        else:
            position_ids_visual = torch.zeros(
                visual_embeddings.size()[:-1],
                dtype=torch.long,
                device=visual_embeddings.device,
            )
            position_embeddings_visual = self.position_embeddings_visual(
                position_ids_visual
            )

        return position_embeddings_visual

    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Optional[Tensor] = None,
        visual_embeddings: Optional[Tensor] = None,
        visual_embeddings_type: Optional[Tensor] = None,
        image_text_alignment: Optional[Tensor] = None,
    ) -> Tensor:
        """
        input_ids = [batch_size, sequence_length]
        token_type_ids = [batch_size, sequence_length]
        visual_embedding = [batch_size, image_feature_length, image_feature_dim]
        image_text_alignment = [batch_size, image_feature_length, alignment_dim]
        """

        # text embeddings
        text_embeddings = self.encode_text(input_ids, token_type_ids=token_type_ids)

        # visual embeddings
        if visual_embeddings is not None and visual_embeddings_type is not None:
            v_embeddings = self.encode_image(
                visual_embeddings,
                visual_embeddings_type=visual_embeddings_type,
                image_text_alignment=image_text_alignment,
            )

            # Concate the two:
            embeddings = torch.cat(
                (text_embeddings, v_embeddings), dim=1
            )  # concat the visual embeddings after the attentions

        else:
            embeddings = text_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings