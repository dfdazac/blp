import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AlbertModel as Albert


class SummaryModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Albert.from_pretrained('albert-base-v2',
                                              output_attentions=False,
                                              output_hidden_states=False)

        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, *data):
        tokens, masks = data

        # TODO: Consider other representations, such as mean or max
        #   pooling of hidden states

        token_emb = self.encoder(tokens, masks)[0]

        # Return state for CLS token
        return token_emb[:, 0]


class EntityAligner(nn.Module):
    def __init__(self):
        super().__init__()

        self.text_encoder = Albert.from_pretrained('albert-base-v2',
                                                   output_attentions=False,
                                                   output_hidden_states=False)
        self.dim = self.text_encoder.config.hidden_size

    def forward(self, *data):
        tokens, mask, summaries = data

        # Obtain word representations
        token_emb = self.text_encoder(tokens, mask)[0]

        # TODO: Consider also including projection matrices for token_emb
        #   and summaries

        # Compute alignment with entity embeddings
        alignments = F.softmax(torch.matmul(token_emb, summaries.t()), dim=-1)

        # Compute entity embeddings as weighted combination of
        # word representations
        ent_embs = torch.matmul(alignments.transpose(1, 2), token_emb)

        return ent_embs.squeeze(), alignments.squeeze()
