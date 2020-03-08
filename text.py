import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AlbertModel as Albert


class SummaryModel(nn.Module):
    pool_first = 'first'
    pool_mean = 'mean'
    pool_max = 'max'
    pooling_strategies = {pool_first, pool_mean, pool_max}

    def __init__(self, pooling):
        super().__init__()

        if pooling not in SummaryModel.pooling_strategies:
            raise ValueError(f'{pooling} '
                             f'not in {self.pooling_strategies}')

        self.pooling = pooling
        self.encoder = Albert.from_pretrained('albert-base-v2',
                                              output_attentions=False,
                                              output_hidden_states=False)

        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, *data):
        tokens, masks = data
        token_emb = self.encoder(tokens, masks)[0]

        if self.pooling == self.pool_first:
            summary = token_emb[:, 0]
        elif self.pooling == self.pool_mean:
            summary = torch.mean(token_emb, dim=1)
        elif self.pooling == self.pool_max:
            summary, _ = torch.max(token_emb, dim=1)

        return summary


class EntityAligner(nn.Module):
    def __init__(self, summaries=None, add_embs=False):
        super().__init__()

        self.text_encoder = Albert.from_pretrained('albert-base-v2',
                                                   output_attentions=False,
                                                   output_hidden_states=False)
        self.dim = self.text_encoder.config.hidden_size
        self.add_embs = add_embs

        if summaries is not None:
            self.summaries = nn.Parameter(summaries, requires_grad=True)
        else:
            self.summaries = None

    def forward(self, *data):
        tokens, mask, summaries = data
        if self.summaries is not None:
            summaries = self.summaries

        # Obtain word representations
        token_emb = self.text_encoder(tokens, mask)[0]

        # TODO: Consider also including projection matrices for token_emb
        #   and summaries

        # Compute alignment with entity embeddings
        alignments = F.softmax(torch.matmul(token_emb, summaries.t()), dim=-1)

        # Compute entity embeddings as weighted combination of
        # word representations
        ent_embs = torch.matmul(alignments.transpose(1, 2), token_emb)
        if self.add_embs:
            ent_embs = summaries + ent_embs

        return ent_embs.squeeze(), alignments.squeeze()
