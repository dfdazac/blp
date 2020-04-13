import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AlbertModel, BertModel


class DualProjection(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        in_features //= 2
        out_features //= 2
        self.linear_1 = nn.Linear(in_features, out_features, bias=False)
        self.linear_2 = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        x_1, x_2 = torch.chunk(x, chunks=2, dim=-1)
        h_1 = self.linear_1(x_1)
        h_2 = self.linear_2(x_2)
        return torch.cat((h_1, h_2), dim=-1)


class DualEmbedding(nn.Module):
    def __init__(self, num_embeddings, embeddings_dim):
        super().__init__()
        embeddings_dim //= 2
        self.embedding_1 = nn.Embedding(num_embeddings, embeddings_dim)
        self.embedding_2 = nn.Embedding(num_embeddings, embeddings_dim)
        nn.init.xavier_uniform_(self.embedding_1.weight)
        nn.init.xavier_uniform_(self.embedding_2.weight)

    def forward(self, x):
        h_1 = self.embedding_1(x)
        h_2 = self.embedding_2(x)
        return torch.cat((h_1, h_2), dim=-1)


class BED(nn.Module):
    def __init__(self, dim, rel_model, loss_fn, num_relations, encoder_name):
        super().__init__()
        self.encoder = BertModel.from_pretrained(encoder_name,
                                                 output_attentions=False,
                                                 output_hidden_states=False)
        hidden_size = self.encoder.config.hidden_size

        self.num_params = dim
        self.normalize_embs = False

        if rel_model == 'transe':
            self.score_fn = transe_score
            self.normalize_embs = True
        elif rel_model == 'distmult':
            self.score_fn = distmult_score
        elif rel_model == 'complex':
            self.score_fn = complex_score
            self.num_params *= 2
        elif rel_model == 'simple':
            self.score_fn = simple_score
            self.num_params *= 2
        else:
            raise ValueError(f'Unknown relational model {rel_model}.')

        if rel_model in {'transe', 'distmult', 'complex'}:  # FIXME
            self.enc_linear = nn.Linear(hidden_size, self.num_params, bias=False)
            self.rel_emb = nn.Embedding(num_relations, self.num_params)
            nn.init.xavier_uniform_(self.rel_emb.weight)
        else:
            self.enc_linear = DualProjection(2 * hidden_size, self.num_params)
            self.rel_emb = DualEmbedding(num_relations, self.num_params)

        if loss_fn == 'margin':
            self.loss_fn = margin_loss
        elif loss_fn == 'nll':
            self.loss_fn = nll_loss
        else:
            raise ValueError(f'Unkown loss function {loss_fn}')

    def encode(self, text_tok, text_mask, end_idx):
        if isinstance(self.enc_linear, DualProjection):
            # Extract representation of [CLS] and [SEP] tokens
            embs = self.encoder(text_tok, text_mask)[0]
            cls_embs = embs[:, 0]
            batch_idx = torch.arange(0, embs.shape[0], device=embs.device)
            sep_embs = embs[batch_idx, end_idx.squeeze()]
            embs = torch.cat((cls_embs, sep_embs), dim=-1)
        else:
            # Extract representation of [CLS] token
            embs = self.encoder(text_tok, text_mask)[0][:, 0]

        embs = self.enc_linear(embs)
        if self.normalize_embs:
            embs = F.normalize(embs, dim=-1)

        return embs

    def forward(self, text_tok, text_mask, end_idx, rels, neg_idx):
        batch_size, _, num_text_tokens = text_tok.shape

        # Obtain embeddings for [CLS] token
        embs = self.encode(text_tok.view(-1, num_text_tokens),
                           text_mask.view(-1, num_text_tokens),
                           end_idx.view(-1, 1))
        embs = embs.view(batch_size, 2, -1)

        # Scores for positive samples
        rels = self.rel_emb(rels)
        heads, tails = torch.chunk(embs, chunks=2, dim=1)
        pos_scores = self.score_fn(heads, tails, rels)

        # Scores for negative samples
        neg_embs = embs.view(batch_size * 2, -1)[neg_idx]
        heads, tails = torch.chunk(neg_embs, chunks=2, dim=1)
        neg_scores = self.score_fn(heads.squeeze(), tails.squeeze(), rels)

        loss = self.loss_fn(pos_scores, neg_scores)

        return loss


def transe_score(heads, tails, rels):
    return -torch.norm(heads + rels - tails, dim=-1, p=1)


def distmult_score(heads, tails, rels):
    return torch.sum(heads * rels * tails, dim=-1)


def complex_score(heads, tails, rels):
    heads_re, heads_im = torch.chunk(heads, chunks=2, dim=-1)
    tails_re, tails_im = torch.chunk(tails, chunks=2, dim=-1)
    rels_re, rels_im = torch.chunk(rels, chunks=2, dim=-1)

    return torch.sum(rels_re * heads_re * tails_re +
                     rels_re * heads_im * tails_im +
                     rels_im * heads_re * tails_im -
                     rels_im * heads_im * tails_re,
                     dim=-1)


def simple_score(heads, tails, rels):
    heads_h, heads_t = torch.chunk(heads, chunks=2, dim=-1)
    tails_h, tails_t = torch.chunk(tails, chunks=2, dim=-1)
    rel_a, rel_b = torch.chunk(rels, chunks=2, dim=-1)

    return torch.sum(heads_h * rel_a * tails_t +
                     tails_h * rel_b * heads_t, dim=-1) / 2


def margin_loss(pos_scores, neg_scores):
    loss = 1 - pos_scores + neg_scores
    loss[loss < 0] = 0
    return loss.mean()


def nll_loss(pos_scores, neg_scores):
    return (F.softplus(-pos_scores).mean() + F.softplus(neg_scores).mean()) / 2
