import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AlbertModel


class BED(nn.Module):
    def __init__(self, num_relations, dim, margin, encoder_name):
        super().__init__()
        self.encoder = AlbertModel.from_pretrained(encoder_name,
                                                   output_attentions=False,
                                                   output_hidden_states=False)
        enc_out_dim = self.encoder.config.hidden_size
        self.enc_linear = nn.Linear(enc_out_dim, dim, bias=False)

        self.rel_emb = nn.Embedding(num_relations, dim)
        nn.init.kaiming_uniform_(self.rel_emb.weight, nonlinearity='linear')

        self.dim = dim
        self.margin = margin

    def ent_emb(self, tokens, masks):
        return F.normalize(self.encoder(tokens, masks)[0], dim=-1)

    def energy(self, head, tail, rel):
        return torch.norm(head + rel - tail, dim=-1, p=1)

    def encode_description(self, tokens, mask):
        # Encode text and extract representation of [CLS] token
        embs = self.encoder(tokens, mask)[0][:, 0]
        # TODO: Normalizing might not be required for all relational models
        embs = F.normalize(self.enc_linear(embs), dim=-1)

        return embs

    def forward(self, pairs_tokens, pairs_mask, rels, neg_idx):
        batch_size, _, num_tokens = pairs_tokens.shape

        # Obtain embeddings for [CLS] token
        embs = self.encode_description(pairs_tokens.view(-1, num_tokens),
                                       pairs_mask.view(-1, num_tokens))
        embs = embs.view(batch_size, 2, -1)

        # Scores for positive samples
        rels = self.rel_emb(rels)
        head, tail = torch.chunk(embs, chunks=2, dim=1)
        pos_energy = self.energy(head, tail, rels)

        # Scores for negative samples
        neg_embs = embs.view(batch_size * 2, -1)[neg_idx]
        head, tail = torch.chunk(neg_embs, chunks=2, dim=1)
        neg_energy = self.energy(head.squeeze(), tail.squeeze(), rels)

        loss = self.margin + pos_energy - neg_energy
        loss[loss < 0] = 0

        return loss.mean()
