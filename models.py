import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AlbertModel


class BED(nn.Module):
    def __init__(self, num_relations, dim, encoder_name):
        super().__init__()
        self.encoder = AlbertModel.from_pretrained(encoder_name,
                                                   output_attentions=False,
                                                   output_hidden_states=False)
        enc_out_dim = self.encoder.config.hidden_size
        self.enc_linear = nn.Linear(enc_out_dim, dim, bias=False)

        self.rel_emb = nn.Embedding(num_relations, dim)
        nn.init.kaiming_uniform_(self.rel_emb.weight, nonlinearity='linear')

    def ent_emb(self, tokens, masks):
        return F.normalize(self.encoder(tokens, masks)[0], dim=-1)

    def energy(self, head, tail, rel):
        return torch.norm(head + rel - tail, dim=-1, p=1)

    def forward(self, pairs_tokens, pairs_mask, rels, neg_idx):
        batch_size, _, num_tokens = pairs_tokens.shape
        # Obtain embeddings for [CLS] token
        embs = self.encoder(pairs_tokens.view(-1, num_tokens),
                            pairs_mask.view(-1, num_tokens))[0][:, 0]
        embs = self.enc_linear(embs.view(batch_size, 2, -1))

        head, tail = torch.chunk(embs, chunks=2, dim=1)
        rels = self.rel_emb(rels)

        pos_energy = self.energy(head, tail, rels)

        neg_embs = embs.view(batch_size * 2, -1)[neg_idx]
        head, tail = torch.chunk(neg_embs, chunks=2, dim=1)
        neg_energy = self.energy(head.squeeze(), tail.squeeze(), rels)

        loss = 1 + pos_energy - neg_energy
        loss[loss < 0] = 0

        return loss.mean()
