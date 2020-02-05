import torch
import torch.nn as nn
import torch.nn.functional as F


class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, dim, margin=1):
        super(TransE, self).__init__()

        # TODO: Initialize
        self.ent_emb = nn.Embedding(num_entities, dim)
        self.rel_emb = nn.Embedding(num_relations, dim)
        self.margin = margin

    def energy(self, triples):
        head, rel, tail = torch.chunk(triples, chunks=3, dim=1)

        h = F.normalize(self.ent_emb(head), dim=-1)
        r = F.normalize(self.rel_emb(rel), dim=-1)
        t = F.normalize(self.ent_emb(tail), dim=-1)

        energy = torch.norm(h + r - t, dim=-1)
        return energy

    def forward(self, data):
        pos_triples, neg_triples = data

        pos_energy = self.energy(pos_triples)
        neg_energy = self.energy(neg_triples)

        loss = self.margin + pos_energy - neg_energy
        loss[loss < 0] = 0

        return loss.mean()
