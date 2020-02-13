import torch
import torch.nn as nn
import torch.nn.functional as F


class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, dim, margin=1, p_norm=2):
        super().__init__()

        self.ent_emb = nn.Embedding(num_entities, dim)
        self.rel_emb = nn.Embedding(num_relations, dim)
        self.margin = margin
        self.p_norm = p_norm
        self.num_entities = num_entities
        self.num_relations = num_relations

        nn.init.kaiming_uniform_(self.ent_emb.weight, nonlinearity='linear')
        nn.init.kaiming_uniform_(self.rel_emb.weight, nonlinearity='linear')

    def energy(self, head, tail, rel):
        h = F.normalize(self.ent_emb(head), dim=-1)
        r = F.normalize(self.rel_emb(rel), dim=-1)
        t = F.normalize(self.ent_emb(tail), dim=-1)

        energy = torch.norm(h + r - t, dim=-1, p=self.p_norm)
        return energy

    def forward(self, *data):
        pos_pairs, neg_pairs, rels = data
        pos_energy = self.energy(*torch.chunk(pos_pairs, chunks=2, dim=1),
                                 rels)
        neg_energy = self.energy(*torch.chunk(neg_pairs, chunks=2, dim=1),
                                 rels)

        loss = self.margin + pos_energy - neg_energy
        loss[loss < 0] = 0

        return loss.mean()
