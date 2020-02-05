import torch
import torch.nn as nn
import torch.nn.functional as F


class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, dim, margin=1):
        super(TransE, self).__init__()

        self.ent_emb = nn.Embedding(num_entities, dim)
        self.rel_emb = nn.Embedding(num_relations, dim)
        self.margin = margin

    def forward(self, data):
        head, pos_tail, neg_tail, rel = torch.chunk(data, chunks=4, dim=1)

        h_emb = F.normalize(self.ent_emb(head), dim=-1)
        t_pos_emb = F.normalize(self.ent_emb(pos_tail), dim=-1)
        t_neg_emb = F.normalize(self.ent_emb(neg_tail), dim=-1)
        rel_emb = F.normalize(self.rel_emb(rel), dim=-1)

        base_energy = h_emb + rel_emb
        pos_energy = torch.norm(base_energy - t_pos_emb, dim=-1)
        neg_energy = torch.norm(base_energy - t_neg_emb, dim=-1)

        loss = self.margin + pos_energy - neg_energy
        loss[loss < 0] = 0

        return loss.mean()
