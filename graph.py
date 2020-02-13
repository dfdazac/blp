import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AlbertForSequenceClassification as Albert


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


class BERTransE(nn.Module):
    def __init__(self, num_entities, num_relations, dim, margin=1, p_norm=2):
        super().__init__()
        self.ent_emb = Albert.from_pretrained('albert-base-v2', num_labels=dim,
                                              output_attentions=False,
                                              output_hidden_states=False)
        self.rel_emb = nn.Embedding(num_relations, dim)

        self.margin = margin
        self.p_norm = p_norm
        self.num_entities = num_entities
        self.num_relations = num_relations

        nn.init.kaiming_uniform_(self.rel_emb.weight, nonlinearity='linear')

    def energy(self, head, tail, head_mask, tail_mask, rel):
        # TODO: normalize embeddings?
        h = self.ent_emb(head.squeeze(), head_mask.squeeze())[0]
        r = F.normalize(self.rel_emb(rel.squeeze()), dim=-1)
        t = self.ent_emb(tail.squeeze(), tail_mask.squeeze())[0]

        energy = torch.norm(h + r - t, dim=-1, p=self.p_norm)
        return energy

    def forward(self, *data):
        (pos_pairs_tokens, pos_pairs_masks,
         neg_pairs_tokens, neg_pairs_masks, rels) = data

        pos_energy = self.energy(*torch.chunk(pos_pairs_tokens, 2, dim=1),
                                 *torch.chunk(pos_pairs_masks, 2, dim=1),
                                 rels)
        neg_energy = self.energy(*torch.chunk(neg_pairs_tokens, 2, dim=1),
                                 *torch.chunk(neg_pairs_masks, 2, dim=1),
                                 rels)

        loss = self.margin + pos_energy - neg_energy
        loss[loss < 0] = 0

        return loss.mean()
