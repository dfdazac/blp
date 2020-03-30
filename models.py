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
        self.dim = dim

        nn.init.kaiming_uniform_(self.ent_emb.weight, nonlinearity='linear')
        nn.init.kaiming_uniform_(self.rel_emb.weight, nonlinearity='linear')

    def energy(self, head, tail, rel, *args, **kwargs):
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
    def __init__(self, num_entities, num_relations, dim, encoder_name,
                 margin=1, p_norm=2):
        super().__init__()
        self.bert = Albert.from_pretrained(encoder_name, num_labels=dim,
                                           output_attentions=False,
                                           output_hidden_states=False,
                                           classifier_dropout_prob=0)
        self.rel_emb = nn.Embedding(num_relations, dim)

        self.margin = margin
        self.p_norm = p_norm
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim

        nn.init.kaiming_uniform_(self.rel_emb.weight, nonlinearity='linear')

    def ent_emb(self, tokens, masks):
        return F.normalize(self.bert(tokens, masks)[0], dim=-1)

    def energy(self, head, tail, rel, ent_emb=None,
               head_mask=None, tail_mask=None):
        if ent_emb is None:
            h = self.ent_emb(head, head_mask)
            t = self.ent_emb(tail, tail_mask)
        else:
            h = ent_emb[head]
            t = ent_emb[tail]

        r = F.normalize(self.rel_emb(rel), dim=-1)

        energy = torch.norm(h + r - t, dim=-1, p=self.p_norm)
        return energy

    def forward(self, *data):
        (pos_pairs_tokens, pos_pairs_masks,
         neg_pairs_tokens, neg_pairs_masks, rels) = data

        pos_energy = self.energy(head=pos_pairs_tokens[:, 0],
                                 tail=pos_pairs_tokens[:, 1],
                                 rel=rels.squeeze(),
                                 head_mask=pos_pairs_masks[:, 0],
                                 tail_mask=pos_pairs_masks[:, 1])

        neg_energy = self.energy(head=neg_pairs_tokens[:, 0],
                                 tail=neg_pairs_tokens[:, 1],
                                 rel=rels.squeeze(),
                                 head_mask=neg_pairs_masks[:, 0],
                                 tail_mask=neg_pairs_masks[:, 1])

        loss = self.margin + pos_energy - neg_energy
        loss[loss < 0] = 0

        return loss.mean()


class RelTransE(nn.Module):
    def __init__(self, num_relations, dim, margin=1, p_norm=2):
        super().__init__()
        self.rel_emb = nn.Embedding(num_relations, dim)

        self.margin = margin
        self.p_norm = p_norm
        self.num_relations = num_relations
        self.dim = dim

        nn.init.kaiming_uniform_(self.rel_emb.weight, nonlinearity='linear')

    def energy(self, head, tail, rel, ent_emb):
        h = F.normalize(ent_emb[head], dim=-1)
        r = F.normalize(self.rel_emb(rel), dim=-1)
        t = F.normalize(ent_emb[tail], dim=-1)

        energy = torch.norm(h + r - t, dim=-1, p=self.p_norm)
        return energy

    def forward(self, *data):
        pos_pairs, neg_pairs, rels, ent_embs, alignments = data
        pos_energy = self.energy(*torch.chunk(pos_pairs, chunks=2, dim=1),
                                 rels, ent_embs)
        neg_energy = self.energy(*torch.chunk(neg_pairs, chunks=2, dim=1),
                                 rels, ent_embs)

        loss = self.margin + pos_energy - neg_energy
        loss[loss < 0] = 0

        return loss.mean()
