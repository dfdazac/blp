import os
import os.path as osp
import numpy as np
from collections import defaultdict
import torch
from torch.utils.data import Dataset


class GraphDataset(Dataset):
    """A Dataset storing the triples of a Knowledge Graph.

    Args:
        triples_file: str, path to the file containing triples. This is a
            text file where each line contains a triple of the form
            'subject predicate object'
    """
    def __init__(self, triples_file):
        base_path = osp.dirname(triples_file)
        out_path = osp.join(base_path, 'processed')
        data_path = osp.join(out_path, 'data.pt')
        maps_path = osp.join(out_path, 'maps.pt')

        if not osp.exists(out_path):
            os.mkdir(out_path)

            # Create maps from entity and relation strings to unique IDs
            ent_ids = defaultdict(lambda: len(ent_ids))
            rel_ids = defaultdict(lambda: len(rel_ids))
            triples = []

            file = open(triples_file)
            for i, line in enumerate(file):
                head, rel, tail = line.split()
                triples.append([ent_ids[head], rel_ids[rel], ent_ids[tail]])

            # Convert to dict for serialization
            ent_ids = dict(ent_ids)
            rel_ids = dict(rel_ids)

            # Save separately IDs tensor, and maps
            self.triples = torch.tensor(triples, dtype=torch.long)
            torch.save(self.triples, data_path)
            torch.save({'ent_ids': ent_ids, 'rel_ids': rel_ids}, maps_path)
        else:
            self.triples = torch.load(data_path)
            maps = torch.load(maps_path)
            ent_ids, rel_ids = maps['ent_ids'], maps['rel_ids']

        self.num_ents = len(ent_ids)
        self.num_rels = len(rel_ids)

    def __getitem__(self, index):
        return self.triples[index]

    def __len__(self):
        return self.triples.shape[0]

    def negative_sampling(self, data_list):
        """Given a batch of triples, return it together with a batch of
        corrupted triples where either the subject or object are replaced
        by a random entity. Use as a collate_fn for a DataLoader.
        """
        pos_triples = torch.stack(data_list)
        num_triples = pos_triples.shape[0]

        # Randomly swap head (0) or tail (2) for a random entity
        corrupt_idx = np.random.choice([0, 2], size=num_triples)
        corrupt_ent = np.random.randint(self.num_ents, size=num_triples)
        corrupt_idx = torch.tensor(corrupt_idx)
        corrupt_ent = torch.tensor(corrupt_ent)
        triple_idx = torch.arange(num_triples)

        neg_triples = pos_triples.clone()
        neg_triples[triple_idx, corrupt_idx] = corrupt_ent

        return pos_triples, neg_triples


def make_data_iterator(data_loader):
    """Create a continuous iterator for a DataLoader.

    Args:
        data_loader: DataLoader to iterate
    """
    iterator = iter(data_loader)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(data_loader)
