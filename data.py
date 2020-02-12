import os
import os.path as osp
import numpy as np
from collections import defaultdict
import torch
from torch.utils.data import Dataset

UNK = '[UNK]'
DELIM = '####'


class GraphDataset(Dataset):
    """A Dataset storing the triples of a Knowledge Graph.

    Args:
        triples_file: str, path to the file containing triples. This is a
            text file where each line contains a triple of the form
            'subject predicate object'
    """
    def __init__(self, triples_file):
        base_path = osp.dirname(triples_file)
        filename = osp.basename(triples_file)
        out_path = osp.join(base_path, 'processed')
        data_path = osp.join(out_path, f'{filename}.pt')
        maps_path = osp.join(out_path, 'maps.pt')

        if not osp.exists(out_path):
            os.mkdir(out_path)

        # Create or load maps from entity and relation strings to unique IDs
        if not osp.exists(maps_path):
            ent_ids = defaultdict(lambda: len(ent_ids))
            rel_ids = defaultdict(lambda: len(rel_ids))
            unk_id = ent_ids[UNK]
        else:
            maps = torch.load(maps_path)
            ent_ids, rel_ids = maps['ent_ids'], maps['rel_ids']
            ent_ids = defaultdict(lambda: ent_ids[UNK], ent_ids)

        if not osp.exists(data_path):
            # Read triples and store as ints in tensor
            file = open(triples_file)
            triples = []
            for i, line in enumerate(file):
                head, rel, tail = line.split()
                triples.append([ent_ids[head], rel_ids[rel], ent_ids[tail]])

            self.triples = torch.tensor(triples, dtype=torch.long)
            torch.save(self.triples, data_path)

            if not osp.exists(maps_path):
                # Convert maps to dict for serialization
                ent_ids = dict(ent_ids)
                rel_ids = dict(rel_ids)
                torch.save({'ent_ids': ent_ids, 'rel_ids': rel_ids}, maps_path)
        else:
            self.triples = torch.load(data_path)

        self.num_ents = len(ent_ids)
        self.num_rels = len(rel_ids)
        self.maps_path = maps_path

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


class TextGraphDataset(GraphDataset):
    """A dataset storing a graph, and textual descriptions of its entities.

    Args:
        triples_file: str, path to the file containing triples. This is a
            text file where each line contains a triple of the form
            'subject predicate object'
        text_file: str, path to the file containing entity descriptions.
            Assumes one description per line for each entity, starting with
            the entity ID, followed by the description.
    """
    def __init__(self, triples_file, text_file):
        super(TextGraphDataset, self).__init__(triples_file)

        maps = torch.load(self.maps_path)
        ent_ids, rel_ids = maps['ent_ids'], maps['rel_ids']
        ent_ids = defaultdict(lambda: ent_ids[UNK], ent_ids)

        # Read descriptions, and build a map from entity ID to text
        ent2text = dict()

        text = open(text_file)

        for line in text:
            name_start = line.find(' ')
            name_end = line.find(DELIM)
            entity = line[:name_start].strip()
            ent2text[ent_ids[entity]] = line[name_start:name_end].strip()

        print('Done reading')


if __name__ == '__main__':
    g = TextGraphDataset('data/wikifb15k-237/train.txt',
                         'data/wikifb15k-237/descriptions.txt')