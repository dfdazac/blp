import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
import transformers

UNK = '[UNK]'
DELIM = '####'


def file_to_ids(file_path):
    """Read one line per file and assign it an ID.

    Args:
        file_path: str, path of file to read

    Returns: dict, mapping str to ID (int)
    """
    str2id = dict()
    with open(file_path) as file:
        for i, line in enumerate(file):
            str2id[line.strip()] = i

    return str2id


class GraphDataset(Dataset):
    """A Dataset storing the triples of a Knowledge Graph.

    Args:
        triples_file: str, path to the file containing triples. This is a
            text file where each line contains a triple of the form
            'subject predicate object'
        ents_file: str, path to file with a list of unique entities, possibly
            shared between Datasets.
        rels_file: str, path to file with a list of unique relations, possibly
            shared between Datasets.
    """
    def __init__(self, triples_file, ents_file=None, rels_file=None):
        base_path = osp.dirname(triples_file)
        filename = osp.basename(triples_file)
        out_path = osp.join(base_path, 'processed')
        data_path = osp.join(out_path, f'{filename}.pt')
        maps_path = osp.join(out_path, 'maps.pt')

        if not osp.exists(out_path):
            os.mkdir(out_path)

        # Create or load maps from entity and relation strings to unique IDs
        if not osp.exists(maps_path):
            if not ents_file or not rels_file:
                raise ValueError('Maps file not found.')

            ent_ids = file_to_ids(ents_file)
            rel_ids = file_to_ids(rels_file)
        else:
            maps = torch.load(maps_path)
            ent_ids, rel_ids = maps['ent_ids'], maps['rel_ids']

        if not osp.exists(data_path):
            # Read triples and store as ints in tensor
            file = open(triples_file)
            triples = []
            for i, line in enumerate(file):
                head, rel, tail = line.split()
                triples.append([ent_ids[head], rel_ids[rel], ent_ids[tail]])

            self.triples = torch.tensor(triples, dtype=torch.long)
            torch.save(self.triples, data_path)

            # Save maps for reuse
            if not osp.exists(maps_path):
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
    Note: if text_data is not provided, both text_file and tokenizer must be
    given.

    Args:
        triples_file: str, path to the file containing triples. This is a
            text file where each line contains a triple of the form
            'subject predicate object'
        ents_file: str, path to file with a list of unique entities, possibly
            shared between Datasets.
        rels_file: str, path to file with a list of unique relations, possibly
            shared between Datasets.
        text_file: str, path to the file containing entity descriptions.
            Assumes one description per line for each entity, starting with
            the entity ID, followed by its description.
        tokenizer: transformers.PreTrainedTokenizer
        text_data: torch.Tensor, containing text IDs obtained from text_file
            after tokenization. Use to share text data between Datasets.
    """
    def __init__(self, triples_file, ents_file=None, rels_file=None,
                 text_data=None, text_file=None,
                 tokenizer: transformers.PreTrainedTokenizer=None):
        super(TextGraphDataset, self).__init__(triples_file,
                                               ents_file,
                                               rels_file)
        if text_data is None:
            if text_file is None or tokenizer is None:
                raise ValueError('If text_data is not provided, both text_file'
                                 ' and tokenizer must be given.')
            else:
                # Check if a serialized version exists, otherwise create
                maps = torch.load(self.maps_path)
                ent_ids = maps['ent_ids']

                # Read descriptions, and build a map from entity ID to text
                ent2text = dict()
                text = open(text_file)
                for line in text:
                    name_start = line.find(' ')
                    name_end = line.find(DELIM)
                    entity = line[:name_start].strip()
                    # TODO: Tokenize and encode
                    # description = tokenizer.encode()
                    ent2text[ent_ids[entity]] = line[name_start:name_end].strip()

                # TODO: store indices in tensor
                self.text_data = torch.tensor([1, 2, 3])

                # Save tensor for serialization
        else:
            self.text_data = text_data


if __name__ == '__main__':
    gtr = TextGraphDataset('data/wikifb15k-237/train.txt',
                           ents_file='data/wikifb15k-237/entities.txt',
                           rels_file='data/wikifb15k-237/relations.txt',
                           text_file='data/wikifb15k-237/descriptions.txt')
    gva = TextGraphDataset('data/wikifb15k-237/valid.txt',
                           text_data=gtr.text_data)
    gte = TextGraphDataset('data/wikifb15k-237/test.txt',
                           text_data=gtr.text_data)

    print('Done')
