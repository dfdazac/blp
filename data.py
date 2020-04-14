import os
import os.path as osp
import torch
from torch.utils.data import Dataset
import transformers
import string
import nltk
from nltk.corpus import stopwords

UNK = '[UNK]'
DELIM = '####'

while True:
    try:
        STOP_WORDS = stopwords.words('english')
        break
    except LookupError:
        nltk.download('stopwords')
        nltk.download('punkt')
        continue

DROPPED = STOP_WORDS + list(string.punctuation)


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
        self.out_path = osp.join(base_path, 'processed')
        maps_path = osp.join(self.out_path, 'maps.pt')

        if not osp.exists(self.out_path):
            os.mkdir(self.out_path)

        # Create or load maps from entity and relation strings to unique IDs
        if not ents_file or not rels_file:
            if not osp.exists(maps_path):
                raise ValueError('Maps file not found.')

            maps = torch.load(maps_path)
            ent_ids, rel_ids = maps['ent_ids'], maps['rel_ids']
        else:
            ent_ids = file_to_ids(ents_file)
            rel_ids = file_to_ids(rels_file)

        entities = set()
        relations = set()

        # Read triples and store as ints in tensor
        file = open(triples_file)
        triples = []
        for i, line in enumerate(file):
            head, rel, tail = line.split()
            entities.update([head, tail])
            relations.add(rel)
            triples.append([ent_ids[head], ent_ids[tail], rel_ids[rel]])

        self.triples = torch.tensor(triples, dtype=torch.long)

        # Save maps for reuse
        torch.save({'ent_ids': ent_ids, 'rel_ids': rel_ids}, maps_path)

        self.num_ents = len(entities)
        self.num_rels = len(relations)
        self.entities = torch.tensor([ent_ids[ent] for ent in entities])
        self.num_triples = self.triples.shape[0]
        self.maps_path = maps_path

    def __getitem__(self, index):
        return self.triples[index]

    def __len__(self):
        return self.num_triples

    def negative_sampling(self, data_list):
        """Given a batch of triples, return it together with a batch of
        corrupted triples where either the subject or object are replaced
        by a random entity. Use as a collate_fn for a DataLoader.
        """
        pos_pairs, rels = torch.stack(data_list).split(2, dim=1)
        num_triples = pos_pairs.shape[0]

        # Randomly swap head or tail for a random entity
        corrupt_pos = torch.randint(high=2, size=[num_triples])
        corrupt_idx = torch.randint(high=self.num_ents, size=[num_triples])
        corrupt_ent = self.entities[corrupt_idx]
        triple_idx = torch.arange(num_triples)

        neg_pairs = pos_pairs.clone()
        neg_pairs[triple_idx, corrupt_pos] = corrupt_ent

        return pos_pairs, neg_pairs, rels


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
        drop_stopwords: bool
    """
    def __init__(self, triples_file, ents_file=None, rels_file=None,
                 text_file=None, max_len=None, neg_samples=1,
                 tokenizer=None, drop_stopwords=False):
        super().__init__(triples_file, ents_file, rels_file)

        if text_file is not None or tokenizer is not None:
            maps = torch.load(self.maps_path)
            ent_ids = maps['ent_ids']

            # Read descriptions, and build a map from entity ID to text
            text_lines = open(text_file)
            if max_len is None:
                max_len = tokenizer.max_len

            self.text_data = torch.zeros((len(ent_ids), max_len + 1),
                                         dtype=torch.long)
            self.name_data = torch.zeros((len(ent_ids), max_len + 1),
                                         dtype=torch.long)

            for line in text_lines:
                name_start = line.find(' ')
                name_end = line.find(DELIM)

                name = line[name_start:name_end].strip()
                text = line[name_end + len(DELIM):].strip()

                entity = line[:name_start].strip()
                ent_id = ent_ids[entity]

                if drop_stopwords:
                    tokens = nltk.word_tokenize(text)
                    text = ' '.join([t for t in tokens if t.lower() not in DROPPED])

                text_tokens = tokenizer.encode(text,
                                               max_length=max_len,
                                               return_tensors='pt')
                name_tokens = tokenizer.encode(name,
                                               max_length=max_len,
                                               return_tensors='pt')

                text_len = text_tokens.shape[1]
                name_len = name_tokens.shape[1]

                # Starting slice of row contains token IDs
                self.name_data[ent_id, :name_len] = name_tokens
                self.text_data[ent_id, :text_len] = text_tokens
                # Last cell contains sequence length
                self.name_data[ent_id, -1] = name_len
                self.text_data[ent_id, -1] = text_len

        self.neg_samples = neg_samples

    @staticmethod
    def get_batch_data(data, ent_ids):
        # Convert ent_ids to tensor of token IDs and sequence length
        # Shape: (*, L + 1), where L is the maximum length sequence
        seq_data = data[ent_ids]
        max_len = seq_data.shape[-1] - 1

        # Separate tokens from lengths
        tokens, seq_len = seq_data.split(max_len, dim=-1)
        max_batch_len = seq_len.max()
        # Truncate batch
        tokens = tokens[..., :max_batch_len]
        masks = (tokens > 0).float()

        return tokens, masks, seq_len

    def get_entity_name_description(self, ent_ids):
        """Retrieve a batch of sequences (entity descriptions),
        for each entity in the input.

        Args:
            ent_ids: torch.Tensor of type torch.Long, of arbitrary shape (*)

        Returns:
            tokens: torch.Tensor of type torch.Long, of shape (*, L) where L
                is the maximum sequence length in the batch.
            masks: torch.Tensor of type torch.Float, of shap (*, L)
                containing 0 in positions of tokens with padding,
                and 1 otherwise.
        """
        name_tok, name_mask, name_len = self.get_batch_data(self.name_data,
                                                            ent_ids)
        text_tok, text_mask, text_len = self.get_batch_data(self.text_data,
                                                            ent_ids)

        return name_tok, name_mask, name_len, text_tok, text_mask, text_len

    def get_entity_description(self, ent_ids):
        text_tok, text_mask, text_len = self.get_batch_data(self.text_data,
                                                            ent_ids)
        return text_tok, text_mask, text_len

    def collate_text(self, data_list):
        """Given a batch of triples, return it in the form of
        entity descriptions, and the relation types between them.
        Use as a collate_fn for a DataLoader.
        """
        batch_size = len(data_list)
        if batch_size <= 1:
            raise ValueError('collate_text can only work with batch sizes'
                             ' larger than 1.')

        pos_pairs, rels = torch.stack(data_list).split(2, dim=1)
        text_tok, text_mask, text_len = self.get_entity_description(pos_pairs)

        # Obtain indices for negative sampling within the batch
        num_ents = batch_size * 2
        idx = torch.arange(num_ents).reshape(batch_size, 2)
        zeros = torch.zeros(batch_size, 2)

        head_weights = torch.ones(batch_size, num_ents, dtype=torch.float)
        head_weights.scatter_(1, idx, zeros)
        head_idx = head_weights.multinomial(self.neg_samples, replacement=True)

        tail_weights = (head_weights < 1).float()
        tail_idx = tail_weights.multinomial(self.neg_samples, replacement=True)

        neg_idx = torch.stack((head_idx, tail_idx), dim=1)

        return text_tok, text_mask, rels, neg_idx

    def negative_sampling(self, data_list):
        pos_pairs, neg_pairs, rels = super().negative_sampling(data_list)
        split = pos_pairs.shape[0]
        pairs = torch.cat((pos_pairs, neg_pairs))

        tokens, masks = self.get_entity_name_description(pairs)

        # Recover positive/negative split
        pos_pairs_tokens, neg_pairs_tokens = tokens.split(split, dim=0)
        pos_pairs_masks, neg_pairs_masks = masks.split(split, dim=0)

        return (pos_pairs_tokens, pos_pairs_masks,
                neg_pairs_tokens, neg_pairs_masks, rels)

    def graph_negative_sampling(self, data_list):
        return super().negative_sampling(data_list)


class GloVeTokenizer:
    def __init__(self, vocab_dict_file):
        self.word2idx = torch.load(vocab_dict_file)

    def encode(self, text, max_length, return_tensors):
        tokens = nltk.word_tokenize(text)
        encoded = [self.word2idx.get(t, self.word2idx[UNK]) for t in tokens]
        encoded = [encoded[:max_length]]

        if return_tensors:
            encoded = torch.tensor(encoded)

        return encoded


def test_text_graph_dataset():
    tok = transformers.AlbertTokenizer.from_pretrained('albert-base-v2')
    gtr = TextGraphDataset('data/wikifb15k-237/train-triples.txt',
                           ents_file='data/wikifb15k-237/entities.txt',
                           rels_file='data/wikifb15k-237/relations.txt',
                           text_file='data/wikifb15k-237/descriptions.txt',
                           tokenizer=tok)
    gva = TextGraphDataset('data/wikifb15k-237/valid-triples.txt')
    gte = TextGraphDataset('data/wikifb15k-237/test-triples.txt')

    from torch.utils.data import DataLoader

    loader = DataLoader(gtr, batch_size=8, collate_fn=gtr.collate_text,
                        shuffle=True)
    data = next(iter(loader))

    print('Done')


if __name__ == '__main__':
    test_text_graph_dataset()
