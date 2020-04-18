import sys
from tqdm import tqdm
from argparse import ArgumentParser
import networkx as nx
import random
import os.path as osp
from collections import Counter
import torch


def parse_triples(triples_file):
    """Read a file containing triples, with head, relation, and tail
    separated by space. Returns list of lists."""
    triples = []
    rel_counts = Counter()
    file = open(triples_file)
    for line in file:
        head, rel, tail = line.split()
        triples.append([head, tail, rel])
        rel_counts[rel] += 1

    return triples, rel_counts


def get_safely_removed_edges(graph, node, rel_counts, min_edges_left=100):
    """Get counts of edge removed by type, after safely removing a given node.
    Safely removing a node entails checking that no nodes are left
    disconnected, and not removing edge types with count less than
    a given amount.
    """
    neighbors = set(nx.all_neighbors(graph, node))
    removed_rel_counts = Counter()
    removed_edges = []

    for m in neighbors:
        # Check if m has more than 2 neighbors (node, and potentially itself)
        # before continuing
        m_neighborhood = set(nx.all_neighbors(graph, m))
        if len(m_neighborhood) > 2:
            # Check edges in both directions between node and m
            pair = [node, m]
            for i in range(2):
                edge_dict = graph.get_edge_data(*pair)
                if edge_dict is not None:
                    # Check that removing the edges between node and m
                    # does not leave less than min_edges_left
                    edges = edge_dict.values()
                    for edge in edges:
                        rel = edge['weight']
                        edges_left = rel_counts[rel] - removed_rel_counts[rel]
                        if edges_left >= min_edges_left:
                            removed_rel_counts[rel] += 1
                            head, tail = pair
                            removed_edges.append((head, tail, rel))
                        else:
                            return None

                # Don't count self-loops twice
                if node == m:
                    break

                pair = list(reversed(pair))
        else:
            return None

    return removed_edges, removed_rel_counts


def drop_entities(triples_file, train_size=0.8, valid_size=0.1, test_size=0.1):
    """Drop entities from a graph, to create training, validation and test
    splits.
    Entities are dropped so that no disconnected nodes are left in the training
    graph. Dropped entities are distributed between disjoint validation
    and test sets.
    """
    if abs(train_size + valid_size + test_size - 1.0) > 1e-9:
        raise ValueError('Split sizes must add to 1.')

    random.seed(0)

    graph = nx.MultiDiGraph()
    triples, rel_counts = parse_triples(triples_file)
    graph.add_weighted_edges_from(triples)
    original_num_edges = graph.number_of_edges()

    print(f'Loaded graph with {graph.number_of_nodes():,} entities '
          f'and {graph.number_of_edges():,} edges')

    dropped_entities = []
    dropped_edges = dict()
    num_to_drop = int(graph.number_of_nodes() * (valid_size + test_size))

    print(f'Removing {num_to_drop:,} entities...')
    progress = tqdm(total=num_to_drop, file=sys.stdout)
    while len(dropped_entities) < num_to_drop:
        # Select a random entity and attempt to remove it
        rand_ent = random.choice(list(graph.nodes))
        removed_tuple = get_safely_removed_edges(graph, rand_ent, rel_counts)

        if removed_tuple is not None:
            removed_edges, removed_counts = removed_tuple
            dropped_edges[rand_ent] = removed_edges
            graph.remove_node(rand_ent)
            dropped_entities.append(rand_ent)
            rel_counts.subtract(removed_counts)
            progress.update(1)

    progress.close()

    # Are there indeed no disconnected nodes?
    assert len(list(nx.isolates(graph))) == 0

    # Did we keep track of removed edges correctly?
    num_removed_edges = sum(map(len, dropped_edges.values()))
    assert num_removed_edges + graph.number_of_edges() == original_num_edges

    split_ratio = test_size/(valid_size + test_size)
    split_idx = int(len(dropped_entities) * split_ratio)
    # Test entities MUST come from first slice! This guarantees that
    # validation entities don't have edges with them (because nodes were
    # removed in sequence)
    test_ents = set(dropped_entities[:split_idx])
    val_ents = set(dropped_entities[split_idx:])

    # Check that val and test entity sets are disjoint
    assert len(val_ents.intersection(test_ents)) == 0

    # Check that training and validation graph do not contain test entities
    val_graph = nx.MultiDiGraph()
    val_edges = []
    for entity in val_ents:
        val_edges += dropped_edges[entity]
    val_graph.add_weighted_edges_from(val_edges)
    assert len(set(val_graph.nodes()).intersection(test_ents)) == 0
    assert len(set(graph.nodes()).intersection(test_ents)) == 0

    names = ('dev', 'test')

    dirname = osp.dirname(triples_file)
    prefix = 'ind-'

    for entity_set, set_name in zip((val_ents, test_ents), names):
        with open(osp.join(dirname, f'{prefix}{set_name}.tsv'), 'w') as file:
            for entity in entity_set:
                triples = dropped_edges[entity]
                for head, tail, rel in triples:
                    file.write(f'{head}\t{rel}\t{tail}\n')

        with open(osp.join(dirname, f'{set_name}-ents.txt'), 'w') as file:
            file.writelines('\n'.join(entity_set))

    with open(osp.join(dirname, f'{prefix}train.tsv'), 'w') as train_file:
        for head, tail, rel in graph.edges(data=True):
            train_file.write(f'{head}\t{rel["weight"]}\t{tail}\n')

    print(f'Dropped {len(val_ents):,} entities for validation'
          f' and {len(test_ents):,} for test.')
    print(f'Saved output files to {dirname}/')


def load_embeddings(embs_file):
    """Read a file containing a word followed by its embedding, as float values
    separated by whitespace.

    Args:
        embs_file: str, path to file

    Returns:
        tensor of shape (vocabulary, embedding_dimension), type torch.float
        dict, mapping words (str) to id (int).
    """
    filename, _ = osp.splitext(embs_file)

    word2idx = {}
    word_embeddings = []
    progress = tqdm()
    with open(embs_file) as file:
        for i, line in enumerate(file):
            word, *embedding = line.split(' ')
            word2idx[word] = i
            word_embeddings.append([float(e) for e in embedding])
            progress.update(1)

    progress.close()

    word_embeddings = torch.tensor(word_embeddings)
    # Add embedding for out-of-vocabulary words
    unk_emb = torch.mean(word_embeddings, dim=0, keepdim=True)
    word_embeddings = torch.cat((word_embeddings, unk_emb))
    word2idx['[UNK]'] = len(word2idx)

    torch.save(word_embeddings, f'{filename}.pt')
    torch.save(word2idx, f'{filename}-maps.pt')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('command', choices=['drop_entities', 'load_embs'])
    parser.add_argument('--file', help='Input file')
    args = parser.parse_args()

    if args.command == 'drop_entities':
        drop_entities(args.file)
    elif args.command == 'load_embs':
        load_embeddings(args.file)
