import sys
from tqdm import tqdm
from argparse import ArgumentParser
import networkx as nx
import random
import os.path as osp
from collections import Counter, defaultdict
import torch
import rdflib


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


def read_entity_types(entity2type_file):
    type2entities = defaultdict(set)
    with open(entity2type_file) as f:
        for line in f:
            entity, label = line.strip().split()
            type2entities[label].add(entity)

    return dict(type2entities)


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


def drop_entities(triples_file, train_size=0.8, valid_size=0.1, test_size=0.1,
                  seed=0, types_file=None):
    """Drop entities from a graph, to create training, validation and test
    splits.
    Entities are dropped so that no disconnected nodes are left in the training
    graph. Dropped entities are distributed between disjoint validation
    and test sets.
    """
    splits_sum = train_size + valid_size + test_size
    if splits_sum < 0 or splits_sum > 1:
        raise ValueError('Sum of split sizes must be between greater than 0'
                         ' and less than or equal to 1.')

    use_types = types_file is not None
    if use_types:
        type2entities = read_entity_types(types_file)
        types = list(type2entities.keys())

    random.seed(seed)

    graph = nx.MultiDiGraph()
    triples, rel_counts = parse_triples(triples_file)
    graph.add_weighted_edges_from(triples)
    original_num_edges = graph.number_of_edges()
    original_num_nodes = graph.number_of_nodes()

    print(f'Loaded graph with {graph.number_of_nodes():,} entities '
          f'and {graph.number_of_edges():,} edges')

    dropped_entities = []
    dropped_edges = dict()
    num_to_drop = int(original_num_nodes * (1 - train_size))
    num_val = int(original_num_nodes * valid_size)
    num_test = int(original_num_nodes * test_size)

    print(f'Removing {num_to_drop:,} entities...')
    progress = tqdm(total=num_to_drop, file=sys.stdout)
    while len(dropped_entities) < num_to_drop:
        if use_types:
            # Sample an entity with probability proportional to its type count
            # (minus 1 to keep at least one entity of any type)
            weights = [len(type2entities[t]) - 1 for t in types]
            rand_type = random.choices(types, weights, k=1)[0]
            rand_ent = random.choice(list(type2entities[rand_type]))
        else:
            # Sample an entity uniformly at random
            rand_ent = random.choice(list(graph.nodes))

        removed_tuple = get_safely_removed_edges(graph, rand_ent, rel_counts)

        if removed_tuple is not None:
            removed_edges, removed_counts = removed_tuple
            dropped_edges[rand_ent] = removed_edges
            graph.remove_node(rand_ent)
            dropped_entities.append(rand_ent)
            rel_counts.subtract(removed_counts)

            if use_types:
                type2entities[rand_type].remove(rand_ent)

            progress.update(1)

    progress.close()

    # Are there indeed no disconnected nodes?
    assert len(list(nx.isolates(graph))) == 0

    # Did we keep track of removed edges correctly?
    num_removed_edges = sum(map(len, dropped_edges.values()))
    assert num_removed_edges + graph.number_of_edges() == original_num_edges

    # Test entities MUST come from first slice! This guarantees that
    # validation entities don't have edges with them (because nodes were
    # removed in sequence)
    test_ents = set(dropped_entities[:num_test])
    val_ents = set(dropped_entities[num_test:num_test + num_val])
    train_ents = set(graph.nodes())

    # Check that entity sets are disjoint
    assert len(train_ents.intersection(val_ents)) == 0
    assert len(train_ents.intersection(test_ents)) == 0
    assert len(val_ents.intersection(test_ents)) == 0

    # Check that validation graph does not contain test entities
    val_graph = nx.MultiDiGraph()
    val_edges = []
    for entity in val_ents:
        val_edges += dropped_edges[entity]
    val_graph.add_weighted_edges_from(val_edges)
    assert len(set(val_graph.nodes()).intersection(test_ents)) == 0

    names = ('train', 'dev', 'test')

    dirname = osp.dirname(triples_file)
    prefix = 'ind-'

    for entity_set, set_name in zip((train_ents, val_ents, test_ents), names):
        # Save file with entities for set
        with open(osp.join(dirname, f'{set_name}-ents.txt'), 'w') as file:
            file.writelines('\n'.join(entity_set))

        if set_name == 'train':
            # Triples for train split are saved later
            continue

        # Save file with triples for entities in set
        with open(osp.join(dirname, f'{prefix}{set_name}.tsv'), 'w') as file:
            for entity in entity_set:
                triples = dropped_edges[entity]
                for head, tail, rel in triples:
                    file.write(f'{head}\t{rel}\t{tail}\n')

    with open(osp.join(dirname, f'{prefix}train.tsv'), 'w') as train_file:
        for head, tail, rel in graph.edges(data=True):
            train_file.write(f'{head}\t{rel["weight"]}\t{tail}\n')

    print(f'Dropped {len(val_ents):,} entities for validation'
          f' and {len(test_ents):,} for test.')
    print(f'{graph.number_of_nodes():,} entities are left for training.')
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


def categorize_relations(triples_file):
    """Given a set of triples, assign a category to a relation from the
    following:
        1 to 1
        1 to many
        many to 1
        many to many
    Results are saved back to disk.

    Args:
        triples_file: str, file containing triples of the form
            head relation tail
    """
    graph = nx.MultiDiGraph()
    triples, rel_counts = parse_triples(triples_file)
    graph.add_weighted_edges_from(triples)

    rel2heads_count = defaultdict(list)
    rel2tails_count = defaultdict(list)

    for entity in graph.nodes:
        rel2heads_entity_count = Counter()
        # Fix entity as tail, and check all heads
        in_edges = graph.in_edges(entity, data=True)
        for u, v, edge in in_edges:
            rel2heads_entity_count[edge['weight']] += 1

        for rel, counts in rel2heads_entity_count.items():
            rel2heads_count[rel].append(counts)

        rel2tails_entity_count = Counter()
        # Fix entity as head, and check all tails
        out_edges = graph.out_edges(entity, data=True)
        for u, v, edge in out_edges:
            rel2tails_entity_count[edge['weight']] += 1

        for rel, counts in rel2tails_entity_count.items():
            rel2tails_count[rel].append(counts)

    rel2category = dict()
    for rel in rel2heads_count:
        head_counts = rel2heads_count[rel]
        tail_counts = rel2tails_count[rel]

        head_avg = sum(head_counts)/len(head_counts)
        tail_avg = sum(tail_counts)/len(tail_counts)

        head_category = '1' if head_avg < 1.5 else 'many'
        tail_category = '1' if tail_avg < 1.5 else 'many'

        rel2category[rel] = f'{head_category}-to-{tail_category}'

    print('Relation category statistics:')
    cat_counts = Counter(rel2category.values())
    for category, count in cat_counts.items():
        proportion = 100 * count/len(rel2category)
        print(f'{category:13} {count:3}  {proportion:4.1f}%')

    dirname = osp.dirname(triples_file)
    output_path = osp.join(dirname, 'relations-cat.txt')
    with open(output_path, 'w') as f:
        for relation, category in rel2category.items():
            f.write(f'{relation}\t{category}\n')

    print(f'Saved relation categories to {output_path}')


def get_ranking_descriptions(run_file, dbpedia_file, redirects_file):
    # Read run file and get unique set of entities
    print('Reading unique entities from run file...')
    entities = set()
    with open(run_file) as f:
        for line in f:
            values = line.strip().split()
            entities.add(values[2])

    basename = osp.splitext(osp.basename(run_file))[0]
    output_file = osp.join(osp.dirname(run_file),
                           basename + '-descriptions.txt')
    missing_file = osp.join(osp.dirname(run_file), basename + '-missing.txt')

    dbpedia_ns = 'http://dbpedia.org/resource/'
    dbpedia_prefix = 'dbpedia:'

    print('Reading redirects...')
    redir2entities = defaultdict(set)
    with open(redirects_file) as f:
        for line in f:
            values = line.strip().split()
            norm_uri = values[0].replace(dbpedia_ns, dbpedia_prefix, 1)
            redirect = values[2]
            if norm_uri in entities:
                redir2entities[redirect].add(norm_uri)

    # Iterate over DBpedia dump and keep required descriptions
    print('Retrieving descriptions of entities...')

    read_entities = set()
    progress = tqdm(file=sys.stdout)
    with open(dbpedia_file) as f, open(output_file, 'w') as out:
        for line in f:
            g = rdflib.Graph().parse(data=line, format='n3')
            for (page, rel, description) in g:
                norm_uri = f'<{page.replace(dbpedia_ns, dbpedia_prefix, 1)}>'
                if norm_uri in entities and norm_uri not in read_entities:
                    read_entities.add(norm_uri)
                    out.write(f'{norm_uri}\t{description.value}\n')

                page_n3_format = page.n3()
                if page_n3_format in redir2entities:
                    for entity in redir2entities[page_n3_format]:
                        if entity not in read_entities:
                            read_entities.add(entity)
                            out.write(f'{entity}\t{description.value}\n')

            if len(read_entities) == len(entities):
                break

            progress.update()

    progress.close()

    with open(missing_file, 'w') as f:
        for entity in entities.difference(read_entities):
            f.write(f'{entity}\n')

    print(f'Retrieved {len(read_entities):,} descriptions, out of'
          f' {len(entities):,} entities.')
    print(f'Descriptions saved in {output_file}')
    print(f'Entities with missing descriptions saved in {missing_file}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('command', choices=['drop_entities', 'load_embs',
                                            'categorize',
                                            'get_ranking_descriptions'])
    parser.add_argument('--file', help='Input file')
    parser.add_argument('--dbp_file', help='DBpedia ttl file with rdf:comment'
                                           ' field for entities')
    parser.add_argument('--redirects_file', help='File redirecting entities')
    parser.add_argument('--types_file', help='Tab-separated file of entities'
                                             ' and their type', default=None)
    parser.add_argument('--train_size', help='Fraction of entities used for'
                        ' training.', default=0.8, type=float)
    parser.add_argument('--seed', help='Random seed', default=0)
    args = parser.parse_args()

    if args.command == 'drop_entities':
        drop_entities(args.file, train_size=args.train_size, seed=args.seed,
                      types_file=args.types_file)
    elif args.command == 'load_embs':
        load_embeddings(args.file)
    elif args.command == 'categorize':
        categorize_relations(args.file)
    elif args.command == 'get_ranking_descriptions':
        if args.file is None or args.dbp_file is None:
            raise ValueError('--file and --dbp_file must be provided to'
                             ' get_ranking_descriptions')
        get_ranking_descriptions(args.file, args.dbp_file, args.redirects_file)
