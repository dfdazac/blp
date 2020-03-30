import sys
import requests
from tqdm import tqdm
import time
import re
from argparse import ArgumentParser
import json
import networkx as nx
import random
import os.path as osp
from collections import Counter

# Maximum number of entities allowed by Wiki APIs
MAX_ENTITIES = 50

WIKIDATA_BASE_URL = 'https://www.wikidata.org/w/api.php' \
                    '?action=wbgetentities&props=sitelinks|labels' \
                    '&languages=en&sitefilter=enwiki&format=json'

WIKIPEDIA_BASE_URL = 'https://en.wikipedia.org/w/api.php' \
                     '?format=json&action=query&prop=extracts&exintro' \
                     '&explaintext&redirects=1'

DELIM = '####'


def get_extracts_from_pages(pages):
    """Return a dictionary mapping Wikipedia titles to their extracts.

    Args:
        pages: A dictionary as returned in a query to the Wikipedia API,
            containing keys: 'pageid', 'ns', 'title', and optionally 'extract'

    Returns: Only pages with an extract key and value
    """
    extracts = {}
    for page in pages:
        if 'extract' in pages[page]:
            # Extract text as a single line
            text = pages[page]['extract'].replace('\n', ' ')
            extracts[pages[page]['title']] = text

    return extracts


def get_retry(url, params, delay=5):
    """Call requests.get() with delay to retry, if there is a connection
    error."""
    while True:
        try:
            return requests.get(url, params, timeout=10)
        except requests.exceptions.RequestException:
            time.sleep(delay)
            continue


def read_entities(ent_fname):
    """Extract a list of entities from a text file with one entity per line.

    Args:
        ent_fname: str, name of the file containing entities.
    Returns:
        list, containing entities as str
    """

    in_file = open(ent_fname)
    entities = []

    # Read entities to fetch
    print(f'Reading entities from {ent_fname}')
    for i, line in enumerate(in_file):
        entities.append(line.rstrip('\n'))

    return entities


def retrieve_pages(in_fname):
    out_fname = f'descriptions-{in_fname}'
    no_fname = f'no-wiki-{in_fname}'

    entities = read_entities(in_fname)

    no_wiki_count = 0
    fetched_count = 0
    out_file = open(out_fname, 'w')
    no_wifi_file = open(no_fname, 'w')

    print('Retrieving Wikipedia pages...', flush=True)
    for i in tqdm(range(0, len(entities), MAX_ENTITIES)):
        # Build request URL from entities
        to_fetch = entities[i:i + MAX_ENTITIES]
        ids_param = '|'.join(to_fetch)

        # Request Wikipedia page titles from Wikidata
        r = get_retry(WIKIDATA_BASE_URL, params={'ids': ids_param})
        link_data = r.json()['entities']
        ent_pages = []

        for e in to_fetch:
            ent_data = link_data[e]
            # Check if enwiki page exists
            if 'missing' not in ent_data and 'enwiki' in ent_data['sitelinks']:
                title = link_data[e]['sitelinks']['enwiki']['title']
                ent_pages.append((e, title))
            else:
                no_wiki_count += 1
                no_wifi_file.write(f'{e}\n')

        titles = [title for (e, title) in ent_pages]
        titles_param = '|'.join(titles)
        req_url = WIKIPEDIA_BASE_URL

        # Request first Wikipedia section
        r = get_retry(WIKIPEDIA_BASE_URL, params={'titles': titles_param})

        text_data = r.json()
        extracts = get_extracts_from_pages(text_data['query']['pages'])
        redirects = text_data['query'].get('redirects')
        redir_titles = {}
        if redirects:
            redir_titles = {r['from']: r['to'] for r in redirects}

        # Usually only some results are returned, so request continuation
        while 'continue' in text_data:
            r = get_retry(req_url, params={**text_data['continue'],
                                           'titles': titles_param})
            text_data = r.json()
            extracts.update(get_extracts_from_pages(text_data['query']['pages']))

        # Save to file
        for (entity, title) in ent_pages:
            # If there was a redirect, change title accordingly
            if title in redir_titles:
                title = redir_titles[title]

            if title in extracts:
                out_file.write(f'{entity} {title} {DELIM} {extracts[title]}\n')
                fetched_count += 1
            else:
                # This might mean Wikidata reported a Wikipedia page, but
                # it actually doesn't exist
                no_wiki_count += 1
                no_wifi_file.write(f'{entity}\n')

    print(f'Retrieved {fetched_count:d} pages.'
          f'There were {no_wiki_count:d} entities with no Wikipedia page.')
    print(f'Saved entities and pages in {out_fname}')
    print(f'Saved entities with no pages in {no_fname}')


def discard_triples(triples_fname, ent_fname):
    """Read a file with triples and save a copy, keeping only triples involving
    entities listed in a separate file.

    Args:
        triples_fname: str, file with one triple per line
        ent_fname: str, file with one entity per line
    """
    triples_file = open(triples_fname)
    out_file = open(f'{triples_fname}.filt', 'w')

    entities = set(read_entities(ent_fname))

    print('Filtering triples...')
    for line in triples_file:
        head, rel, tail = line.split()
        if head in entities and tail in entities:
            out_file.write(f'{head} {rel} {tail}\n')

    print(f'Saved filtered triples in {triples_fname}.filt')


def discard_descriptions(desc_file, ent_fname):
    entities = set(read_entities(ent_fname))

    descriptions = open(desc_file)
    out_file = open(f'{desc_file}.disc', 'w')

    for line in descriptions:
        ent_id = line[:line.find(' ')].strip()
        if ent_id in entities:
            out_file.write(line)


def clean(in_fname, min_tokens=5):
    """Read a file with entity descriptions, and save a clean copy with:
    - No entities with less than min_tokens words in the description
    - Long spaces collapsed to a single one
    """
    in_file = open(in_fname)
    out_file = open(f'{in_fname}.clean', 'w')

    rex = re.compile(r'\s+')

    for line in in_file:
        desc_idx = line.find(DELIM) + len(DELIM)
        prefix, desc = line[:desc_idx], line[desc_idx:]
        desc = rex.sub(' ', desc).strip()
        desc = desc.replace(u'\u200b', '')
        desc = desc.replace(u'\u200e', '')
        tokens = desc.split(' ')
        if len(tokens) >= min_tokens:
            out_file.write(f'{prefix} {desc}\n')


def get_wikidata_freebase_map(in_file):
    """Create a dictionary from Wikidata to Freebase IDs, using
    entity2wikidata,json
    """
    with open(in_file) as f:
        fb_json = json.load(f)

    wikidata2fb = dict()
    for fb_id in fb_json:
        wikidata2fb[fb_json[fb_id]['wikidata_id']] = fb_id

    return wikidata2fb


def write_descriptions(in_file, desc_file):
    """Retrieve Wikipedia descriptions for Freebase entities listed in
    in_file."""
    wikidata2fb = get_wikidata_freebase_map(in_file)
    wikipedia = open(desc_file)
    out_file = open(f'{in_file}.desc', 'w')

    for line in wikipedia:
        title_idx = line.find(' ')
        wikidata_id = line[:title_idx].strip()
        if wikidata_id in wikidata2fb:
            description = line[title_idx:]
            out_file.write(f'{wikidata2fb[wikidata_id]}{description}')


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

    names = ('valid', 'test')

    dirname = osp.dirname(triples_file)
    basename = osp.basename(triples_file)

    for entity_set, set_name in zip((val_ents, test_ents), names):
        with open(osp.join(dirname, f'{set_name}-{basename}'), 'w') as file:
            for entity in entity_set:
                triples = dropped_edges[entity]
                for head, tail, rel in triples:
                    file.write(f'{head} {rel} {tail}\n')

        with open(osp.join(dirname, f'{set_name}-ents.txt'), 'w') as file:
            file.writelines('\n'.join(entity_set))

    with open(osp.join(dirname, f'train-{basename}'), 'w') as train_file:
        for head, tail, rel in graph.edges(data=True):
            train_file.write(f'{head} {rel["weight"]} {tail}\n')

    print(f'Dropped {len(val_ents):,} entities for validation'
          f' and {len(test_ents):,} for test.')
    print(f'Saved output files to {dirname}/')


if __name__ == '__main__':
    parser = ArgumentParser(description='Extract Wikipedia pages for a file'
                                        'with a list of Wikidata entities.')
    parser.add_argument('command', choices=['fetch', 'clean', 'discard',
                                            'describe', 'discard_desc',
                                            'drop_entities'])
    parser.add_argument('--in_file', help='File with a list of entities')
    parser.add_argument('--triples_file', help='File with a list of triples')
    parser.add_argument('--desc_file', help='File with Wikipedia descriptions')
    args = parser.parse_args()

    if args.command == 'fetch':
        retrieve_pages(args.in_file)
    elif args.command == 'clean':
        clean(args.in_file)
    elif args.command == 'discard':
        discard_triples(args.triples_file, args.in_file)
    elif args.command == 'describe':
        write_descriptions(args.in_file, args.desc_file)
    elif args.command == 'discard_desc':
        discard_descriptions(args.desc_file, args.in_file)
    elif args.command == 'drop_entities':
        drop_entities(args.in_file)
