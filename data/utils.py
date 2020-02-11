import requests
from tqdm import tqdm
import time
import re
from argparse import ArgumentParser
import json

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


if __name__ == '__main__':
    parser = ArgumentParser(description='Extract Wikipedia pages for a file'
                                        'with a list of Wikidata entities.')
    parser.add_argument('command', choices=['fetch', 'clean', 'discard',
                                            'describe'])
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
