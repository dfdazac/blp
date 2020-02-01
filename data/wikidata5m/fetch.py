"""Fetch the first section of a Wikipedia page given a list of Wikidata
entities. The list is retrieved from entities.txt
"""

import requests
from tqdm import tqdm
import time

# Maximum number of entities allowed by Wiki APIs
MAX_ENTITIES = 50

WIKIDATA_BASE_URL = 'https://www.wikidata.org/w/api.php' \
                    '?action=wbgetentities&props=sitelinks|labels' \
                    '&languages=en&sitefilter=enwiki&format=json'

WIKIPEDIA_BASE_URL = 'https://en.wikipedia.org/w/api.php' \
                     '?format=json&action=query&prop=extracts&exintro' \
                     '&explaintext&redirects=1'

IN_FNAME = 'ent-debug.txt'
OUT_FNAME = 'descriptions.txt'
NO_FNAME = 'no_wiki.txt'


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
            return requests.get(url, params)
        except requests.exceptions.ConnectionError:
            time.sleep(delay)
            continue


f = open(IN_FNAME)
entities = []

# Read entities to fetch
for i, line in enumerate(f):
    entities.append(line.rstrip('\n'))

no_wiki_count = 0
fetched_count = 0
out_file = open(OUT_FNAME, 'w')
no_wifi_file = open(NO_FNAME, 'w')

print('Retrieving Wikipedia page titles...', flush=True)
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
        r = requests.get(req_url, params={**text_data['continue'],
                                          'titles': titles_param})
        text_data = r.json()
        extracts.update(get_extracts_from_pages(text_data['query']['pages']))

    # Save to file
    for (entity, title) in ent_pages:
        # If there was a redirect, change title accordingly
        if title in redir_titles:
            title = redir_titles[title]

        out_file.write(f'{entity} {title}: {extracts[title]}\n')
        fetched_count += 1


print(f'Retrieved {fetched_count:d} pages.'
      f'There were {no_wiki_count:d} entities with no Wikipedia page.')
print(f'Saved entities and pages in {OUT_FNAME}')
print(f'Saved entities with no pages in {NO_FNAME}')
