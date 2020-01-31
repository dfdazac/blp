"""Fetch the first section of a Wikipedia page given a list of Wikidata
entities. The list is retrieved from entities.txt
"""

import requests
from pprint import pprint
from tqdm import tqdm

# Maximum number of entities allowed by Wiki APIs
MAX_ENTITIES = 50

f = open('entities.txt')
entities = []

WIKIDATA_BASE_REQUEST = 'https://www.wikidata.org/w/api.php' \
                        '?action=wbgetentities&' \
                        '&props=sitelinks&sitefilter=enwiki&format=json'

WIKIPEDIA_BASE_REQUEST = 'https://en.wikipedia.org/w/api.php' \
                         '?format=json&action=query&prop=extracts&exintro' \
                         '&explaintext&redirects=1'

OUT_FNAME = 'descriptions.txt'


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


# Read entities to fetch
for i, line in enumerate(f):
    entities.append(line.rstrip('\n'))
    if i == 1000:
        break

no_wiki_count = 0
out_file = open(OUT_FNAME, 'w')

print('Retrieving Wikipedia page titles...', flush=True)
for i in tqdm(range(0, len(entities), MAX_ENTITIES)):
    # Build request URL from entities
    to_fetch = entities[i:i + MAX_ENTITIES]
    ids_param = '|'.join(to_fetch)

    # Request Wikipedia page titles from Wikidata
    r = requests.get(WIKIDATA_BASE_REQUEST, params={'ids': ids_param})
    link_data = r.json()['entities']
    ent_pages = []

    for e in to_fetch:
        # Proceed only if enwiki page exists
        if 'enwiki' in link_data[e]['sitelinks']:
            title = link_data[e]['sitelinks']['enwiki']['title']
            ent_pages.append((e, title))
        else:
            no_wiki_count += 1

    titles = [title for (e, title) in ent_pages]
    titles_param = '|'.join(titles)
    req_url = WIKIPEDIA_BASE_REQUEST

    # Request first Wikipedia section
    r = requests.get(WIKIPEDIA_BASE_REQUEST, params={'titles': titles_param})

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
        # Was there a redirect?
        if title in redir_titles:
            title = redir_titles[title]

        out_file.write(f'{entity} {title} #### {extracts[title]}\n')


print(f'{no_wiki_count:d} entities with no Wikipedia page.')
print(f'Saved entities to {OUT_FNAME}')
