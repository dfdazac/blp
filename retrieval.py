import os
import os.path as osp
from collections import defaultdict
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import BertTokenizer
from logging import Logger
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.run import Run
import json
import pytrec_eval
import numpy as np
import scipy.stats
import nltk
from data import DROPPED

from data import GloVeTokenizer
import utils

OUT_PATH = 'output/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ex = Experiment()
ex.logger = utils.get_logger()
# Set up database logs
uri = os.environ.get('DB_URI')
database = os.environ.get('DB_NAME')
if all([uri, database]):
    ex.observers.append(MongoObserver(uri, database))


def remove_stopwords(text):
    tokens = nltk.word_tokenize(text)
    text = ' '.join([t for t in tokens if
                     t.lower() not in DROPPED])
    return text


@ex.config
def config():
    dim = 128
    model = 'bert-dkrl'
    rel_model = 'transe'
    max_len = 64
    emb_batch_size = 512
    checkpoint = 'output/model-348.pt'
    run_file = 'data/DBpedia-Entity/runs/v2/bm25f-ca_v2.run'
    queries_file = 'data/DBpedia-Entity/collection/v2/queries-v2_stopped.txt'
    descriptions_file = 'data/DBpedia-Entity/runs/v2/' \
                        'bm25f-ca_v2-descriptions.txt'
    qrels_file = 'data/DBpedia-Entity/collection/v2/qrels-v2.txt'
    folds_file = 'data/DBpedia-Entity/collection/v2/folds/all_queries.json'


@ex.capture
def embed_entities(dim, model, rel_model, max_len, emb_batch_size, checkpoint,
                   run_file, descriptions_file, drop_stopwords, _log: Logger):
    def encode_batch(batch):
        tokenized_data = tokenizer.batch_encode_plus(batch,
                                                     max_length=max_len,
                                                     pad_to_max_length=True,
                                                     return_token_type_ids=False,
                                                     return_tensors='pt')
        tokens = tokenized_data['input_ids'].to(device)
        masks = tokenized_data['attention_mask'].float().to(device)

        return encoder.encode(tokens.to(device), masks.to(device))

    if model.startswith('bert') or model == 'blp':
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    else:
        tokenizer = GloVeTokenizer('data/glove/glove.6B.300d-maps.pt')

    encoder = utils.get_model(model, dim, rel_model,
                              encoder_name='bert-base-cased',
                              loss_fn='margin', num_entities=0,
                              num_relations=1, regularizer=0.0).to(device)
    encoder = torch.nn.DataParallel(encoder)
    state_dict = torch.load(checkpoint, map_location=device)

    # We don't need relation embeddings for this task
    state_dict.pop('module.rel_emb.weight', None)
    encoder.load_state_dict(state_dict, strict=False)
    encoder = encoder.module
    for param in encoder.parameters():
        param.requires_grad = False

    # Encode entity descriptions
    run_file_name = osp.splitext(osp.basename(run_file))[0]
    get_entity_embeddings = True
    qent_checkpoint = osp.join(osp.dirname(checkpoint),
                               f'{run_file_name}-qent-{osp.basename(checkpoint)}')
    if osp.exists(qent_checkpoint):
        _log.info(f'Loading entity embeddings from {qent_checkpoint}')
        ent_embeddings = torch.load(qent_checkpoint, map_location=device)
        get_entity_embeddings = False
    else:
        ent_embeddings = []

    entity2idx = dict()
    descriptions_batch = []
    progress = tqdm(desc='Encoding entity descriptions',
                    disable=not get_entity_embeddings)
    with open(descriptions_file) as f:
        for i, line in enumerate(f):
            values = line.strip().split('\t')
            entity = values[0]
            entity2idx[entity] = i

            if get_entity_embeddings:
                text = ' '.join(values[1:])
                if drop_stopwords:
                    text = remove_stopwords(text)
                descriptions_batch.append(text)

                if len(descriptions_batch) == emb_batch_size:
                    embedding = encode_batch(descriptions_batch)
                    ent_embeddings.append(embedding)
                    descriptions_batch = []
                    progress.update(emb_batch_size)

        if get_entity_embeddings:
            if len(descriptions_batch) > 0:
                embedding = encode_batch(descriptions_batch)
                ent_embeddings.append(embedding)

            ent_embeddings = torch.cat(ent_embeddings)
            torch.save(ent_embeddings, qent_checkpoint)
            _log.info(f'Saved entity embeddings to {qent_checkpoint}')

        progress.close()

    return ent_embeddings, entity2idx, encoder, tokenizer


def rerank_on_fold(fold, qrels, baseline_run, id2query, tokenizer, encoder,
                   entity2idx, ent_embeddings, alpha, drop_stopwords):
    train_run = dict()
    qrel_run = dict()
    for query_id in fold:
        results = baseline_run[query_id]

        # Encode query
        query = id2query[query_id]
        if drop_stopwords:
            query = remove_stopwords(query)
        query_tokens = tokenizer.encode(query, return_tensors='pt',
                                        max_length=64)
        query_embedding = encoder.encode(query_tokens.to(device),
                                         text_mask=None)

        # Get embeddings of entities to rerank for this query
        ent_ids_to_rerank = []
        original_scores = []
        selected_results = []
        missing_results = []
        missing_scores = []
        for entity, orig_score in results.items():
            if entity in entity2idx:
                ent_ids_to_rerank.append(entity2idx[entity])
                original_scores.append(orig_score)
                selected_results.append(entity)
            else:
                missing_results.append(entity)
                missing_scores.append(orig_score)

        candidate_embeddings = ent_embeddings[ent_ids_to_rerank]

        candidate_embeddings = F.normalize(candidate_embeddings, dim=-1)
        query_embedding = F.normalize(query_embedding, dim=-1)

        # Compute relevance
        scores = candidate_embeddings @ query_embedding.t()
        scores = scores.flatten().cpu().tolist() + [0] * len(missing_scores)

        results_scores = zip(selected_results + missing_results,
                             scores,
                             original_scores + missing_scores)
        results_scores = [[result, alpha * s1 + (1 - alpha) * s2] for
                          result, s1, s2 in results_scores]

        train_run[query_id] = {r: s for r, s in results_scores}
        qrel_run[query_id] = qrels[query_id]

    evaluator = pytrec_eval.RelevanceEvaluator(qrel_run, {'ndcg_cut_100'})
    train_results = evaluator.evaluate(train_run)
    mean = np.mean([res['ndcg_cut_100'] for res in train_results.values()])

    return mean, train_run


@ex.automain
def rerank(model, rel_model, run_file, queries_file, qrels_file, folds_file,
           _run: Run, _log: Logger):
    drop_stopwords = model in {'bert-bow', 'bert-dkrl',
                               'glove-bow', 'glove-dkrl'}

    ent_embeddings, entity2idx, encoder, tokenizer = embed_entities(
        drop_stopwords=drop_stopwords)

    # Read queries
    id2query = dict()
    with open(queries_file) as f:
        for line in f:
            values = line.strip().split('\t')
            query_id = values[0]
            query = ' '.join(values[1:])
            id2query[query_id] = query

    # Read baseline and ground truth rankings
    baseline_run = defaultdict(dict)
    qrels = defaultdict(dict)
    for query_dict, file in ((baseline_run, run_file),
                             (qrels, qrels_file)):
        with open(file) as f:
            for line in f:
                values = line.strip().split()
                if len(values) >= 6:
                    query_id, q0, entity, rank, score, *_ = values
                    score = float(score)
                else:
                    query_id, q0, entity, score = values
                    score = int(score)
                query_dict[query_id][entity] = score

    # Read query folds
    with open(folds_file) as f:
        folds = json.load(f)

    # Keep only query type of interest
    new_baseline_run = {}
    new_qrels = {}
    for f in folds.values():
        relevant_queries = f['testing']
        for query_id in relevant_queries:
            new_baseline_run.update({query_id: baseline_run[query_id]})
            new_qrels.update({query_id: qrels[query_id]})
    baseline_run = new_baseline_run
    qrels = new_qrels

    # Choose best reranking on training set
    alpha_choices = np.linspace(0, 1, 20)
    test_run = dict()
    for i, (idx, fold) in enumerate(folds.items()):
        train_queries = fold['training']

        best_result = 0.0
        best_alpha = alpha_choices[0]
        for alpha in alpha_choices:
            result, _ = rerank_on_fold(train_queries, qrels,
                                       baseline_run, id2query, tokenizer,
                                       encoder, entity2idx, ent_embeddings,
                                       alpha, drop_stopwords)
            if result > best_result:
                best_result = result
                best_alpha = alpha

        _log.info(f'[Fold {i + 1}/{len(folds)}]'
                  f' Best training result: {best_result:.3f}'
                  f' with alpha={best_alpha:.3}')

        test_queries = fold['testing']
        fold_mean, fold_run = rerank_on_fold(test_queries, qrels,
                                             baseline_run, id2query,
                                             tokenizer, encoder, entity2idx,
                                             ent_embeddings, best_alpha,
                                             drop_stopwords)

        _log.info(f'Test fold result: {fold_mean:.3f}')

        test_run.update(fold_run)

    _log.info(f'Finished hyperparameter search')
    _log.info(f'Saving run file')
    output_run_path = osp.join(OUT_PATH, f'{_run._id}.run')
    with open(output_run_path, 'w') as f:
        for query, results in test_run.items():
            ranking = sorted(results.items(), key=lambda x: x[1], reverse=True)
            for i, (entity, score) in enumerate(ranking):
                f.write(
                    f'{query} Q0 {entity} {i + 1} {score} {model}-{rel_model}\n')

    metrics = {'ndcg_cut_10', 'ndcg_cut_100'}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)
    baseline_results = evaluator.evaluate(baseline_run)
    # This shouldn't be necessary, but there seems to be a bug that requires
    # to instantiate the evaluator again, otherwise only one metric is obtained
    # See https://github.com/cvangysel/pytrec_eval/issues/22
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)
    test_results = evaluator.evaluate(test_run)

    for metric in metrics:
        baseline_mean = np.mean(
            [res[metric] for res in baseline_results.values()])
        test_mean = np.mean([res[metric] for res in test_results.values()])

        _log.info(f'Metric: {metric}')
        _log.info(f'Baseline result: {baseline_mean:.3f}')
        _log.info(f'Test result: {test_mean:.3f}')

        first_scores = [baseline_results[query_id][metric] for query_id in
                        baseline_results]
        second_scores = [test_results[query_id][metric] for query_id in
                         baseline_results]

        _log.info(scipy.stats.ttest_rel(first_scores, second_scores))
