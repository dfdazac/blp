import os
import os.path as osp
import networkx as nx
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from sacred.run import Run
from logging import Logger
from sacred import Experiment
from sacred.observers import MongoObserver
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from collections import defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from data import CATEGORY_IDS
from data import GraphDataset, TextGraphDataset, GloVeTokenizer
import models
import utils

OUT_PATH = 'output/'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

ex = Experiment()
ex.logger = utils.get_logger()
# Set up database logs
uri = os.environ.get('DB_URI')
database = os.environ.get('DB_NAME')
if all([uri, database]):
    ex.observers.append(MongoObserver(uri, database))


@ex.config
def config():
    dataset = 'umls'
    inductive = True
    dim = 128
    model = 'blp'
    rel_model = 'transe'
    loss_fn = 'margin'
    encoder_name = 'bert-base-cased'
    regularizer = 0
    max_len = 32
    num_negatives = 64
    lr = 2e-5
    use_scheduler = True
    batch_size = 64
    emb_batch_size = 512
    eval_batch_size = 64
    max_epochs = 40
    checkpoint = None
    use_cached_text = False


@ex.capture
@torch.no_grad()
def eval_link_prediction(model, triples_loader, text_dataset, entities,
                         epoch, emb_batch_size, _run: Run, _log: Logger,
                         prefix='', max_num_batches=None,
                         filtering_graph=None, new_entities=None,
                         return_embeddings=False):
    compute_filtered = filtering_graph is not None
    mrr_by_position = torch.zeros(3, dtype=torch.float).to(device)
    mrr_pos_counts = torch.zeros_like(mrr_by_position)

    rel_categories = triples_loader.dataset.rel_categories.to(device)
    mrr_by_category = torch.zeros([2, 4], dtype=torch.float).to(device)
    mrr_cat_count = torch.zeros([1, 4], dtype=torch.float).to(device)

    hit_positions = [1, 3, 10]
    hits_at_k = {pos: 0.0 for pos in hit_positions}
    mrr = 0.0
    mrr_filt = 0.0
    hits_at_k_filt = {pos: 0.0 for pos in hit_positions}

    if device != torch.device('cpu'):
        model = model.module

    if isinstance(model, models.InductiveLinkPrediction):
        num_entities = entities.shape[0]
        if compute_filtered:
            max_ent_id = max(filtering_graph.nodes)
        else:
            max_ent_id = entities.max()
        ent2idx = utils.make_ent2idx(entities, max_ent_id)
    else:
        # Transductive models have a lookup table of embeddings
        num_entities = model.ent_emb.num_embeddings
        ent2idx = torch.arange(num_entities)
        entities = ent2idx

    # Create embedding lookup table for evaluation
    ent_emb = torch.zeros((num_entities, model.dim), dtype=torch.float,
                          device=device)
    idx = 0
    num_iters = np.ceil(num_entities / emb_batch_size)
    iters_count = 0
    while idx < num_entities:
        # Get a batch of entity IDs and encode them
        batch_ents = entities[idx:idx + emb_batch_size]

        if isinstance(model, models.InductiveLinkPrediction):
            # Encode with entity descriptions
            data = text_dataset.get_entity_description(batch_ents)
            text_tok, text_mask, text_len = data
            batch_emb = model(text_tok.unsqueeze(1).to(device),
                              text_mask.unsqueeze(1).to(device))
        else:
            # Encode from lookup table
            batch_emb = model(batch_ents)

        ent_emb[idx:idx + batch_ents.shape[0]] = batch_emb

        iters_count += 1
        if iters_count % np.ceil(0.2 * num_iters) == 0:
            _log.info(f'[{idx + batch_ents.shape[0]:,}/{num_entities:,}]')

        idx += emb_batch_size

    ent_emb = ent_emb.unsqueeze(0)

    batch_count = 0
    _log.info('Computing metrics on set of triples')
    total = len(triples_loader) if max_num_batches is None else max_num_batches
    for i, triples in enumerate(triples_loader):
        if max_num_batches is not None and i == max_num_batches:
            break

        heads, tails, rels = torch.chunk(triples, chunks=3, dim=1)
        # Map entity IDs to positions in ent_emb
        heads = ent2idx[heads].to(device)
        tails = ent2idx[tails].to(device)

        assert heads.min() >= 0
        assert tails.min() >= 0

        # Embed triple
        head_embs = ent_emb.squeeze()[heads]
        tail_embs = ent_emb.squeeze()[tails]
        rel_embs = model.rel_emb(rels.to(device))

        # Score all possible heads and tails
        heads_predictions = model.score_fn(ent_emb, tail_embs, rel_embs)
        tails_predictions = model.score_fn(head_embs, ent_emb, rel_embs)

        pred_ents = torch.cat((heads_predictions, tails_predictions))
        true_ents = torch.cat((heads, tails))

        hits = utils.hit_at_k(pred_ents, true_ents, hit_positions)
        for j, h in enumerate(hits):
            hits_at_k[hit_positions[j]] += h
        mrr += utils.mrr(pred_ents, true_ents).mean().item()

        if compute_filtered:
            filters = utils.get_triple_filters(triples, filtering_graph,
                                               num_entities, ent2idx)
            heads_filter, tails_filter = filters
            # Filter entities by assigning them the lowest score in the batch
            filter_mask = torch.cat((heads_filter, tails_filter)).to(device)
            pred_ents[filter_mask] = pred_ents.min() - 1.0
            hits_filt = utils.hit_at_k(pred_ents, true_ents, hit_positions)
            for j, h in enumerate(hits_filt):
                hits_at_k_filt[hit_positions[j]] += h
            mrr_filt_per_triple = utils.mrr(pred_ents, true_ents)
            mrr_filt += mrr_filt_per_triple.mean().item()

            if new_entities is not None:
                by_position = utils.split_by_new_position(triples,
                                                          mrr_filt_per_triple,
                                                          new_entities)
                batch_mrr_by_position, batch_mrr_pos_counts = by_position
                mrr_by_position += batch_mrr_by_position
                mrr_pos_counts += batch_mrr_pos_counts

            if triples_loader.dataset.has_rel_categories:
                by_category = utils.split_by_category(triples,
                                                      mrr_filt_per_triple,
                                                      rel_categories)
                batch_mrr_by_cat, batch_mrr_cat_count = by_category
                mrr_by_category += batch_mrr_by_cat
                mrr_cat_count += batch_mrr_cat_count

        batch_count += 1
        if (i + 1) % int(0.2 * total) == 0:
            _log.info(f'[{i + 1:,}/{total:,}]')

    for hits_dict in (hits_at_k, hits_at_k_filt):
        for k in hits_dict:
            hits_dict[k] /= batch_count

    mrr = mrr / batch_count
    mrr_filt = mrr_filt / batch_count

    log_str = f'{prefix} mrr: {mrr:.4f}  '
    _run.log_scalar(f'{prefix}_mrr', mrr, epoch)
    for k, value in hits_at_k.items():
        log_str += f'hits@{k}: {value:.4f}  '
        _run.log_scalar(f'{prefix}_hits@{k}', value, epoch)

    if compute_filtered:
        log_str += f'mrr_filt: {mrr_filt:.4f}  '
        _run.log_scalar(f'{prefix}_mrr_filt', mrr_filt, epoch)
        for k, value in hits_at_k_filt.items():
            log_str += f'hits@{k}_filt: {value:.4f}  '
            _run.log_scalar(f'{prefix}_hits@{k}_filt', value, epoch)

    _log.info(log_str)

    if new_entities is not None and compute_filtered:
        mrr_pos_counts[mrr_pos_counts < 1.0] = 1.0
        mrr_by_position = mrr_by_position / mrr_pos_counts
        log_str = ''
        for i, t in enumerate((f'{prefix}_mrr_filt_both_new',
                               f'{prefix}_mrr_filt_head_new',
                               f'{prefix}_mrr_filt_tail_new')):
            value = mrr_by_position[i].item()
            log_str += f'{t}: {value:.4f}  '
            _run.log_scalar(t, value, epoch)
        _log.info(log_str)

    if compute_filtered and triples_loader.dataset.has_rel_categories:
        mrr_cat_count[mrr_cat_count < 1.0] = 1.0
        mrr_by_category = mrr_by_category / mrr_cat_count

        for i, case in enumerate(['pred_head', 'pred_tail']):
            log_str = f'{case} '
            for cat, cat_id in CATEGORY_IDS.items():
                log_str += f'{cat}_mrr: {mrr_by_category[i, cat_id]:.4f}  '
            _log.info(log_str)

    if return_embeddings:
        out = (mrr, ent_emb)
    else:
        out = (mrr, None)

    return out


@ex.command
def link_prediction(dataset, inductive, dim, model, rel_model, loss_fn,
                    encoder_name, regularizer, max_len, num_negatives, lr,
                    use_scheduler, batch_size, emb_batch_size, eval_batch_size,
                    max_epochs, checkpoint, use_cached_text,
                    _run: Run, _log: Logger):
    drop_stopwords = model in {'bert-bow', 'bert-dkrl',
                               'glove-bow', 'glove-dkrl'}

    prefix = 'ind-' if inductive and model != 'transductive' else ''
    triples_file = f'data/{dataset}/{prefix}train.tsv'

    if device != torch.device('cpu'):
        num_devices = torch.cuda.device_count()
        if batch_size % num_devices != 0:
            raise ValueError(f'Batch size ({batch_size}) must be a multiple of'
                             f' the number of CUDA devices ({num_devices})')
        _log.info(f'CUDA devices used: {num_devices}')
    else:
        num_devices = 1
        _log.info('Training on CPU')

    if model == 'transductive':
        train_data = GraphDataset(triples_file, num_negatives,
                                  write_maps_file=True,
                                  num_devices=num_devices)
    else:
        if model.startswith('bert') or model == 'blp':
            tokenizer = BertTokenizer.from_pretrained(encoder_name)
        else:
            tokenizer = GloVeTokenizer('data/glove/glove.6B.300d-maps.pt')

        train_data = TextGraphDataset(triples_file, num_negatives,
                                      max_len, tokenizer, drop_stopwords,
                                      write_maps_file=True,
                                      use_cached_text=use_cached_text,
                                      num_devices=num_devices)

    train_loader = DataLoader(train_data, batch_size, shuffle=True,
                              collate_fn=train_data.collate_fn,
                              num_workers=0, drop_last=True)

    train_eval_loader = DataLoader(train_data, eval_batch_size)

    valid_data = GraphDataset(f'data/{dataset}/{prefix}dev.tsv')
    valid_loader = DataLoader(valid_data, eval_batch_size)

    test_data = GraphDataset(f'data/{dataset}/{prefix}test.tsv')
    test_loader = DataLoader(test_data, eval_batch_size)

    # Build graph with all triples to compute filtered metrics
    if dataset != 'Wikidata5M':
        graph = nx.MultiDiGraph()
        all_triples = torch.cat((train_data.triples,
                                 valid_data.triples,
                                 test_data.triples))
        graph.add_weighted_edges_from(all_triples.tolist())

        train_ent = set(train_data.entities.tolist())
        train_val_ent = set(valid_data.entities.tolist()).union(train_ent)
        train_val_test_ent = set(test_data.entities.tolist()).union(train_val_ent)
        val_new_ents = train_val_ent.difference(train_ent)
        test_new_ents = train_val_test_ent.difference(train_val_ent)
    else:
        graph = None

        train_ent = set(train_data.entities.tolist())
        train_val_ent = set(valid_data.entities.tolist())
        train_val_test_ent = set(test_data.entities.tolist())
        val_new_ents = test_new_ents = None

    _run.log_scalar('num_train_entities', len(train_ent))

    train_ent = torch.tensor(list(train_ent))
    train_val_ent = torch.tensor(list(train_val_ent))
    train_val_test_ent = torch.tensor(list(train_val_test_ent))

    model = utils.get_model(model, dim, rel_model, loss_fn,
                            len(train_val_test_ent), train_data.num_rels,
                            encoder_name, regularizer)
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location='cpu'))

    if device != torch.device('cpu'):
        model = torch.nn.DataParallel(model).to(device)

    optimizer = Adam(model.parameters(), lr=lr)
    total_steps = len(train_loader) * max_epochs
    if use_scheduler:
        warmup = int(0.2 * total_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=warmup,
                                                    num_training_steps=total_steps)
    best_valid_mrr = 0.0
    checkpoint_file = osp.join(OUT_PATH, f'model-{_run._id}.pt')
    for epoch in range(1, max_epochs + 1):
        train_loss = 0
        for step, data in enumerate(train_loader):
            loss = model(*data).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if use_scheduler:
                scheduler.step()

            train_loss += loss.item()

            if step % int(0.05 * len(train_loader)) == 0:
                _log.info(f'Epoch {epoch}/{max_epochs} '
                          f'[{step}/{len(train_loader)}]: {loss.item():.6f}')
                _run.log_scalar('batch_loss', loss.item())

        _run.log_scalar('train_loss', train_loss / len(train_loader), epoch)

        if dataset != 'Wikidata5M':
            _log.info('Evaluating on sample of training set')
            eval_link_prediction(model, train_eval_loader, train_data, train_ent,
                                 epoch, emb_batch_size, prefix='train',
                                 max_num_batches=len(valid_loader))

        _log.info('Evaluating on validation set')
        val_mrr, _ = eval_link_prediction(model, valid_loader, train_data,
                                          train_val_ent, epoch,
                                          emb_batch_size, prefix='valid')

        # Keep checkpoint of best performing model (based on raw MRR)
        if val_mrr > best_valid_mrr:
            best_valid_mrr = val_mrr
            torch.save(model.state_dict(), checkpoint_file)

    # Evaluate with best performing checkpoint
    if max_epochs > 0:
        model.load_state_dict(torch.load(checkpoint_file))

    if dataset == 'Wikidata5M':
        graph = nx.MultiDiGraph()
        graph.add_weighted_edges_from(valid_data.triples.tolist())

    _log.info('Evaluating on validation set (with filtering)')
    eval_link_prediction(model, valid_loader, train_data, train_val_ent,
                         max_epochs + 1, emb_batch_size, prefix='valid',
                         filtering_graph=graph,
                         new_entities=val_new_ents)

    if dataset == 'Wikidata5M':
        graph = nx.MultiDiGraph()
        graph.add_weighted_edges_from(test_data.triples.tolist())

    _log.info('Evaluating on test set')
    _, ent_emb = eval_link_prediction(model, test_loader, train_data,
                                      train_val_test_ent, max_epochs + 1,
                                      emb_batch_size, prefix='test',
                                      filtering_graph=graph,
                                      new_entities=test_new_ents,
                                      return_embeddings=True)

    # Save final entity embeddings obtained with trained encoder
    torch.save(ent_emb, osp.join(OUT_PATH, f'ent_emb-{_run._id}.pt'))
    torch.save(train_val_test_ent, osp.join(OUT_PATH, f'ents-{_run._id}.pt'))


@ex.command
def node_classification(dataset, checkpoint, _run: Run, _log: Logger):
    ent_emb = torch.load(f'output/ent_emb-{checkpoint}.pt', map_location='cpu')
    if isinstance(ent_emb, tuple):
        ent_emb = ent_emb[0]

    ent_emb = ent_emb.squeeze().numpy()
    num_embs, emb_dim = ent_emb.shape
    _log.info(f'Loaded {num_embs} embeddings with dim={emb_dim}')

    emb_ids = torch.load(f'output/ents-{checkpoint}.pt', map_location='cpu')
    ent2idx = utils.make_ent2idx(emb_ids, max_ent_id=emb_ids.max()).numpy()
    maps = torch.load(f'data/{dataset}/maps.pt')
    ent_ids = maps['ent_ids']
    class2label = defaultdict(lambda: len(class2label))

    splits = ['train', 'dev', 'test']
    split_2data = dict()
    for split in splits:
        with open(f'data/{dataset}/{split}-ents-class.txt') as f:
            idx = []
            labels = []
            for line in f:
                entity, ent_class = line.strip().split()
                entity_id = ent_ids[entity]
                entity_idx = ent2idx[entity_id]
                idx.append(entity_idx)
                labels.append(class2label[ent_class])

            x = ent_emb[idx]
            y = np.array(labels)
            split_2data[split] = (x, y)

    x_train, y_train = split_2data['train']
    x_dev, y_dev = split_2data['dev']
    x_test, y_test = split_2data['test']

    best_dev_metric = 0.0
    best_c = 0
    for k in range(-4, 2):
        c = 10 ** -k
        model = LogisticRegression(C=c, multi_class='multinomial',
                                   max_iter=1000)
        model.fit(x_train, y_train)

        dev_preds = model.predict(x_dev)
        dev_acc = accuracy_score(y_dev, dev_preds)
        _log.info(f'{c:.3f} - {dev_acc:.3f}')

        if dev_acc > best_dev_metric:
            best_dev_metric = dev_acc
            best_c = c

    _log.info(f'Best regularization coefficient: {best_c:.4f}')
    model = LogisticRegression(C=best_c, multi_class='multinomial',
                               max_iter=1000)
    x_train_all = np.concatenate((x_train, x_dev))
    y_train_all = np.concatenate((y_train, y_dev))
    model.fit(x_train_all, y_train_all)

    for metric_fn in (accuracy_score, balanced_accuracy_score):
        train_preds = model.predict(x_train_all)
        train_metric = metric_fn(y_train_all, train_preds)

        test_preds = model.predict(x_test)
        test_metric = metric_fn(y_test, test_preds)

        _log.info(f'Train {metric_fn.__name__}: {train_metric:.3f}')
        _log.info(f'Test {metric_fn.__name__}: {test_metric:.3f}')


ex.run_commandline()
