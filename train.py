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

from data import CATEGORY_IDS
from data import GraphDataset, TextGraphDataset, GloVeTokenizer
import models
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


@ex.config
def config():
    dataset = 'Wikidata5M'
    inductive = True
    dim = 128
    model = 'bert-dkrl'
    rel_model = 'transe'
    loss_fn = 'margin'
    encoder_name = 'bert-base-cased'
    regularizer = 1e-2
    max_len = 32
    num_negatives = 64
    lr = 1e-3
    use_scheduler = False
    batch_size = 64
    eval_batch_size = 64
    max_epochs = 10
    num_workers = 4
    checkpoint = None


@ex.capture
@torch.no_grad()
def eval_link_prediction(model, triples_loader, text_dataset, entities,
                         epoch, _run: Run, _log: Logger,
                         prefix='', max_num_batches=None,
                         filtering_graph=None, new_entities=None):
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
    while idx < num_entities:
        # Get a batch of entity IDs and encode them
        batch_ents = entities[idx:idx + triples_loader.batch_size]

        if isinstance(model, models.InductiveLinkPrediction):
            # Encode with entity descriptions
            data = text_dataset.get_entity_description(batch_ents)
            text_tok, text_mask, text_len = data
            batch_emb = model.encode(text_tok.to(device), text_mask.to(device))
        else:
            # Encode from lookup table
            batch_emb = model.encode(batch_ents.to(device))

        ent_emb[idx:idx + batch_ents.shape[0]] = batch_emb
        idx += triples_loader.batch_size

    ent_emb = ent_emb.unsqueeze(0)

    batch_count = 0
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

    return ent_emb, mrr


def get_model(model, dim, rel_model, loss_fn, num_entities, num_relations,
              encoder_name, regularizer):
    if model == 'bed':
        return models.BertEmbeddingsLP(dim, rel_model, loss_fn, num_relations,
                                       encoder_name, regularizer)
    elif model == 'bert-bow':
        return models.BOW(rel_model, loss_fn, num_relations, regularizer,
                          encoder_name=encoder_name)
    elif model == 'bert-dkrl':
        return models.DKRL(dim, rel_model, loss_fn, num_relations, regularizer,
                           encoder_name=encoder_name)
    elif model == 'glove-bow':
        return models.BOW(rel_model, loss_fn, num_relations, regularizer,
                          embeddings='data/glove/glove.6B.300d.pt')
    elif model == 'glove-dkrl':
        return models.DKRL(dim, rel_model, loss_fn, num_relations, regularizer,
                           embeddings='data/glove/glove.6B.300d.pt')
    elif model == 'transductive':
        return models.TransductiveLinkPrediction(dim, rel_model, loss_fn,
                                                 num_entities, num_relations,
                                                 regularizer)
    else:
        raise ValueError(f'Unkown model {model}')


@ex.automain
def link_prediction(dataset, inductive, dim, model, rel_model, loss_fn,
                    encoder_name, regularizer, max_len, num_negatives, lr,
                    use_scheduler, batch_size, eval_batch_size,
                    max_epochs, num_workers, checkpoint,
                    _run: Run, _log: Logger):
    drop_stopwords = model in {'bert-bow', 'bert-dkrl',
                               'glove-bow', 'glove-dkrl'}

    prefix = 'ind-' if inductive and model != 'transductive' else ''
    triples_file = f'data/{dataset}/{prefix}train.tsv'
    if model == 'transductive':
        train_data = GraphDataset(triples_file, num_negatives,
                                  write_maps_file=True)
    else:
        if model.startswith('bert') or model == 'bed':
            tokenizer = BertTokenizer.from_pretrained(encoder_name)
        else:
            tokenizer = GloVeTokenizer('data/glove/glove.6B.300d-maps.pt')

        train_data = TextGraphDataset(triples_file, num_negatives,
                                      max_len, tokenizer, drop_stopwords,
                                      write_maps_file=True)

    train_loader = DataLoader(train_data, batch_size, shuffle=True,
                              collate_fn=train_data.collate_fn,
                              num_workers=num_workers, drop_last=True)

    train_eval_loader = DataLoader(train_data, eval_batch_size)

    valid_data = GraphDataset(f'data/{dataset}/{prefix}dev.tsv')
    valid_loader = DataLoader(valid_data, eval_batch_size)

    test_data = GraphDataset(f'data/{dataset}/{prefix}test.tsv')
    test_loader = DataLoader(test_data, eval_batch_size)

    # Build graph with all triples to compute filtered metrics
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

    _run.log_scalar('num_train_entities', len(train_ent))

    train_ent = torch.tensor(list(train_ent))
    train_val_ent = torch.tensor(list(train_val_ent))
    train_val_test_ent = torch.tensor(list(train_val_test_ent))

    model = get_model(model, dim, rel_model, loss_fn, len(train_val_test_ent),
                      train_data.num_rels, encoder_name, regularizer)
    model = model.to(device)

    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location=device))

    optimizer = Adam(model.parameters(), lr=lr)
    total_steps = len(train_loader) * max_epochs
    if use_scheduler:
        warmup = int(0.2 * total_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=warmup,
                                                    num_training_steps=total_steps)
    best_valid_mrr = 0.0
    checkpoint_file = osp.join(OUT_PATH, f'bed-{_run._id}.pt')
    for epoch in range(1, max_epochs + 1):
        train_loss = 0
        for step, data in enumerate(train_loader):
            loss = model(*[tensor.to(device) for tensor in data])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if use_scheduler:
                scheduler.step()

            train_loss += loss.item()

            if step % 500 == 0:
                _log.info(f'Epoch {epoch}/{max_epochs} '
                          f'[{step}/{len(train_loader)}]: {loss.item():.6f}')
                _run.log_scalar('batch_loss', loss.item())

        _run.log_scalar('train_loss', train_loss / len(train_loader), epoch)

        _log.info('Evaluating on sample of training set...')
        eval_link_prediction(model, train_eval_loader, train_data, train_ent,
                             epoch, prefix='train',
                             max_num_batches=len(valid_loader))

        _log.info('Evaluating on validation set...')
        _, val_mrr = eval_link_prediction(model, valid_loader, train_data,
                                          train_val_ent, epoch, prefix='valid')

        # Keep checkpoint of best performing model (based on raw MRR)
        if val_mrr > best_valid_mrr:
            best_valid_mrr = val_mrr
            torch.save(model.state_dict(), checkpoint_file)

    # Evaluate with best performing checkpoint
    if max_epochs > 0:
        model.load_state_dict(torch.load(checkpoint_file))

    _log.info('Evaluating on validation set (with filtering)...')
    eval_link_prediction(model, valid_loader, train_data, train_val_ent,
                         max_epochs + 1, prefix='valid', filtering_graph=graph,
                         new_entities=val_new_ents)

    _log.info('Evaluating on test set...')
    ent_emb, _ = eval_link_prediction(model, test_loader, train_data,
                                      train_val_test_ent, max_epochs + 1,
                                      prefix='test', filtering_graph=graph,
                                      new_entities=test_new_ents)

    # Save final entity embeddings obtained with trained encoder
    torch.save(ent_emb, osp.join(OUT_PATH, f'ent_emb-{_run._id}.pt'))
    torch.save(train_val_test_ent, osp.join(OUT_PATH, f'ents-{_run._id}.pt'))
