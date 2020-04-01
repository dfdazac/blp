import os
import os.path as osp
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from sacred.run import Run
from logging import Logger
from sacred import Experiment
from sacred.observers import MongoObserver
from transformers import AlbertTokenizer, get_linear_schedule_with_warmup

from data import TextGraphDataset
from models import BED
import utils

OUT_PATH = 'output/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ex = Experiment()
# Set up database logs
uri = os.environ.get('MLAB_URI')
database = os.environ.get('MLAB_DB')
if all([uri, database]):
    ex.observers.append(MongoObserver(uri, database))


@ex.config
def config():
    encoder_name = 'albert-base-v2'
    max_len = 32
    dim = 128
    lr = 1e-5
    batch_size = 64
    margin = 1
    p_norm = 1
    max_epochs = 20
    num_workers = 4


@ex.capture
@torch.no_grad()
def eval_link_prediction(model, triples_loader, text_dataset, entities,
                         epoch, _run: Run, _log: Logger,
                         prefix='', max_num_batches=None):
    hits = 0.0
    mrr = 0.0
    num_entities = entities.shape[0]
    ent2idx = utils.make_ent2idx(entities).to(device)
    # For TextGraphDataset, preload entity embeddings from their text
    # descriptions, using the text encoder. This is done on mini-batches
    # because entity descriptions have arbitrary length.
    if isinstance(triples_loader.dataset, TextGraphDataset):
        # Create embedding lookup table
        ent_emb = torch.zeros((num_entities, model.dim),
                              dtype=torch.float)
        idx = 0
        while idx < num_entities:
            # Get a batch of entity IDs
            batch_ents = entities[idx:idx + triples_loader.batch_size]
            # Get their corresponding descriptions
            tokens, mask, _ = text_dataset.get_entity_description(batch_ents)
            # Encode with BERT and store result
            batch_emb = model.encode_description(tokens.to(device),
                                                 mask.to(device))
            ent_emb[idx:idx + batch_ents.shape[0]] = batch_emb
            idx += triples_loader.batch_size

        ent_emb = ent_emb.unsqueeze(0).to(device)
    else:
        ent_emb = None

    batch_count = 0
    for i, triples in enumerate(triples_loader):
        if max_num_batches is not None and i == max_num_batches:
            break

        triples = triples.to(device)
        head, tail, rel = torch.chunk(triples, chunks=3, dim=1)

        # Map entity IDs to positions in ent_emb
        head = ent2idx[head]
        tail = ent2idx[tail]

        assert head.min() >= 0
        assert tail.min() >= 0

        # Embed triple
        head_embs = ent_emb.squeeze()[head]
        tail_embs = ent_emb.squeeze()[tail]
        rel_embs = model.rel_emb(rel)

        # Score all possible heads and tails
        heads_predictions = model.energy(ent_emb, tail_embs, rel_embs)
        tails_predictions = model.energy(head_embs, ent_emb, rel_embs)

        pred_ents = torch.cat((tails_predictions, heads_predictions))
        true_ents = torch.cat((tail, head))

        hits += utils.hit_at_k(pred_ents, true_ents, k=10)
        mrr += utils.mrr(pred_ents, true_ents)
        batch_count += 1

    hits = hits / batch_count
    mrr = mrr / batch_count

    _log.info(f'{prefix} mrr: {mrr:.4f}  hits@10: {hits:.4f}')
    _run.log_scalar(f'{prefix}_mrr', mrr, epoch)
    _run.log_scalar(f'{prefix}_hits@10', hits, epoch)

    return ent_emb


@ex.automain
def link_prediction(dim, margin, lr, max_len, batch_size, max_epochs,
                    encoder_name, num_workers, _run: Run, _log: Logger):
    tokenizer = AlbertTokenizer.from_pretrained(encoder_name)
    dataset = TextGraphDataset('data/wikifb15k-237/train-triples.txt',
                               ents_file='data/wikifb15k-237/entities.txt',
                               rels_file='data/wikifb15k-237/relations.txt',
                               text_file='data/wikifb15k-237/descriptions.txt',
                               max_len=max_len, neg_samples=32,
                               tokenizer=tokenizer)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=dataset.collate_text,
                              num_workers=num_workers, drop_last=True)
    train_eval_loader = DataLoader(dataset, batch_size=128)

    valid_data = TextGraphDataset('data/wikifb15k-237/valid-triples.txt',
                                  max_len=max_len)
    valid_loader = DataLoader(valid_data, batch_size=128)

    test_data = TextGraphDataset('data/wikifb15k-237/test-triples.txt',
                                 max_len=max_len)
    test_loader = DataLoader(test_data, batch_size=128)

    train_ent = set(dataset.entities.tolist())
    train_val_ent = set(valid_data.entities.tolist()).union(train_ent)
    train_val_test_ent = set(test_data.entities.tolist()).union(train_val_ent)

    train_ent = torch.tensor(list(train_ent))
    train_val_ent = torch.tensor(list(train_val_ent))
    train_val_test_ent = torch.tensor(list(train_val_test_ent))

    model = BED(dataset.num_rels, dim, margin, encoder_name).to(device)

    optimizer = Adam([{'params': model.encoder.parameters(), 'lr': lr},
                      {'params': model.rel_emb.parameters()},
                      {'params': model.enc_linear.parameters()}],
                     lr=1e-4)

    total_steps = len(train_loader) * max_epochs
    warmup = int(0.2 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup,
                                                num_training_steps=total_steps)

    for epoch in range(1, max_epochs + 1):
        train_loss = 0
        for step, data in enumerate(train_loader):
            loss = model(*[tensor.to(device) for tensor in data])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

            if step % 200 == 0:
                _log.info(f'Epoch {epoch}/{max_epochs} '
                          f'[{step}/{len(train_loader)}]: {loss.item():.6f}')
                _run.log_scalar('batch_loss', loss.item())

        _run.log_scalar('train_loss', train_loss/len(train_loader), epoch)

        _log.info('Evaluating on sample of training set...')
        eval_link_prediction(model, train_eval_loader, dataset, train_ent,
                             epoch, prefix='train',
                             max_num_batches=len(valid_loader))

        _log.info('Evaluating on validation set...')
        eval_link_prediction(model, valid_loader, dataset, train_val_ent,
                             epoch, prefix='valid')

    _log.info('Evaluating on test set...')
    ent_emb = eval_link_prediction(model, test_loader, dataset,
                                   train_val_test_ent, max_epochs,
                                   prefix='test')

    torch.save(model.state_dict(), osp.join(OUT_PATH, f'bed-{_run._id}.pt'))
    torch.save(ent_emb, osp.join(OUT_PATH, f'ent_emb-{_run._id}.pt'))
    torch.save(train_val_test_ent, osp.join(OUT_PATH, f'ents-{_run._id}.pt'))
