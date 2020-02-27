import os.path as osp
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from sacred.run import Run
from logging import Logger
from transformers import AlbertTokenizer, get_linear_schedule_with_warmup

from data import GraphDataset, TextGraphDataset
from graph import TransE, BERTransE, RelTransE
from text import SummaryModel, EntityAligner
import utils

OUT_PATH = 'output/'

ex = utils.create_experiment()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@ex.config
def config():
    dim = 100
    lr = 1e-5
    margin = 5
    p_norm = 1
    max_epochs = 500
    pooling = 'max'


@ex.capture
@torch.no_grad()
def eval_link_prediction(model, loader, epoch, _run: Run, _log: Logger,
                         prefix='', max_num_batches=None):
    hits = 0.0
    mrr = 0.0

    all_ents = torch.arange(end=model.num_entities, device=device).unsqueeze(0)

    # For TextGraphDataset, preload entity embeddings from their text
    # descriptions, using the text encoder. This is done on mini-batches
    # because entity descriptions have arbitrary length.
    if isinstance(loader.dataset, TextGraphDataset):
        # Create embedding lookup table
        ent_emb = torch.zeros((model.num_entities, model.dim),
                              dtype=torch.float)
        idx = 0
        while idx < model.num_entities:
            # Get a batch of entity IDs
            batch_ents = all_ents[0][idx:idx + loader.batch_size]
            # Get their corresponding descriptions
            tokens, masks = loader.dataset.get_entity_descriptions(batch_ents)
            # Encode with BERT and store result
            batch_emb = model.ent_emb(tokens.to(device), masks.to(device))
            ent_emb[idx:idx + loader.batch_size] = batch_emb
            idx += loader.batch_size

        ent_emb = ent_emb.to(device)
    else:
        ent_emb = None

    batch_count = 0
    for i, triples in enumerate(loader):
        if max_num_batches is not None and i == max_num_batches:
            break

        triples = triples.to(device)
        head, tail, rel = torch.chunk(triples, chunks=3, dim=1)

        # Check all possible heads and tails
        heads_predictions = model.energy_eval(all_ents, tail, rel, ent_emb)
        tails_predictions = model.energy_eval(head, all_ents, rel, ent_emb)

        pred_ents = torch.cat((tails_predictions, heads_predictions))
        true_ents = torch.cat((tail, head))

        hits += utils.hit_at_k(pred_ents, true_ents, k=10)
        mrr += utils.mrr(pred_ents, true_ents)
        batch_count += 1

    hits = hits / batch_count
    mrr = mrr / batch_count

    _log.info(f'{prefix} mrr: {mrr:.4f}  hits@10: {hits:.4f}')
    _run.log_scalar(f'{prefix}_valid_mrr', mrr, epoch)
    _run.log_scalar(f'{prefix}_valid_hits@10', hits, epoch)


@ex.command
def link_prediction(dim, lr, margin, p_norm, max_epochs,
                    _run: Run, _log: Logger):
    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    dataset = TextGraphDataset(triples_file='data/wikifb15k-237/train.txt',
                               ents_file='data/wikifb15k-237/entities.txt',
                               rels_file='data/wikifb15k-237/relations.txt',
                               text_file='data/wikifb15k-237/descriptions.txt',
                               tokenizer=tokenizer)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True,
                              collate_fn=dataset.negative_sampling,
                              num_workers=4)
    train_eval_loader = DataLoader(dataset, batch_size=128)

    valid_data = TextGraphDataset('data/wikifb15k-237/valid.txt',
                                  text_data=dataset.text_data)
    valid_loader = DataLoader(valid_data, batch_size=128)

    test_data = TextGraphDataset('data/wikifb15k-237/test.txt',
                                 text_data=dataset.text_data)
    test_loader = DataLoader(test_data, batch_size=128)

    model = BERTransE(dataset.num_ents, dataset.num_rels, dim=dim,
                      margin=margin, p_norm=p_norm).to(device)

    optimizer = Adam([{'params': model.bert.albert.parameters(), 'lr': lr},
                      {'params': model.bert.classifier.parameters()},
                      {'params': model.rel_emb.parameters()}],
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
        eval_link_prediction(model, train_eval_loader, epoch, prefix='train',
                             max_num_batches=len(valid_loader))

        _log.info('Evaluating on validation set...')
        eval_link_prediction(model, valid_loader, epoch, prefix='valid')

    _log.info('Evaluating on test set...')
    eval_link_prediction(model, test_loader, max_epochs, prefix='test')


@ex.capture
def get_entity_summaries(dataset, pooling, _run: Run, _log: Logger):
    """Load pretrained entity summaries"""
    summarizer = SummaryModel(pooling).to(device)
    emb_dim = summarizer.encoder.config.hidden_size
    batch_size = 1  # FIXME

    all_ents = torch.arange(dataset.num_ents)
    ent_summaries = torch.zeros((dataset.num_ents, emb_dim)).to(device)

    _log.info('Loading pretrained entity summaries...')
    for i in range(0, dataset.num_ents, batch_size):
        # Get a batch of entity IDs
        batch_ents = all_ents[i:i + batch_size]
        # Get their corresponding descriptions
        tokens, masks = dataset.get_entity_descriptions(batch_ents)
        # Encode with BERT and store result
        batch_emb = summarizer(tokens.to(device), masks.to(device))
        ent_summaries[i:i + batch_size] = batch_emb

        if i % batch_size == 0:
            _log.info(f'[{i + 1}/{dataset.num_ents}]')

    torch.save(ent_summaries, osp.join(OUT_PATH, f'summaries-{_run._id}.pt'))

    return ent_summaries


@ex.automain
def train_linker(lr, margin, p_norm, max_epochs, pooling,
                 _run: Run, _log: Logger):

    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    dataset = TextGraphDataset(triples_file='data/wikifb15k-237/train.txt',
                               ents_file='data/wikifb15k-237/entities.txt',
                               rels_file='data/wikifb15k-237/relations.txt',
                               text_file='data/wikifb15k-237/descriptions.txt',
                               tokenizer=tokenizer)

    loader = DataLoader(dataset, batch_size=256, shuffle=True,
                        collate_fn=dataset.graph_negative_sampling,
                        num_workers=4)

    summaries = get_entity_summaries(dataset, pooling)

    aligner = EntityAligner().to(device)
    graph_model = RelTransE(dataset.num_rels, aligner.dim,
                            margin, p_norm).to(device)

    optimizer = Adam([{'params': aligner.parameters()},
                      {'params': graph_model.parameters()}], lr=lr)

    warmup = int(0.2 * max_epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup,
                                                num_training_steps=max_epochs)

    for i in range(max_epochs):
        # Get entity description
        ent_id = torch.tensor([i], dtype=torch.long)
        tokens, mask = dataset.get_entity_descriptions(ent_id)
        tokens = tokens.to(device)
        mask = mask.to(device)

        # Use alignment model to obtain entity embeddings
        ent_embs = aligner(tokens, mask, summaries)

        # Run 1 epoch of TransE, accumulating gradients
        train_loss = 0
        for step, data in enumerate(loader):
            loss = graph_model(*[tensor.to(device) for tensor in data],
                               ent_embs)

            loss.backward(retain_graph=True)

            train_loss += loss.item()

            if step % 200 == 0:
                _log.info(f'Entity {i + 1}/{dataset.num_ents} '
                          f'[{step}/{len(loader)}]: {loss.item():.6f}')
                _run.log_scalar('batch_loss', loss.item())

        # Check gradient health
        min_grad_all = float('inf')
        max_grad_all = -float('inf')

        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                min_grad = grad.min().item()
                max_grad = grad.max().item()

                if min_grad <= min_grad_all:
                    min_grad_all = min_grad
                if max_grad >= max_grad_all:
                    max_grad_all = max_grad

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        _run.log_scalar('train_loss', train_loss / len(loader), i)

        _run.log_scalar('min_grad', min_grad_all)
        _run.log_scalar('max_grad', max_grad_all)

        # Save parameters
        torch.save(aligner.state_dict(),
                   osp.join(OUT_PATH, f'aligner-{_run._id}.pt'))

        torch.save(graph_model.state_dict(),
                   osp.join(OUT_PATH, f'transe-{_run._id}.pt'))
