import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from sacred.run import Run
from logging import Logger

from data import GraphDataset
from graph import TransE
import utils

ex = utils.create_experiment()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@ex.config
def config():
    dim = 100
    lr = 1e-2
    margin = 1
    p_norm = 1
    max_epochs = 600


@ex.capture
@torch.no_grad()
def evaluate(model, loader, epoch, _run: Run, _log: Logger):
    hits = 0.0
    mrr = 0.0

    all_ents = torch.arange(end=model.num_entities, device=device).unsqueeze(0)
    for triples in loader:
        triples = triples.to(device)
        head, rel, tail = torch.chunk(triples, chunks=3, dim=1)

        # Check all possible heads and tails
        heads_predictions = model.energy(all_ents, tail, rel)
        tails_predictions = model.energy(head, all_ents, rel)

        pred_ents = torch.cat((tails_predictions, heads_predictions))
        true_ents = torch.cat((tail, head))

        hits += utils.hit_at_k(pred_ents, true_ents, k=10)
        mrr += utils.mrr(pred_ents, true_ents)

    hits = hits / len(loader)
    mrr = mrr / len(loader)

    _log.info(f'mrr: {mrr:.4f}  hits@10: {hits:.4f}')
    _run.log_scalar('valid_mrr', mrr, epoch)
    _run.log_scalar('valid_hits@10', hits, epoch)


@ex.automain
def train(dim, lr, margin, p_norm, max_epochs, _run: Run, _log: Logger):
    train_data = GraphDataset('data/wikifb15k-237/train.txt',
                              ents_file='data/wikifb15k-237/entities.txt',
                              rels_file='data/wikifb15k-237/relations.txt')
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True,
                              collate_fn=train_data.negative_sampling,
                              num_workers=4)

    valid_data = GraphDataset('data/wikifb15k-237/valid.txt')
    valid_loader = DataLoader(valid_data, batch_size=128)

    test_data = GraphDataset('data/wikifb15k-237/test.txt')
    test_loader = DataLoader(test_data, batch_size=128)

    model = TransE(train_data.num_ents, train_data.num_rels, dim=dim,
                   margin=margin, p_norm=p_norm).to(device)
    optimizer = Adam(model.parameters(), lr)

    for epoch in range(1, max_epochs + 1):
        train_loss = 0
        for step, data in enumerate(train_loader):
            loss = model(*[tensor.to(device) for tensor in data])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if step % 200 == 0:
                _log.info(f'Epoch {epoch}/{max_epochs} '
                          f'[{step}/{len(train_loader)}]: {loss.item():.6f}')
                _run.log_scalar('batch_loss', loss.item())

        _run.log_scalar('train_loss', train_loss/len(train_loader), epoch)
        _log.info('Evaluating on validation set...')
        evaluate(model, valid_loader, epoch)

    _log.info('Evaluating on test set...')
    evaluate(model, test_loader, max_epochs)
