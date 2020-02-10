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
    lr = 1e-2
    margin = 1
    p_norm = 1
    epochs = 2


@torch.no_grad()
def evaluate(model, loader):
    hits_at_10 = 0.0
    mrr = 0.0

    all_ents = torch.arange(end=model.num_entities, device=device).unsqueeze(0)
    for triples in loader:
        triples = triples.to(device)
        head, rel, tail = torch.chunk(triples, chunks=3, dim=1)

        # Check all possible heads and tails
        heads_predictions = model.energy(all_ents, rel, tail)
        tails_predictions = model.energy(head, rel, all_ents)

        pred_ents = torch.cat((tails_predictions, heads_predictions))
        true_ents = torch.cat((tail, head))

        hits_at_10 += utils.hit_at_k(pred_ents, true_ents, k=10)
        mrr += utils.mrr(pred_ents, true_ents)

    hits_at_10 = hits_at_10 / len(loader)
    mrr = mrr / len(loader)

    return mrr, hits_at_10


@ex.automain
def train(lr, margin, p_norm, epochs,
          _run: Run, _log: Logger):
    train_data = GraphDataset('data/fb15k-237/train.txt')
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True,
                              collate_fn=train_data.negative_sampling,
                              num_workers=4)
    valid_data = GraphDataset('data/fb15k-237/valid.txt')
    valid_loader = DataLoader(valid_data, batch_size=64)

    model = TransE(train_data.num_ents, train_data.num_rels, dim=64,
                   margin=margin, p_norm=p_norm).to(device)
    optimizer = Adam(model.parameters(), lr)

    for epoch in range(1, epochs + 1):
        train_loss = 0
        for step, data in enumerate(train_loader):
            pos_triples, neg_triples = data
            loss = model(pos_triples.to(device), neg_triples.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if step % 100 == 0:
                _log.info(f'Epoch {epoch}/{epochs} '
                          f'[{step}/{len(train_loader)}]: {loss.item():.6f}')
                _run.log_scalar('batch_loss', loss.item(), step)

        _run.log_scalar('train_loss', train_loss/len(train_loader))

        mrr, hits = evaluate(model, valid_loader)
        _log.info(f'Validation - mrr: {mrr:.4f}  hits@10: {hits:.4f}')
        _run.log_scalar('valid_mrr', mrr, epoch)
        _run.log_scalar('valid_hits@10', hits, epoch)
