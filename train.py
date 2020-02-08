import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from sacred.run import Run
from logging import Logger

from data import GraphDataset, make_data_iterator
from graph import TransE
import utils

ex = utils.create_experiment()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@ex.config
def config():
    lr = 1e-2
    margin = 1
    p_norm = 1


@ex.automain
def train(lr, margin, p_norm,
          _run: Run, _log: Logger):
    dataset = GraphDataset('data/wikidata5m/triples.txt')
    loader = DataLoader(dataset, collate_fn=dataset.negative_sampling,
                        batch_size=64, shuffle=True, num_workers=4)
    iterator = make_data_iterator(loader)

    model = TransE(dataset.num_ents, dataset.num_rels, dim=64,
                   margin=margin, p_norm=p_norm).to(device)
    optimizer = Adam(model.parameters(), lr)
    train_iters = 25000

    for step in range(train_iters):
        data = next(iterator)
        pos_triples, neg_triples = data
        loss = model(pos_triples.to(device), neg_triples.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            _log.info(f'[{step}/{train_iters}]: {loss.item():.6f}')
            _run.log_scalar('loss', loss.item(), step)
