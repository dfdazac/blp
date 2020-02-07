from torch.utils.data import DataLoader
from torch.optim import Adam

from data import GraphDataset, make_data_iterator
from graph import TransE
import utils

ex = utils.create_experiment()


@ex.automain
def train(_run, _log):
    dataset = GraphDataset('data/wikidata5m/triples.txt')
    loader = DataLoader(dataset, collate_fn=dataset.negative_sampling,
                        batch_size=64, shuffle=True, num_workers=4)
    iterator = make_data_iterator(loader)

    model = TransE(dataset.num_ents, dataset.num_rels, dim=32)
    optimizer = Adam(model.parameters())
    train_iters = 100

    for i, data in enumerate(iterator):
        loss = model(data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            _log.info(f'[{i}/{train_iters}]: {loss.item():.6f}')
