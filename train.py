from torch.utils.data import DataLoader
from torch.optim import Adam
from data import GraphDataset
from graph import TransE

dataset = GraphDataset('data/wikidata5m/triples.txt')
loader = DataLoader(dataset, collate_fn=dataset.negative_sampling,
                    batch_size=64, num_workers=4)
model = TransE(dataset.num_ents, dataset.num_rels, dim=32)
optimizer = Adam(model.parameters())

for i, data in enumerate(loader):
    loss = model(data)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 10 == 0:
        print(f'{i}: {loss.item():.6f}')
