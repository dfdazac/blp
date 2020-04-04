import torch


def make_ent2idx(entities):
    """Given a tensor with entity IDs, return a tensor indexed with
    an entity ID, containing the position of the entity.
    Empty positions are filled with -1.

    Example:
    > make_ent2idx(torch.tensor([4, 5, 0]))
    tensor([ 2, -1, -1, -1,  0,  1])
    """
    max_ent_id = entities.max()
    idx = torch.arange(entities.shape[0])
    ent2idx = torch.empty(max_ent_id + 1, dtype=torch.long).fill_(-1)
    ent2idx.scatter_(0, entities, idx)
    return ent2idx


def get_triple_filters(triples, graph, num_ents, ent2idx):
    """Given a set of triples, filter candidate entities that are valid
    substitutes of an entity in the triple at a given position (head or tail).
    For a particular triple, this allows to compute rankings for an entity of
    interest, against other entities in the graph that would actually be wrong
    substitutes.
    Results are returned as a mask array with a value of 1.0 for filtered
    entities, and 0.0 otherwise.

    Args:
        triples: Bx3 tensor of type torch.long, where B is the batch size,
            and each row contains a triple of the form (head, tail, rel)
        graph: nx.MultiDiGraph containing all edges used to filter candidates
        num_ents: int, number of candidate entities
        ent2idx: tensor, contains at index ent_id the index of the column for
            that entity in the output mask array
    """
    num_triples = triples.shape[0]
    heads_filter = torch.zeros((num_triples, num_ents), dtype=torch.bool)
    tails_filter = torch.zeros_like(heads_filter)

    triples = triples.tolist()
    for i, (head, tail, rel) in enumerate(triples):
        head_edges = graph.out_edges(head, data='weight')
        for (h, t, r) in head_edges:
            if r == rel and t != tail:
                ent_idx = ent2idx[t]
                if ent_idx != -1:
                    tails_filter[i, ent_idx] = True

        tail_edges = graph.in_edges(tail, data='weight')
        for (h, t, r) in tail_edges:
            if r == rel and h != head:
                ent_idx = ent2idx[h]
                if ent_idx != -1:
                    heads_filter[i, ent_idx] = True

    return heads_filter, tails_filter


def hit_at_k(predictions, ground_truth_idx, k=10):
    """Calculates mean number of hits@k.

    Args:
        predictions: BxN tensor of prediction values where B is batch size
            and N number of classes.
        ground_truth_idx: Bx1 tensor with index of ground truth class
        k: number of top K results to be considered as hits

    Returns: float, Hits@K score
    """
    _, indices = predictions.topk(k=k, largest=False)
    return (indices == ground_truth_idx).sum(dim=1).float().mean().item()


def mrr(predictions, ground_truth_idx):
    """Calculates mean reciprocal rank (MRR) for given predictions
    and ground truth values.

    Args:
        predictions: BxN tensor of prediction values where B is batch size
            and N number of classes.
        ground_truth_idx: Bx1 tensor with index of ground truth class

    Returns: float, Mean reciprocal rank score
    """
    indices = predictions.argsort()
    rankings = (indices == ground_truth_idx).nonzero()[:, 1].float() + 1.0
    return torch.mean(rankings.reciprocal()).item()
