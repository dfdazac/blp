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


def hit_at_k(predictions, ground_truth_idx, k=10):
    """Calculates mean number of hits@k.

    Args:
        predictions: BxN tensor of prediction values where B is batch size
            and N number of classes. Must be sorted in IDs order
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
            and N number of classes. Must be sorted in IDs order
        ground_truth_idx: Bx1 tensor with index of ground truth class

    Returns: float, Mean reciprocal rank score
    """
    indices = predictions.argsort()
    rankings = (indices == ground_truth_idx).nonzero()[:, 1].float() + 1.0
    return torch.mean(rankings.reciprocal()).item()
