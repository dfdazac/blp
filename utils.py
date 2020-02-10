import os
from sacred import Experiment
from sacred.observers import MongoObserver
import torch


def create_experiment():
    ex = Experiment()
    # Set up database logs
    uri = os.environ.get('MLAB_URI')
    database = os.environ.get('MLAB_DB')
    if all([uri, database]):
        ex.observers.append(MongoObserver(uri, database))

    return ex


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
