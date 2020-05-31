import torch
import logging
import models


def get_model(model, dim, rel_model, loss_fn, num_entities, num_relations,
              encoder_name, regularizer):
    if model == 'blp':
        return models.BertEmbeddingsLP(dim, rel_model, loss_fn, num_relations,
                                       encoder_name, regularizer)
    elif model == 'bert-bow':
        return models.BOW(rel_model, loss_fn, num_relations, regularizer,
                          encoder_name=encoder_name)
    elif model == 'bert-dkrl':
        return models.DKRL(dim, rel_model, loss_fn, num_relations, regularizer,
                           encoder_name=encoder_name)
    elif model == 'glove-bow':
        return models.BOW(rel_model, loss_fn, num_relations, regularizer,
                          embeddings='data/glove/glove.6B.300d.pt')
    elif model == 'glove-dkrl':
        return models.DKRL(dim, rel_model, loss_fn, num_relations, regularizer,
                           embeddings='data/glove/glove.6B.300d.pt')
    elif model == 'transductive':
        return models.TransductiveLinkPrediction(dim, rel_model, loss_fn,
                                                 num_entities, num_relations,
                                                 regularizer)
    else:
        raise ValueError(f'Unkown model {model}')


def make_ent2idx(entities, max_ent_id):
    """Given a tensor with entity IDs, return a tensor indexed with
    an entity ID, containing the position of the entity.
    Empty positions are filled with -1.

    Example:
    > make_ent2idx(torch.tensor([4, 5, 0]))
    tensor([ 2, -1, -1, -1,  0,  1])
    """
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


def hit_at_k(predictions, ground_truth_idx, hit_positions):
    """Calculates mean number of hits@k. Higher values are ranked first.

    Args:
        predictions: BxN tensor of prediction values where B is batch size
            and N number of classes.
        ground_truth_idx: Bx1 tensor with index of ground truth class
        hit_positions: list, containing number of top K results to be
            considered as hits.

    Returns: list of float, of the same length as hit_positions, containing
        Hits@K score.
    """
    max_position = max(hit_positions)
    _, indices = predictions.topk(k=max_position)
    hits = []

    for position in hit_positions:
        idx_at_k = indices[:, :position]
        hits_at_k = (idx_at_k == ground_truth_idx).sum(dim=1).float().mean()
        hits.append(hits_at_k.item())

    return hits


def mrr(predictions, ground_truth_idx):
    """Calculates mean reciprocal rank (MRR) for given predictions
    and ground truth values. Higher values are ranked first.

    Args:
        predictions: BxN tensor of prediction values where B is batch size
            and N number of classes.
        ground_truth_idx: Bx1 tensor with index of ground truth class

    Returns: float, Mean reciprocal rank score
    """
    indices = predictions.argsort(descending=True)
    rankings = (indices == ground_truth_idx).nonzero()[:, 1].float() + 1.0
    return rankings.reciprocal()


def split_by_new_position(triples, mrr_values, new_entities):
    """Split MRR results by the position of new entity. Use to break down
    results for a triple where a new entity is at the head and the tail,
    at the head only, or the tail only.
    Since MRR is calculated by corrupting the head first, and then the head,
    the size of mrr_values should be twice the size of triples. The calculated
    MRR is then the average of the two cases.
    Args:
        triples: Bx3 tensor containing (head, tail, rel).
        mrr_values: 2B tensor, with first half containing MRR for corrupted
            triples at the head position, and second half at the tail position.
        new_entities: set, entities to be considered as new.
    Returns:
        mrr_by_position: tensor of 3 elements breaking down MRR by new entities
            at both positions, at head, and tail.
        mrr_pos_counts: tensor of 3 elements containing counts for each case.
    """
    mrr_by_position = torch.zeros(3, device=mrr_values.device)
    mrr_pos_counts = torch.zeros_like(mrr_by_position)
    num_triples = triples.shape[0]

    for i, (h, t, r) in enumerate(triples):
        head, tail = h.item(), t.item()
        mrr_val = (mrr_values[i] + mrr_values[i + num_triples]).item() / 2.0
        if head in new_entities and tail in new_entities:
            mrr_by_position[0] += mrr_val
            mrr_pos_counts[0] += 1.0
        elif head in new_entities:
            mrr_by_position[1] += mrr_val
            mrr_pos_counts[1] += 1.0
        elif tail in new_entities:
            mrr_by_position[2] += mrr_val
            mrr_pos_counts[2] += 1.0

    return mrr_by_position, mrr_pos_counts


def split_by_category(triples, mrr_values, rel_categories):
    mrr_by_category = torch.zeros([2, 4], device=mrr_values.device)
    mrr_cat_count = torch.zeros([1, 4], dtype=torch.float,
                                device=mrr_by_category.device)
    num_triples = triples.shape[0]

    for i, (h, t, r) in enumerate(triples):
        rel_category = rel_categories[r]

        mrr_val_head_pred = mrr_values[i]
        mrr_by_category[0, rel_category] += mrr_val_head_pred

        mrr_val_tail_pred = mrr_values[i + num_triples]
        mrr_by_category[1, rel_category] += mrr_val_tail_pred

        mrr_cat_count[0, rel_category] += 1

    return mrr_by_category, mrr_cat_count


def get_logger():
    """Get a default logger that includes a timestamp."""
    logger = logging.getLogger("")
    logger.handlers = []
    ch = logging.StreamHandler()
    str_fmt = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    formatter = logging.Formatter(str_fmt, datefmt='%H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel('INFO')

    return logger
