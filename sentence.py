import logging
import senteval
import torch
from transformers import BertTokenizer
from pprint import pprint

import models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def prepare(params, samples):
    encoder_name = 'bert-base-cased'
    encoder = models.BertEmbeddingsLP(dim=128, rel_model='transe',
                                      loss_fn='margin', num_relations=237,
                                      encoder_name=encoder_name,
                                      regularizer=0.0)
    encoder.load_state_dict(torch.load(f'output/bed-70.pt',
                                       map_location='cpu'))
    for param in encoder.parameters():
        param.requires_grad = False

    params.encoder = encoder.to(device)
    params.tokenizer = BertTokenizer.from_pretrained(encoder_name)


def batcher(params, batch):
    batch = [sent if sent != [] else ['.'] for sent in batch]
    tokenizer = params.tokenizer
    tokenized_data = tokenizer.batch_encode_plus(batch, pad_to_max_length=True,
                                                 return_token_type_ids=False,
                                                 return_tensors='pt')
    tokens = tokenized_data['input_ids'].to(device)
    masks = tokenized_data['attention_mask'].to(device)

    embeddings = params.encoder.encode(tokens, masks).cpu()
    return embeddings


params = {'task_path': 'data/senteval', 'usepytorch': True, 'kfold': 10}
params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                        'tenacity': 5, 'epoch_size': 4}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    se = senteval.engine.SE(params, batcher, prepare)
    transfer_tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']
    results = se.eval(transfer_tasks)
    pprint(results)
