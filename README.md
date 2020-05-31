# Inductive entity representations from text via link prediction

<img src="fig.png" width="400" />



This repository contains the code used for the experiments in the paper "Inductive entity representations from text via link prediction".

## Usage

### 1. Install the requirements

Creating a new environment (e.g. with `conda`) is recommended. Use `requirements.txt` to install the dependencies:

```sh
conda create -n blp python=3.7
conda activate blp
pip install -r requirements.txt
```

### 2. Download the data

Download the required compressed datasets into the `data` folder:

| Download link                                                | Size (compressed) |
| ------------------------------------------------------------ | ----------------- |
| [UMLS](https://surfdrive.surf.nl/files/index.php/s/NvuKQuBetmOUe1b/download) (small graph for tests) | 121 KB            |
| [WN18RR](https://surfdrive.surf.nl/files/index.php/s/N1c8VRH0I6jTJuN/download) | 6.6 MB            |
| [FB15k-237](https://surfdrive.surf.nl/files/index.php/s/rGqLTDXRFLPJYg7/download) | 21 MB             |
| [Wikidata5M](https://surfdrive.surf.nl/files/index.php/s/TEE96zweMxsoGmR/download) | 1.4 GB            |
| [GloVe embeddings](https://surfdrive.surf.nl/files/index.php/s/zAHCIBc6PAb3NXi/download) | 423 MB            |
| [DBpedia-Entity](https://surfdrive.surf.nl/files/index.php/s/BOD7SoDTchVO9ed/download) | 1.3 GB            |

Then use `tar` to extract the files, e.g.

```sh
tar -xzvf WN18RR.tar.gz
```

### 2. Reproduce the experiments

**Link prediction**

To check that all dependencies are correctly installed, run a quick test on a small graph (this should take less than 1 minute on GPU):

```sh
./scripts/test-umls.sh
```

The following table is a adapted from our paper. The last column contains the name of the script that reproduces the experiment for the corresponding row.

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-0lax{text-align:left;vertical-align:top}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-0lax"></th>
    <th class="tg-0lax" colspan="2">WN18RR</th>
    <th class="tg-0lax" colspan="2">FB15k-237</th>
    <th class="tg-0lax" colspan="2">Wikidata5M</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0lax">Model</td>
    <td class="tg-0lax">MRR</td>
    <td class="tg-0lax">Script</td>
    <td class="tg-0lax">MRR</td>
    <td class="tg-0lax">Script</td>
    <td class="tg-0lax">MRR</td>
    <td class="tg-0lax">Script</td>
  </tr>
  <tr>
    <td class="tg-0lax">GlovE-BOW</td>
    <td class="tg-0lax">0.170</td>
    <td class="tg-0lax">glove-bow-wn18rr.sh</td>
    <td class="tg-0lax">0.172</td>
    <td class="tg-0lax"><span style="font-weight:400;font-style:normal">glove-bow-fb15k237.sh</span></td>
    <td class="tg-0lax"><span style="font-weight:400;font-style:normal">0.343</span></td>
    <td class="tg-0lax"><span style="font-weight:400;font-style:normal">glove-bow-wikidata5m.sh</span></td>
  </tr>
  <tr>
    <td class="tg-0lax">BE-BOW</td>
    <td class="tg-0lax">0.180</td>
    <td class="tg-0lax">bert-bow-wn18rr.sh</td>
    <td class="tg-0lax">0.173</td>
    <td class="tg-0lax"><span style="font-weight:400;font-style:normal">bert-bow-fb15k237.sh</span></td>
    <td class="tg-0lax">0.362</td>
    <td class="tg-0lax"><span style="font-weight:400;font-style:normal">bert-bow-wikidata5m.sh</span></td>
  </tr>
  <tr>
    <td class="tg-0lax">GloVe-DKRL</td>
    <td class="tg-0lax">0.115</td>
    <td class="tg-0lax">glove-dkrl-wn18rr.sh</td>
    <td class="tg-0lax">0.112</td>
    <td class="tg-0lax"><span style="font-weight:400;font-style:normal">glove-dkrl-fb15k237.sh</span></td>
    <td class="tg-0lax">0.282</td>
    <td class="tg-0lax"><span style="font-weight:400;font-style:normal">glove-dkrl-wikidata5m.sh</span></td>
  </tr>
  <tr>
    <td class="tg-0lax">BE-DKRL</td>
    <td class="tg-0lax">0.139</td>
    <td class="tg-0lax">bert-dkrl-wn18rr.sh</td>
    <td class="tg-0lax">0.144</td>
    <td class="tg-0lax"><span style="font-weight:400;font-style:normal">bert-dkrl-fb15k237.sh</span></td>
    <td class="tg-0lax">0.322</td>
    <td class="tg-0lax"><span style="font-weight:400;font-style:normal">bert-dkrl-wikidata5m.sh</span></td>
  </tr>
  <tr>
    <td class="tg-0lax">BLP-TransE</td>
    <td class="tg-0lax">0.285</td>
    <td class="tg-0lax">blp-transe-wn18rr.sh</td>
    <td class="tg-0lax">0.195</td>
    <td class="tg-0lax"><span style="font-weight:400;font-style:normal">blp-transe-fb15k237.sh</span></td>
    <td class="tg-0lax">0.478</td>
    <td class="tg-0lax"><span style="font-weight:400;font-style:normal">blp-transe-wikidata5m.sh</span></td>
  </tr>
  <tr>
    <td class="tg-0lax">BLP-DistMult</td>
    <td class="tg-0lax">0.248</td>
    <td class="tg-0lax"><span style="font-weight:400;font-style:normal">blp</span>-distmult-wn18rr.sh</td>
    <td class="tg-0lax">0.146</td>
    <td class="tg-0lax"><span style="font-weight:400;font-style:normal">blp-distmult-fb15k237.sh</span></td>
    <td class="tg-0lax">0.472</td>
    <td class="tg-0lax"><span style="font-weight:400;font-style:normal">blp-distmult-wikidata5m.sh</span></td>
  </tr>
  <tr>
    <td class="tg-0lax">BLP-ComplEx</td>
    <td class="tg-0lax">0.261</td>
    <td class="tg-0lax"><span style="font-weight:400;font-style:normal">blp</span>-complex-wn18rr.sh</td>
    <td class="tg-0lax">0.148</td>
    <td class="tg-0lax"><span style="font-weight:400;font-style:normal">blp-complex-fb15k237.sh</span></td>
    <td class="tg-0lax">0.489</td>
    <td class="tg-0lax"><span style="font-weight:400;font-style:normal">blp-complex-wikidata5m.sh</span></td>
  </tr>
  <tr>
    <td class="tg-0lax">BLP-SimplE</td>
    <td class="tg-0lax">0.239</td>
    <td class="tg-0lax"><span style="font-weight:400;font-style:normal">blp</span>-simple-wn18rr.sh</td>
    <td class="tg-0lax">0.144</td>
    <td class="tg-0lax"><span style="font-weight:400;font-style:normal">blp-simple-fb15k237.sh</span></td>
    <td class="tg-0lax">0.493</td>
    <td class="tg-0lax"><span style="font-weight:400;font-style:normal">blp-simple-wikidata5m.sh</span></td>
  </tr>
</tbody>
</table>



**Entity classification**

After training for link prediction, a tensor of embeddings for all entities is computed and saved in a file with name `ent_emb-[ID].pt` where `[ID]` is the id of the experiment in the database (we use [Sacred](https://sacred.readthedocs.io/en/stable/index.html) to manage experiments). Another file called `ents-[ID].pt` contains entity identifiers for every row in the tensor of embeddings.

To ease reproducibility, we provide these tensors, which are required in the entity classification task. Click on the ID, download the file into the `output` folder, and decompress it. An experiment can be reproduced using the following command:

```sh
python train.py node_classification with checkpoint=ID dataset=DATASET
```

where `DATASET` is either `WN18RR` or `FB15k-237`. For example:

```sh
python train.py node_classification with checkpoint=199 dataset=WN18RR
```


<table>
<thead>
  <tr>
    <th></th>
    <th colspan="2">WN18RR</th>
    <th colspan="2">FB15k-237</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Model</td>
    <td>Acc.</td>
    <td>ID</td>
    <td>Acc. Bal.</td>
    <td>ID</td>
  </tr>
  <tr>
    <td>GloVe-BOW</td>
    <td>55.3</td>
    <td><a href="https://surfdrive.surf.nl/files/index.php/s/dAac2HSzTVOZXXF/download" target="_blank" rel="noopener noreferrer">219</a></td>
    <td>34.4</td>
    <td><a href="https://surfdrive.surf.nl/files/index.php/s/CmM3S3zFBeB2rIx/download" target="_blank" rel="noopener noreferrer">293</a></td>
  </tr>
  <tr>
    <td>BE-BOW</td>
    <td>60.7</td>
    <td><a href="https://surfdrive.surf.nl/files/index.php/s/3YQzViDa2xfskSt/download" target="_blank" rel="noopener noreferrer">218</a></td>
    <td>28.3</td>
    <td><a href="https://surfdrive.surf.nl/files/index.php/s/TGWmNq7MXlFPiQ0/download" target="_blank" rel="noopener noreferrer">296</a></td>
  </tr>
  <tr>
    <td>GloVe-DKRL</td>
    <td>55.5</td>
    <td><a href="https://surfdrive.surf.nl/files/index.php/s/tTs6x39SY4DxXV2/download" target="_blank" rel="noopener noreferrer">206</a></td>
    <td>26.6</td>
    <td><a href="https://surfdrive.surf.nl/files/index.php/s/bRrQiLl2ZVxfhDO/download" target="_blank" rel="noopener noreferrer">295</a></td>
  </tr>
  <tr>
    <td>BE-DKRL</td>
    <td>48.8</td>
    <td><a href="https://surfdrive.surf.nl/files/index.php/s/wwfU6lxnywxAeO2/download" target="_blank" rel="noopener noreferrer">207</a></td>
    <td>30.9</td>
    <td><a href="https://surfdrive.surf.nl/files/index.php/s/nRqY6dWS0ermX4X/download" target="_blank" rel="noopener noreferrer">294</a></td>
  </tr>
  <tr>
    <td>BLP-TransE</td>
    <td>81.5</td>
    <td><a href="https://surfdrive.surf.nl/files/index.php/s/SGYvIT2iuPqhett/download" target="_blank" rel="noopener noreferrer">199</a></td>
    <td>42.5</td>
    <td><a href="https://surfdrive.surf.nl/files/index.php/s/VpmltGpMHzUdbsw/download" target="_blank" rel="noopener noreferrer">297</a></td>
  </tr>
  <tr>
    <td>BLP-DistMult</td>
    <td>78.5</td>
    <td><a href="https://surfdrive.surf.nl/files/index.php/s/evFOoBufK6BQ69V/download" target="_blank" rel="noopener noreferrer">200</a></td>
    <td>41.0</td>
    <td><a href="https://surfdrive.surf.nl/files/index.php/s/lXrvINpxHTkmdup/download" target="_blank" rel="noopener noreferrer">298</a></td>
  </tr>
  <tr>
    <td>BLP-ComplEx</td>
    <td>78.1</td>
    <td><a href="https://surfdrive.surf.nl/files/index.php/s/MNMrI6dddeybgum/download" target="_blank" rel="noopener noreferrer">201</a></td>
    <td>38.1</td>
    <td><a href="https://surfdrive.surf.nl/files/index.php/s/MkDGfJXy1ANTBeQ/download" target="_blank" rel="noopener noreferrer">300</a></td>
  </tr>
  <tr>
    <td>BLP-SimplE</td>
    <td>83.0</td>
    <td><a href="https://surfdrive.surf.nl/files/index.php/s/n5sJIY4QZhzo3bU/download" target="_blank" rel="noopener noreferrer">202</a></td>
    <td>45.7</td>
    <td><a href="https://surfdrive.surf.nl/files/index.php/s/B7nuyaJw9F61dwP/download" target="_blank" rel="noopener noreferrer">299</a></td>
  </tr>
</tbody>
</table>


**Information retrieval**

This task runs with a pre-trained model saved from the link prediction task. For example, if the model trained is `blp` with `transe` and it was saved as `model.pt`, then run the following command to run the information retrieval task:

```sh
python retrieval.py with model=blp rel_model=transe \
checkpoint='output/model.pt'
```
