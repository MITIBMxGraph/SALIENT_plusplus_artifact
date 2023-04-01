#
# NOTE: 
# This script is included so that the preprocessing we did on MAG240 is clear, it is not meant to be run as-is.
print("This script is not intended to be used directly, without modification.")
exit(1)

import os
import time
import glob
import argparse
import os.path as osp
from tqdm import tqdm

from typing import Optional, List, NamedTuple

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList, Sequential, Linear, BatchNorm1d, ReLU, Dropout
from torch.optim.lr_scheduler import StepLR

from torch_sparse import SparseTensor
from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric.data import NeighborSampler

from ogb.lsc import MAG240MDataset, MAG240MEvaluator

from driver.dataset import FastDataset


'''
 |  MAG240MDataset(root: str = 'dataset')
 |
 |  Methods defined here:
 |
 |  __init__(self, root: str = 'dataset')
 |      Initialize self.  See help(type(self)) for accurate signature.
 |
 |  __repr__(self) -> str
 |      Return repr(self).
 |
 |  download(self)
 |
 |  edge_index(self, id1: str, id2: str, id3: Union[str, NoneType] = None) -> numpy.ndarray
 |
 |  get_idx_split(self, split: Union[str, NoneType] = None) -> Union[Dict[str, numpy.ndarray], numpy.ndarray]
 |
 |  ----------------------------------------------------------------------
 |  Readonly properties defined here:
 |
 |  all_paper_feat
 |
 |  all_paper_label
 |
 |  all_paper_year
 |
 |  num_authors
 |
 |  num_classes
 |
 |  num_institutions
 |
 |  num_paper_features
 |
 |  num_papers
 |
 |  paper_feat
 |
 |  paper_label
 |
 |  paper_year
 |
 |  ----------------------------------------------------------------------
 |  Data descriptors defined here:
 |
 |  __dict__
 |      dictionary for instance variables (if defined)
 |
 |  __weakref__
 |      list of weak references to the object (if defined)
 |
 |  ----------------------------------------------------------------------
 |  Data and other attributes defined here:
 |
 |  __rels__ = {('author', 'institution'): 'affiliated_with', ('author', '...
'''

print ("Before loading dataset")
dataset = MAG240MDataset('/scratch/dataset')
print("Done with getting 'dataset'")
path = f'/scratch/dataset/paper_to_paper_symmetric.pt'
if not osp.exists(path):
    t = time.perf_counter()
    print('Converting adjacency matrix...', end=' ', flush=True)
    edge_index = dataset.edge_index('paper', 'cites', 'paper')
    edge_index = torch.from_numpy(edge_index)
    adj_t = SparseTensor(
        row=edge_index[0], col=edge_index[1],
        sparse_sizes=(dataset.num_papers, dataset.num_papers),
        is_sorted=True)
    torch.save(adj_t.to_symmetric(), path)
    print(f'Done! [{time.perf_counter() - t:.2f}s]')
print("Hello world")

t = time.perf_counter()
print('Reading dataset...', end=' ', flush=True)
dataset = MAG240MDataset('/scratch/dataset')

metainfo = dict()
metainfo['num_authors'] = dataset.num_authors
metainfo['num_classes'] = dataset.num_classes
metainfo['num_institutions'] = dataset.num_institutions
metainfo['num_paper_features'] = dataset.num_paper_features
metainfo['num_papers'] = dataset.num_papers

split_idx = dict()
split_idx['train'] = torch.from_numpy(dataset.get_idx_split('train'))
split_idx['valid'] = torch.from_numpy(dataset.get_idx_split('valid'))
split_idx['test'] = torch.from_numpy(dataset.get_idx_split('test-dev'))

print("test idx split below")
print(dataset.get_idx_split('test'))

x = torch.from_numpy(dataset.all_paper_feat)
y = torch.from_numpy(dataset.all_paper_label)
path = f'{dataset.dir}/paper_to_paper_symmetric.pt'
adj_t = torch.load(path)

dataset = FastDataset.import_mag240(adj_t, x, y, split_idx, metainfo)
dataset.save('/data/dataset')
