import rdkit
from rdkit.Chem import MolFromSmiles,RDKFingerprint
from rdkit import DataStructs
from rdkit.DataStructs import FingerprintSimilarity
from sklearn.cluster import SpectralClustering
from scipy.sparse import coo_matrix
import pandas as pd
import os
import numpy as np
from utils import save_dict_pickle,load_pkl
import argparse

parser = argparse.ArgumentParser('prototype')
opt= parser.parse_args()
parser.add_argument('--dataset', default='tox21')

path = 'data/'
# dataset ='tier12'
dataset = 'opt.dataset
if dataset =='tier12':
    print('ah')
    combined = path +'tier12/'
    endpoint_list =np.load( path+'tier12-endpoints.npy',allow_pickle=True)
else:
    combined = path+'moleculenet-by-endpoint/'
    endpoint_list = np.load(path +'tox21-endpoints.npy',allow_pickle=True)

for i in range()