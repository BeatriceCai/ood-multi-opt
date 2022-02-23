import os
import numpy as np
import pandas as pd
from rdkit.Chem import MolFromSmiles,RDKFingerprint
from rdkit import DataStructs,Chem
from rdkit.ML.Cluster import Butina
from rdkit.Chem import AllChem
from typing import List
from utils import save_dict_pickle,load_pkl


import argparse

parser = argparse.ArgumentParser('prototype')
parser.add_argument('--dataset', default='tox21')
opt= parser.parse_args()


cutoff=0.6
frac_train= 0.8
frac_test = 0.2

cwd = 'Desktop/prototype/'

path = cwd + 'data/'
dataset = opt.dataset
if dataset =='tier12':
    print(dataset)
    combined = path +'tier12/'
    endpoint_list =np.load( path+'tier12-endpoints.npy',allow_pickle=True)
else:
    print(dataset)
    combined = path+'moleculenet-by-endpoint/'
    endpoint_list = np.load(path +'tox21-old-endpoints.npy',allow_pickle=True)
stats ={}
for i in range(len(endpoint_list)):
    print(i)
    data = pd.read_csv(combined + endpoint_list[i] + '.csv')
    # create fingerprint dict

    dict_file = cwd+ 'data/fingerprint_dict/' + dataset + '/' + endpoint_list[i] + '.pkl'
    if os.path.exists(dict_file):
        fp_dict = load_pkl(dict_file)
        print('load existing fp_dict')
    else:
        print('...... create new fp_dict')
        fp_dict = {}
        for s in data['smiles'].values.tolist():
            mol = MolFromSmiles(s)
            fps = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)
            fp_dict[s] = fps
        save_dict_pickle(fp_dict, dict_file)

    fps = list(fp_dict.values())
    print(f'number of fps: {len(fps)}')
    keys = list(fp_dict.keys())

    dists = []
    nfps = len(fps)
    for f in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[f], fps[:f])
        dists.extend([1 - x for x in sims])

    scaffold_sets = Butina.ClusterData(dists, nfps, cutoff, isDistData=True)
    scaffold_sets = sorted(scaffold_sets, key=lambda x: -len(x))

    train_cutoff = frac_train * len(data)

    train_inds: List[int] = []
    test_inds: List[int] = []

    for scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) > train_cutoff:
            test_inds += scaffold_set
        else:
            train_inds += scaffold_set

    train_set = np.array(keys)[train_inds]
    test_set = np.array(keys)[test_inds]
    train_df = data[data['smiles'].isin(train_set)]
    test_df = data[data['smiles'].isin(test_set)]

    if dataset == 'tier12':
        train_file =cwd+ 'data/train/' + dataset + '/' + endpoint_list[i] + '.csv'
        test_file = cwd +'data/test/' + dataset + '/' + endpoint_list[i] + '.csv'
    else:
        train_file = cwd+'data/train/' + dataset + '/' + endpoint_list[i][:-6] + '.csv'
        test_file = cwd+'data/test/' + dataset + '/' + endpoint_list[i][:-6] + '.csv'

    print(f'train size:{ train_df.shape[0]}, test size: {test_df.shape[0]}')
    stats[endpoint_list[i]]=(train_df.shape[0], test_df.shape[0])
    train_df.to_csv(train_file, index=None)
    test_df.to_csv(test_file, index=None)
    save_dict_pickle(stats,f'{cwd}data/stats_{dataset}.pkl')