#---------import---------
import argparse
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import time
import torch
from sklearn.metrics import roc_auc_score
#------------------------
from models import *
from utils import *
#----------------------------------
#      set hyperparameter
#----------------------------------
parser = argparse.ArgumentParser('prototype')
parser.add_argument('--dataset', default='tox21')
parser.add_argument('--cwd', default='Desktop/prototype/')
parser.add_argument('--the_endpoint_i',type=int, default=0)
parser.add_argument('--batch_size',type=int,default=4)
parser.add_argument('--exp_setting',default='single_task')
parser.add_argument('--epoch',type=int,default=10)
parser.add_argument('--eval_step', type=int, default=30)
parser.add_argument('--eval_ratio', type=float, default=0.1)
opt= parser.parse_args()
all_config = vars(opt)
contextpred_config={
                 'num_layer': 5,
                 'emb_dim': 300,
                 'JK': 'last',
                 'drop_ratio': 0.5,
                 'gnn_type': 'gin'
             }

if __name__ =='__main__':
    #----------------------------------
    #      set up admin
    #----------------------------------

    checkpoint_dir = set_up_exp_folder(all_config)

    all_config['gnn_config'] = contextpred_config
    all_config['checkpoint_dir'] = checkpoint_dir
    config_file = checkpoint_dir + 'config.json'

    #----------------------------------
    #      set up data
    #----------------------------------
    endpoint_list = np.load(f"data/{all_config['dataset']}-endpoints.npy", allow_pickle=True)
    the_endpoint = endpoint_list[all_config['the_endpoint_i']]
    train_df = pd.read_csv(f"{all_config['cwd']}data/train/{all_config['dataset']}/{the_endpoint}.csv")
    test_df  = pd.read_csv(f"{all_config['cwd']}data/test/{all_config['dataset']}/{the_endpoint}.csv")
    all_config['total_step'] = int(train_df.shape[0]/all_config['batch_size'])*all_config['epoch']
    all_config['eval_at'] = min(200, int(all_config['total_step']*all_config['eval_ratio']))
    print(f"total step: {all_config['total_step']}")
    print(f"eval at: {all_config['eval_at']}")
    all_config['the_endpoint']=the_endpoint
    save_json(all_config, config_file)
    #----------------------------------
    #      set up model & opt
    #----------------------------------
    model = single_task_model(all_config['gnn_config'])
    parameters = list(model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=2e-5, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    loss_fn = torch.nn.CrossEntropyLoss()
    #----------------------------------
    #      training
    #----------------------------------
    loss_train,train_score,test_score=[],[],[]
    best_score = -np.inf

    for step in range(all_config['total_step']):
        # print(f'step ..........{step}')
        start = time.time()
        model.train()
        batch_smiles, batch_target = batch_sample(train_df,all_config,endpoint_list)
        pred_logit = model(batch_smiles)
        loss = loss_fn(pred_logit, batch_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_train.append(loss.item())
        if step% all_config['eval_at']==0:
            model.eval()
            with torch.no_grad():
                eval_logit_train, eval_target_train = eval_fn(train_df,all_config, model,endpoint_list)
                train_score.append(roc_auc_score(eval_target_train,eval_logit_train))
                eval_logit_test, eval_target_test = eval_fn(test_df,all_config, model,endpoint_list)
                test_score.append(roc_auc_score(eval_target_test,eval_logit_test))
                if test_score[-1]> best_score:
                    best_score= test_score[-1]
                    torch.save(model.state_dict(),checkpoint_dir+'model.dat')
                    print(f'------ improved: saved at{checkpoint_dir}')
            print(f"step....... {step}    train: {train_score[-1]}, test: {test_score[-1]}")
            np.save(checkpoint_dir+'train_score.npy',train_score)
            np.save(checkpoint_dir+'test_score.npy',test_score)
            np.save(checkpoint_dir+'loss_train.npy',loss_train)
            print(f'time cost: {time.time() - start}')
    print('~ done :) ~')
    print(checkpoint_dir)

