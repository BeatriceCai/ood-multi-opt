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
from utils import *
from models import *
from trainer import *
from ood_enhance_loss import *
#----------------------------------
#      set hyperparameter
#----------------------------------
parser = argparse.ArgumentParser('prototype')

parser.add_argument('--cwd', default='Desktop/prototype/')
parser.add_argument('--cwd_data', default='/Users/tiancai/Desktop/', help='/home/tian/')
parser.add_argument('--use_cuda',default=1)
parser.add_argument('--AWS',default=0)
#----------------
parser.add_argument('--batch_size',type=int,default=4)
parser.add_argument('--epoch',type=int,default=10)
parser.add_argument('--eval_step', type=int, default=30)
parser.add_argument('--eval_ratio', type=float, default=0.1)
#----------------
parser.add_argument('--dataset', default='tox21')
parser.add_argument('--exp_setting',default='multi_task_ood')
parser.add_argument('--seperate_opt',default=1)
parser.add_argument('--pgi',default=1)
#----------------
parser.add_argument('--OOD_ratio', type=float, default=0.5)
parser.add_argument('--OOD_step', type=int, default=30)
parser.add_argument('--outer_step', type=int, default=30)
parser.add_argument('--min_eval_at', type=int, default=300)
#----------------
opt= parser.parse_args()
all_config = vars(opt)
contextpred_config={
                 'num_layer': 5,
                 'emb_dim': 300,
                 'JK': 'last',
                 'drop_ratio': 0.5,
                 'gnn_type': 'gin'
             }
if opt.use_cuda==1 and torch.cuda.is_available():
    torch.cuda.manual_seed(7)
    print('using GPU')
torch.set_num_threads(8)
if __name__ =='__main__':
    print(f"seperate_opt: {all_config['seperate_opt']} ,pgi enabled:{all_config['pgi']}")
    #----------------------------------
    #      set up admin
    #----------------------------------
    if all_config['AWS']==1:
        checkpoint_dir=''
    else:
        checkpoint_dir = set_up_exp_folder(all_config)
    all_config['gnn_config'] = contextpred_config
    all_config['checkpoint_dir'] = checkpoint_dir
    config_file = checkpoint_dir + 'config.json'
    #----------------------------------
    #      set up data
    #----------------------------------
    endpoint_list = np.load(f"{all_config['cwd_data']}prototype/data/{all_config['dataset']}-endpoints.npy", allow_pickle=True)
    all_config['endpoint_num'] = len(endpoint_list)
    train_df = pd.read_csv(f"{all_config['cwd_data']}prototype/data/train/{all_config['dataset']}-all.csv")
    test_df  = pd.read_csv(f"{all_config['cwd_data']}prototype/data/test/{all_config['dataset']}-all.csv")
    STEP = int(train_df.shape[0]/all_config['batch_size']/(all_config['OOD_step'] + all_config['outer_step']))
    all_config['total_loop'] = STEP * all_config['epoch']
    all_config['eval_at'] = min(all_config['min_eval_at'], int(all_config['total_loop']*all_config['eval_ratio']))
    print(f"total loop: {all_config['total_loop']}")
    print(f"eval at: {all_config['eval_at']}")
    head_config = {'input_size': all_config['gnn_config']['emb_dim'],
                   'hidden_size': 256,
                   'output_size': 2,
                   'channel_size': len(endpoint_list),
                   'batch_size': all_config['batch_size']}
    all_config['head_config']= head_config
    save_json(all_config, config_file)
    #----------------------------------
    #      set up model
    #----------------------------------
    model = multi_task_baseline(all_config)
    if opt.use_cuda == 1 and torch.cuda.is_available():
        model=model.to('cuda')
    parameters = list(model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=2e-5, weight_decay=0.001)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    #----------------------------------
    #      set up training
    #----------------------------------
    loss_train,train_score,test_score,loss_ood, loss_pgi=[],[],[],[],[]
    best_score = -np.inf
    pick_OOD = int( all_config['batch_size']*2*all_config['OOD_ratio'])
    all_config['pick_OOD'] = pick_OOD
    all_config['kl_fn'] = nn.KLDivLoss(reduction='batchmean', log_target=True)
    start = time.time()
    for loop in range(all_config['total_loop']):
        # print(f'loop ..........{loop}')

        model.train()
        loss_train_i , train_score_i,loss_ood_i,loss_pgi_i = trainer_alternative_1_loop(model, train_df, all_config,optimizer,scheduler)
        loss_train.extend(loss_train_i)
        train_score.extend(train_score_i)
        loss_ood.extend(loss_ood_i)
        loss_pgi.extend(loss_pgi_i)

        if loop % all_config['eval_at']==0:
            model.eval()
            with torch.no_grad():
                eval_logit_train, eval_target_train = eval_fn(train_df,all_config, model,endpoint_list)
                train_score.append(roc_auc_score(eval_target_train,eval_logit_train,average=None))
                eval_logit_test, eval_target_test = eval_fn(test_df,all_config, model,endpoint_list)
                test_score.append(roc_auc_score(eval_target_test,eval_logit_test,average=None))
                # print(test_score[-1].shape)
                if np.mean(test_score[-1])> best_score:
                    best_score= np.mean(test_score[-1])
                    torch.save(model.state_dict(),checkpoint_dir+'model.dat')
                    print(f'------ improved: saved at{checkpoint_dir}')
            print(f"loop....... {loop}   ")
            print(f"train: {train_score[-1]}")
            print(f"test: {test_score[-1]}")
            np.save(checkpoint_dir+'train_score.npy',train_score)
            np.save(checkpoint_dir+'test_score.npy',test_score)
            np.save(checkpoint_dir+'loss_train.npy',loss_train)
            np.save(checkpoint_dir+'loss_ood.npy',loss_ood)
            np.save(checkpoint_dir+'loss_pgi.npy',loss_pgi)
            print(f'time cost: {time.time() - start}')
            start = time.time()
    print('~ done :) ~')
    print(checkpoint_dir)
