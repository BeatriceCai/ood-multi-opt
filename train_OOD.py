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
#----------------------------------
#      set hyperparameter
#----------------------------------
parser = argparse.ArgumentParser('prototype')
parser.add_argument('--dataset', default='tox21')
parser.add_argument('--cwd', default='Desktop/prototype/')
parser.add_argument('--use_cuda',default=1)
parser.add_argument('--batch_size',type=int,default=4)
parser.add_argument('--exp_setting',default='multi_task_ood')
parser.add_argument('--epoch',type=int,default=10)
parser.add_argument('--eval_step', type=int, default=30)
parser.add_argument('--eval_ratio', type=float, default=0.1)
#----------------
parser.add_argument('--OOD_ratio', type=float, default=0.5)
parser.add_argument('--OOD_step', type=int, default=30)
parser.add_argument('--outer_step', type=int, default=30)
# parser.add_argument('--OOD_loop', type=int, default=10)
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
    train_df = pd.read_csv(f"{all_config['cwd']}data/train/{all_config['dataset']}-all.csv")
    test_df  = pd.read_csv(f"{all_config['cwd']}data/test/{all_config['dataset']}-all.csv")
    STEP = int(train_df.shape[0]/all_config['batch_size']/(all_config['OOD_step'] + all_config['outer_step']))
    all_config['total_loop'] = STEP * all_config['epoch']
    all_config['eval_at'] = min(100, int(all_config['total_loop']*all_config['eval_ratio']))
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
    loss_fn = torch.nn.CrossEntropyLoss()
    #----------------------------------
    #      set up training
    #----------------------------------
    loss_train,train_score,test_score,loss_ood=[],[],[],[]
    best_score = -np.inf
    pick_OOD = int( all_config['batch_size']*2*all_config['OOD_ratio'])
    for loop in range(all_config['total_loop']):
        # print(f'loop ..........{loop}')
        start = time.time()
        model.train()
        batch_smiles, batch_target = batch_sample_multi(train_df, all_config)
        # if step % all_config['OOD_loop'] !=0:
        for outer_step  in range(all_config['outer_step']):
            # print(f'    outer step {outer_step}')
            pred_logit = model(batch_smiles)
            loss = loss_fn(pred_logit, batch_target)
            loss_train.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        # else:
        for ood_step in range(all_config['OOD_step']):
            # print(f'    ood step {ood_step}')
            x = model.chem_embed(batch_smiles)
            mul = model.cal_mul(x)
            mode_set = mul.mean(0, keepdim=True)[0]
            ood_enhance_loss = 0
            for i in range(len(endpoint_list)):
                # ............
                H = mul[:, i, :]
                mode = mode_set[i]
                dist_to_mode = torch.cdist(H, mode.unsqueeze(0))
                id_OOD = torch.argsort(dist_to_mode.squeeze(1),
                                       descending=True)[:pick_OOD].detach().cpu().numpy().tolist()
                h_ood = H[id_OOD]
                # ............
                mo = mode_set.clone()
                mo[i] = torch.tensor([9] * model.w1.shape[1])
                dist_to_other_mode = torch.cdist(h_ood, mo)
                pick_task = dist_to_other_mode.argsort(descending=False)[:, 0].detach().cpu().numpy().tolist()
                pick_x = x[:, i, :, :][id_OOD].unsqueeze(1)
                pick_w = model.w1[pick_task].unsqueeze(1)
                pick_b = model.b1[:, pick_task].squeeze(0).unsqueeze(1)
                # ............
                h_ood_pick = ((pick_x * pick_w).sum(-1) + pick_b)
                ood_enhance_loss_i = torch.dist(h_ood, h_ood_pick.squeeze())
                ood_enhance_loss += ood_enhance_loss_i / pick_OOD
            loss = ood_enhance_loss/len(endpoint_list)
            loss_ood.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
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
            print(f'time cost: {time.time() - start}')
    print('~ done :) ~')
    print(checkpoint_dir)
