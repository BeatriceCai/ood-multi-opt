import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import time
import torch
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
#------------------------
from utils import *
from models import *
from ood_enhance_loss import *

def trainer_alternative_1_loop(model, train_df,all_config,optimizer,scheduler):
    loss_train, train_score, loss_ood,loss_pgi= [], [], [],[]
    loss_fn = torch.nn.CrossEntropyLoss()
    for outer_step in range(all_config['outer_step']):
        batch_smiles, batch_target = batch_sample_multi(train_df, all_config)
        # print(f'    outer step {outer_step}')
        pred_logit = model(batch_smiles)
        loss = loss_fn(pred_logit, batch_target)
        loss_train.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    for ood_step in range(all_config['OOD_step']):
        batch_smiles, batch_target = batch_sample_multi(train_df, all_config)
        x = model.chem_embed(batch_smiles,OOD_enhence_step=True)
        mul = model.cal_mul(x)
        mode_set = mul.mean(0, keepdim=True)[0]
        i = np.random.choice(all_config['endpoint_num'], 1)[0]

        loss_ood_i, loss_pgi_i = get_ood_enhance_loss(i, mul, mode_set, all_config, model, x)



        loss_ood.append(loss_ood_i.item())
        optimizer.zero_grad()
        loss_ood_i.backward()
        optimizer.step()
        scheduler.step()


    return loss_train, train_score,loss_ood,loss_pgi

# def trainer_alternative_1_loop(model, train_df,all_config,optimizer,scheduler):
#     loss_train, train_score, loss_ood = [], [], []
#     loss_fn = torch.nn.CrossEntropyLoss()
#     for outer_step in range(all_config['outer_step']):
#         batch_smiles, batch_target = batch_sample_multi(train_df, all_config)
#         # print(f'    outer step {outer_step}')
#         pred_logit = model(batch_smiles)
#         loss = loss_fn(pred_logit, batch_target)
#         loss_train.append(loss.item())
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         scheduler.step()
#
#     for ood_step in range(all_config['OOD_step']):
#         batch_smiles, batch_target = batch_sample_multi(train_df, all_config)
#         x = model.chem_embed(batch_smiles,OOD_enhence_step=True)
#         mul = model.cal_mul(x)
#         mode_set = mul.mean(0, keepdim=True)[0]
#
#         i = np.random.choice(all_config['endpoint_num'], 1)[0]
#         loss = get_ood_enhance_loss(i, mul, mode_set, all_config['pick_OOD'], model, x)
#
#         loss_ood.append(loss.item())
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         scheduler.step()
#     return loss_train, train_score,loss_ood