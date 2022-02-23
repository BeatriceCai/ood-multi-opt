import os
from datetime import datetime
import json
import pickle
import warnings
warnings.filterwarnings('ignore')
import torch

def set_up_exp_folder(all_config):
    now = datetime.now()
    timestamp = now.strftime("%d-%m-%Y-%H-%M-%S")
    print('timestamp: ',timestamp)
    save_folder = all_config['cwd'] + 'exp_log_'+all_config['exp_setting']+'/'
    dataset_folder = save_folder +all_config['dataset']+'/'

    if os.path.exists(save_folder) == False:
            os.mkdir(save_folder)
    if os.path.exists(dataset_folder) == False:
            os.mkdir(dataset_folder)
    checkpoint_dir = '{}exp{}/'.format(dataset_folder, timestamp)
    if os.path.exists(checkpoint_dir ) == False:
            os.mkdir(checkpoint_dir )
    return checkpoint_dir

##### JSON modules #####
def save_json(data,filename):
    with open(filename, 'w') as fp:
        json.dump(data, fp, sort_keys=True, indent=4)

def load_json(filename):
    with open(filename, 'r') as fp:
        data = json.load(fp)
    return data

##### pickle modules #####
def save_dict_pickle(data,filename):
    with open(filename,'wb') as handle:
        pickle.dump(data,handle, pickle.HIGHEST_PROTOCOL)

def load_pkl(path):
    with open(path, 'rb') as f:
        dict = pickle.load(f)
    return  dict

##### optimization modules #####
def batch_sample(data, all_config,endpoint_list):
    batch_data = data.sample(all_config['batch_size'])
    batch_smiles = batch_data['smiles'].values.tolist()
    batch_target = torch.tensor(batch_data[all_config['the_endpoint']].values.tolist()).long()
    return batch_smiles,batch_target

def eval_fn(train_df,all_config,model,endpoint_list):
    eval_logit = []
    eval_target = []
    for t in range(all_config['eval_step']):
        if all_config['exp_setting']=='single_task':
            batch_smiles, batch_target = batch_sample(train_df, all_config,endpoint_list)
        elif all_config['exp_setting']=='multi_task_baseline' or  all_config['exp_setting']=='multi_task_ood':
            head_config=all_config['head_config']
            batch = train_df.groupby(['endpoint', 'y']).sample(n=all_config['batch_size'])
            batch_smiles = batch['similes'].values
            batch_y = batch['y'].values
            batch_target = torch.tensor(batch_y.reshape((head_config['channel_size'],
                                                         head_config['batch_size'] * head_config[
                                                             'output_size'])).swapaxes(0, 1)).long()
            if all_config['use_cuda'] == 1 and torch.cuda.is_available():
                batch_target=batch_target.to('cuda')
        elif all_config['exp_setting']=='multi_task_all_shared':
            head_config = all_config['head_config']
            batch = train_df.groupby(['endpoint', 'y']).sample(n=all_config['batch_size'])
            batch_smiles = batch['similes'].values
            batch_y = batch['y'].values
            batch_target = torch.tensor(batch_y).reshape((head_config['batch_size']*head_config['output_size'],
                                            head_config['channel_size']))
        pred_logit = model(batch_smiles)
        if all_config['exp_setting'] == 'multi_task_all_shared':
            pred_logit = pred_logit.reshape((head_config['batch_size'] * head_config['output_size'],
                                             head_config['channel_size'],2))
            pred = list(pred_logit.detach().cpu().numpy()[:, :,1])
        else:
            pred = list(pred_logit.detach().cpu().numpy()[:, 1])
        target = list(batch_target.detach().cpu().numpy())
        eval_logit.extend(pred)
        eval_target.extend(target)
    return eval_logit,eval_target

def batch_sample_multi(train_df,all_config):
    head_config = all_config['head_config']
    batch = train_df.groupby(['endpoint', 'y']).sample(n=all_config['batch_size'])
    batch_smiles = batch['similes'].values
    batch_y = batch['y'].values
    batch_target = torch.tensor(batch_y.reshape((head_config['channel_size'],
                                                 head_config['batch_size'] * head_config['output_size'])).swapaxes(0,
                                                                                                                   1)).long()
    if all_config['use_cuda'] == 1 and torch.cuda.is_available():
        batch_target = batch_target.to('cuda')
    return batch_smiles,batch_target