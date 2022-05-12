import torch
from torch import nn
from model_Yang import GNN
from torch_geometric.utils import to_dense_batch
from featurizer import featurizer
import math
import warnings
warnings.filterwarnings('ignore')

class single_task_model(nn.Module):
    def __init__(self, all_config):
        super(single_task_model, self).__init__()
        contextpred_config = all_config['gnn_config']
        self.all_config = all_config
        self.ligandEmbedding = GNN(num_layer=contextpred_config['num_layer'],
                       emb_dim=contextpred_config['emb_dim'],
                       JK=contextpred_config['JK'],
                       drop_ratio=contextpred_config['drop_ratio'],
                       gnn_type=contextpred_config['gnn_type'])
        self.task_head= nn.Sequential(nn.Linear(contextpred_config['emb_dim'],512),
                                  nn.ReLU(),
                                  nn.Linear(512,2))

    def forward(self,batch_smiles):
        chem_graphs = featurizer(batch_smiles)
        if self.all_config['use_cuda'] == 1 and torch.cuda.is_available():
            chem_graphs = chem_graphs.to('cuda')
        node_representation = self.ligandEmbedding(chem_graphs.x, chem_graphs.edge_index,
                                              chem_graphs.edge_attr)

        batch_chem_graphs_repr_masked, mask_graph = to_dense_batch(node_representation,
                                                                   chem_graphs.batch)
        batch_chem_graphs_repr_pooled = batch_chem_graphs_repr_masked.sum(axis=1)

        logits = self.task_head(batch_chem_graphs_repr_pooled)
        return logits

    def infer(self,batch_smiles):
        chem_graphs = featurizer(batch_smiles)
        if self.all_config['use_cuda'] == 1 and torch.cuda.is_available():
            chem_graphs = chem_graphs.to('cuda')
        node_representation = self.ligandEmbedding(chem_graphs.x, chem_graphs.edge_index,
                                              chem_graphs.edge_attr)

        batch_chem_graphs_repr_masked, mask_graph = to_dense_batch(node_representation,
                                                                   chem_graphs.batch)
        batch_chem_graphs_repr_pooled = batch_chem_graphs_repr_masked.sum(axis=1)

        logits = self.task_head(batch_chem_graphs_repr_pooled)
        # embed = self.task_head[0](batch_chem_graphs_repr_pooled)
        return logits, batch_chem_graphs_repr_pooled

def init_w_b(channel_size,output_size, input_size):
    w= torch.nn.Parameter(torch.zeros(channel_size,output_size,input_size) )
    b = torch.nn.Parameter(torch.zeros(1,channel_size, output_size))
    torch.nn.init.kaiming_uniform_(w,a=math.sqrt(3))
    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(w)
    bound = 1/math.sqrt(fan_in)
    torch.nn.init.uniform_(b,-bound,bound)
    return w, b


def multi_task_head(head_config):
    channel_size = head_config['channel_size']
    hidden_size = head_config['hidden_size']
    input_size = head_config['input_size']
    output_size= head_config['output_size']

    w1, b1 = init_w_b(channel_size, hidden_size, input_size)
    w2, b2 = init_w_b(channel_size, output_size, hidden_size)
    relu = nn.ReLU()

    return w1, b1, w2, b2, relu

class multi_task_baseline(nn.Module):
    def __init__(self, all_config):
        super(multi_task_baseline, self).__init__()
        # -------- hyper param ---------
        contextpred_config = all_config['gnn_config']
        self.all_config=all_config
        head_config = all_config['head_config']
        self.channel_size = head_config['channel_size']
        self.hidden_size = head_config['hidden_size']
        self.input_size = head_config['input_size']
        self.batch_size = head_config['batch_size']
        self.output_size = head_config['output_size']

        # -------- model---------

        self.ligandEmbedding = GNN(num_layer=contextpred_config['num_layer'],
                       emb_dim=contextpred_config['emb_dim'],
                       JK=contextpred_config['JK'],
                       drop_ratio=contextpred_config['drop_ratio'],
                       gnn_type=contextpred_config['gnn_type'])
        self.w1,self.b1, self.w2, self.b2, self.relu = multi_task_head(all_config['head_config'])

#
    def forward(self,batch_smiles):
        x = self.chem_embed(batch_smiles)
        mul = (x * self.w1).sum(-1) + self.b1
        mul = self.relu(mul)

        mul_next = mul.unsqueeze(2)
        logit = (mul_next*self.w2).sum(-1) + self.b2
        logit = logit.swapaxes(1,2)
        return logit

    def chem_embed(self,batch_smiles,OOD_enhence_step=False):
        # ------ shared graph embedding ------
        chem_graphs = featurizer(batch_smiles)
        if self.all_config['use_cuda'] == 1 and torch.cuda.is_available():
            chem_graphs = chem_graphs.to('cuda')
        if self.all_config['seperate_opt']==1 and OOD_enhence_step ==True:
            # print('seperate-opt')
            with torch.no_grad():
                node_representation = self.ligandEmbedding(chem_graphs.x,chem_graphs.edge_index,chem_graphs.edge_attr)

        else:
            node_representation = self.ligandEmbedding(chem_graphs.x, chem_graphs.edge_index, chem_graphs.edge_attr)
        batch_chem_graphs_repr_masked, mask_graph = to_dense_batch(node_representation,
                                                                   chem_graphs.batch)
        batch_chem_graphs_repr_pooled = batch_chem_graphs_repr_masked.sum(axis=1)
        # ----- multi head individual adaptation -----
        x = batch_chem_graphs_repr_pooled.reshape((self.channel_size,
                                                   self.batch_size * self.output_size,
                                                   self.input_size)).swapaxes(0, 1).unsqueeze(2)
        return x
    def cal_mul(self,x):
        mul = (x * self.w1).sum(-1) + self.b1
        return mul



class multi_task_baseline_share_till_logit(nn.Module):
    def __init__(self, all_config):
        super(multi_task_baseline_share_till_logit, self).__init__()
        # -------- hyper param ---------
        contextpred_config = all_config['gnn_config']
        self.all_config=all_config
        head_config = all_config['head_config']
        self.channel_size = head_config['channel_size']
        self.hidden_size = head_config['hidden_size']
        self.input_size = head_config['input_size']
        self.batch_size = head_config['batch_size']
        self.output_size = head_config['output_size']

        # -------- model---------

        self.ligandEmbedding = GNN(num_layer=contextpred_config['num_layer'],
                       emb_dim=contextpred_config['emb_dim'],
                       JK=contextpred_config['JK'],
                       drop_ratio=contextpred_config['drop_ratio'],
                       gnn_type=contextpred_config['gnn_type'])
        self.linear = nn.Sequential(nn.Linear(contextpred_config['emb_dim'], 512), nn.ReLU())
        self.w,self.b = init_w_b(head_config['channel_size'], head_config['output_size'],512)

#
    def forward(self,batch_smiles):
        batch_chem_graphs_repr_pooled = self.chem_embed(batch_smiles)
        last_shared = self.linear(batch_chem_graphs_repr_pooled)
        x = last_shared.reshape((self.channel_size, self.batch_size*self.output_size, 512)).swapaxes(0, 1).unsqueeze(2)
        logit = (x * self.w).sum(-1) + self.b
        pred_logit = logit.swapaxes(1, 2)
        return pred_logit

    def chem_embed(self,batch_smiles):
        # ------ shared graph embedding ------
        chem_graphs = featurizer(batch_smiles)
        if self.all_config['use_cuda'] == 1 and torch.cuda.is_available():
            chem_graphs =chem_graphs.to('cuda')
        node_representation = self.ligandEmbedding(chem_graphs.x,chem_graphs.edge_index,chem_graphs.edge_attr)
        batch_chem_graphs_repr_masked, mask_graph = to_dense_batch(node_representation,
                                                                   chem_graphs.batch)
        batch_chem_graphs_repr_pooled = batch_chem_graphs_repr_masked.sum(axis=1)
        #----- multi head individual adaptation -----
        # x = batch_chem_graphs_repr_pooled.reshape((self.channel_size,
        #                                            self.batch_size*self.output_size,
        #                                            self.input_size)).swapaxes(0, 1).unsqueeze(2)
        return batch_chem_graphs_repr_pooled
    # def cal_mul(self,x):
    #     mul = (x * self.w1).sum(-1) + self.b1
    #     return mul




