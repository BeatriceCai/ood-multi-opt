
import torch
import torch.nn.functional as F
import torch.nn as nn

def cal_logit(x,w1,b1,w2,b2,relu):
    mul =( x*w1).sum(-1) +b1
    mul_relu = relu(mul)
    mul_next = mul_relu.unsqueeze(2)
    logit =( mul_next*w2).sum(-1)+b2
    logit = logit.swapaxes(1,2)
    return mul,logit

def get_ood_enhance_loss(i,mul,mode_set,all_config,model,x):
    # pick_OOD = all_config['pick_OOD']
    # ............
    H = mul[:, i, :]
    mode = mode_set[i]
    dist_to_mode = torch.cdist(H, mode.unsqueeze(0))
    id_sorted = torch.argsort(dist_to_mode.squeeze(1),
                              descending=True).detach().cpu().numpy().tolist()
    id_OOD = id_sorted[: all_config['pick_OOD']]
    # id_OOD = torch.argsort(dist_to_mode.squeeze(1),
    #                        descending=True)[:pick_OOD].detach().cpu().numpy().tolist()
    h_ood = H[id_OOD]
    # ............
    mo = mode_set.clone()
    mo[i] = torch.tensor([9] * model.w1.shape[1])
    dist_to_other_mode = torch.cdist(h_ood, mo)
    pick_task = dist_to_other_mode.argsort(descending=False)[:, 0].detach().cpu().numpy().tolist()
    # ............
    pick_x = x[:, i, :, :][id_OOD].unsqueeze(0)

    pick_w1 = model.w1[pick_task]
    pick_b1 = model.b1[:, pick_task].squeeze(0)
    pick_w2 = model.w2[pick_task]
    pick_b2 = model.b2[:, pick_task].squeeze(0)
    relu = model.relu
    # # ............
    h_ood_pick, _ = cal_logit(pick_x, pick_w1, pick_b1, pick_w2, pick_b2, relu)
    # h_ood_pick = ((pick_x * pick_w).sum(-1) + pick_b)
    ood_enhance_loss_i = torch.dist(h_ood, h_ood_pick.squeeze())
    loss_ood = ood_enhance_loss_i /  all_config['pick_OOD']

    if model.all_config['pgi']==1:
        # kl_loss = nn.KLDivLoss(reduction='batchmean', log_target=True)
        w1_in = model.w1[[i]]
        b1_in = model.b1[:, [i]].squeeze(0)
        w2_in = model.w2[[i]]
        b2_in = model.b2[:, [i]].squeeze(0)
        id_IN = id_sorted[ all_config['pick_OOD']:]
        x_IN = x[:, i, :, :][id_IN].unsqueeze(1)
        _, logit_in = cal_logit(x_IN, w1_in, b1_in, w2_in, b2_in, relu)
        logit_in_repeat = torch.repeat_interleave(logit_in,  all_config['pick_OOD'], dim=-1)
        _, logit_in_pick = cal_logit(x_IN, pick_w1, pick_b1, pick_w2, pick_b2, relu)
        p = torch.vstack(torch.unbind(logit_in_repeat, dim=-1))
        q = torch.vstack(torch.unbind(logit_in_pick, dim=-1))

        p_softmax_log = F.log_softmax(p, dim=1)
        q_softmax_log = F.log_softmax(q, dim=1)

        loss_pgi = all_config['kl_fn'](p_softmax_log, q_softmax_log)
    else:
        loss_pgi = None

    return loss_ood, loss_pgi

# def get_ood_enhance_loss(i,mul,mode_set,pick_OOD,model,x):
#
#     # ............
#     H = mul[:, i, :]
#     mode = mode_set[i]
#     dist_to_mode = torch.cdist(H, mode.unsqueeze(0))
#     id_OOD = torch.argsort(dist_to_mode.squeeze(1),
#                            descending=True)[:pick_OOD].detach().cpu().numpy().tolist()
#     h_ood = H[id_OOD]
#     # ............
#     mo = mode_set.clone()
#     mo[i] = torch.tensor([9] * model.w1.shape[1])
#     dist_to_other_mode = torch.cdist(h_ood, mo)
#     pick_task = dist_to_other_mode.argsort(descending=False)[:, 0].detach().cpu().numpy().tolist()
#     pick_x = x[:, i, :, :][id_OOD].unsqueeze(1)
#     pick_w = model.w1[pick_task].unsqueeze(1)
#     pick_b = model.b1[:, pick_task].squeeze(0).unsqueeze(1)
#     # ............
#     h_ood_pick = ((pick_x * pick_w).sum(-1) + pick_b)
#     ood_enhance_loss_i = torch.dist(h_ood, h_ood_pick.squeeze())
#
#     return ood_enhance_loss_i / pick_OOD