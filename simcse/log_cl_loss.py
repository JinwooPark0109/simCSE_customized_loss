import torch
import sys
from sklearn.mixture import GaussianMixture

def log_original_contrastive_loss(cls, tup):
    '''
    contrastive loss from original code
    '''
    z1, z2, z3 = tup
    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    loss_fct = torch.nn.CrossEntropyLoss()

    neg_sim=torch.triu(cos_sim, diagonal=1)[:, 1:] + torch.tril(cos_sim, diagonal=-1)[:, :-1]
    cls.log_sim(cos_sim.diagonal(), neg_sim, neg_sim) # generated nis dummy

    loss=loss_fct(cos_sim, labels)

    return cos_sim, loss


def log_very_hard_similar_detached_rewrite_loss(cls, tup):
    #print("very hard negative")
    '''
    z1+N(0,z1.std(0)
    '''
    z1, z2, _ = tup  # b,d

    hard_list=[]
    zero=torch.zeros(z1.size(1)).to(cls.device) # d
    z_std=z1.std(0) # d
    for sample in z1:
        hard_list.append(sample.detach().clone()+torch.normal(mean=zero, std=z_std))
    hard=torch.stack(hard_list).to(cls.device)

    pos_cos_sim = cls.sim(z1, z2).unsqueeze(1)
    neg_cos_sim = cls.sim(z1.unsqueeze(1), hard.unsqueeze(0)) # remove diagnoal
    neg_cos_sim = torch.triu(neg_cos_sim, diagonal=1)[:, 1:]+torch.tril(neg_cos_sim, diagonal=-1)[:,:-1] # bx(b-1)

    cos_sim = torch.concat([pos_cos_sim, neg_cos_sim], dim=1) # (b, b)
    labels = torch.zeros(cos_sim.size(0)).long().to(cls.device)   # 0th is pos at all batch items.

    loss_fct = torch.nn.CrossEntropyLoss()

    cls.log_sim(pos_cos_sim, neg_cos_sim, neg_cos_sim) # original neg is dummy

    loss=loss_fct(cos_sim, labels)
    return cos_sim, loss


def log_very_hard_similar_detached_rewrite_0_01_loss(cls, tup):
    #print("very hard negative")
    z1, z2, _ = tup  # b,d

    hard_list=[]
    zero=torch.zeros(z1.size(1)).to(cls.device) # d
    z_std=z1.std(0) # d
    for sample in z1:
        hard_list.append(sample.detach().clone()+torch.normal(mean=zero, std=0.01))
    hard=torch.stack(hard_list).to(cls.device)

    pos_cos_sim = cls.sim(z1, z2).unsqueeze(1)
    neg_cos_sim = cls.sim(z1.unsqueeze(1), hard.unsqueeze(0)) # remove diagnoal
    neg_cos_sim = torch.triu(neg_cos_sim, diagonal=1)[:, 1:]+torch.tril(neg_cos_sim, diagonal=-1)[:,:-1] # bx(b-1)

    cos_sim = torch.concat([pos_cos_sim, neg_cos_sim], dim=1) # (b, b)
    labels = torch.zeros(cos_sim.size(0)).long().to(cls.device)   # 0th is pos at all batch items.

    loss_fct = torch.nn.CrossEntropyLoss()

    cls.log_sim(pos_cos_sim, neg_cos_sim, neg_cos_sim) # original neg is dummy

    loss=loss_fct(cos_sim, labels)
    return cos_sim, loss


def log_k_easy_origin_negative_loss(cls, tup):
    #print("easy easy origin negative")
    z1, z2, _ = tup  # b,d

    noise_list=[]
    z_mean=z1.mean(0)
    z_std=z1.std(0)
    for _ in range(z1.size(0)*cls.k):
        noise_list.append(torch.normal(mean=z_mean, std=z_std))
    noise=torch.stack(noise_list)

    origin_cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    easy_cos_sim = cls.sim(z1.unsqueeze(1), noise.unsqueeze(0)) # (b,1,d)x(1,2b,d)=(bx2b)

    cos_sim = torch.concat([origin_cos_sim, easy_cos_sim], dim=1) # (b, 3b)

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)   # b

    loss_fct = torch.nn.CrossEntropyLoss()

    neg_sim = torch.triu(origin_cos_sim, diagonal=1)[:, 1:] + torch.tril(origin_cos_sim, diagonal=-1)[:, :-1]
    cls.log_sim(origin_cos_sim.diagonal(), neg_sim, easy_cos_sim)

    loss = loss_fct(cos_sim, labels)  #
    return cos_sim, loss


def log_easy_cov_origin_loss(cls, tup):
    #print("easy origin negative")
    z1, z2, _ = tup  # b,d

    noise_list=[]
    z_mean=z1.mean(0)
    z_std=z1.std(0)
    cov_mat=torch.cov(z1.T)
    for _ in range(z1.size(0)):
        noise_list.append(torch.matmul(torch.normal(mean=z_mean, std=z_std), cov_mat))
    noise=torch.stack(noise_list)

    origin_cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    easy_cos_sim = cls.sim(z1.unsqueeze(1), noise.unsqueeze(0))

    cos_sim = torch.concat([origin_cos_sim, easy_cos_sim], dim=1) # (b, 2b)

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)   # b

    loss_fct = torch.nn.CrossEntropyLoss()

    neg_sim = torch.triu(origin_cos_sim, diagonal=1)[:, 1:] + torch.tril(origin_cos_sim, diagonal=-1)[:, :-1]
    cls.log_sim(origin_cos_sim.diagonal(), neg_sim, easy_cos_sim)

    loss = loss_fct(cos_sim, labels)  #
    return cos_sim, loss


def log_k_gmm_buffer_loss(cls, tup):
    z1, z2, _ = tup  # b,d

    cls.replace_gmm(z1)
    noise = cls.gmm_sampling(z1.size(0) * cls.k)

    origin_cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    easy_cos_sim = cls.sim(z1.unsqueeze(1), noise.unsqueeze(0)) # (b,1,d)x(1,kb,d)=(b,kb)

    cos_sim = torch.concat([origin_cos_sim, easy_cos_sim], dim=1) # (b, kb+1)
    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)   # b

    loss_fct = torch.nn.CrossEntropyLoss()

    neg_sim = torch.triu(origin_cos_sim, diagonal=1)[:, 1:] + torch.tril(origin_cos_sim, diagonal=-1)[:, :-1]
    cls.log_sim(origin_cos_sim.diagonal(), neg_sim, easy_cos_sim)

    loss = loss_fct(cos_sim, labels)
    return cos_sim, loss


def log_k_vae_loss(cls, tup):
    z1, z2, _ = tup  # b,d

    cls.update_vae(z1.clone().detach())
    noise = cls.vae_sampling(z1.size(0) * cls.k)

    origin_cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    neg_cos_sim = cls.sim(z1.unsqueeze(1), noise.unsqueeze(0)) # (b,1,d)x(1,kb,d)=(b,kb)

    cos_sim = torch.concat([origin_cos_sim, neg_cos_sim], dim=1) # (b, kb+1)
    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)   # b

    loss_fct = torch.nn.CrossEntropyLoss()

    neg_sim = torch.triu(origin_cos_sim, diagonal=1)[:, 1:] + torch.tril(origin_cos_sim, diagonal=-1)[:, :-1]
    cls.log_sim(origin_cos_sim.diagonal(), neg_sim, neg_cos_sim)

    loss = loss_fct(cos_sim, labels)
    return cos_sim, loss



def get_cl_loss(loss_name=None):
    if not loss_name:
        return None
    try:
        modules=sys.modules[__name__]
        loss= getattr(modules, loss_name)
    except AttributeError as e:
        #print(e)
        print("------------current implemented loss list------------------")
        loss_list=[m for m in dir(modules) if callable(getattr(modules,m))]
        print("------------------------------------")
        print(loss_list)
        print(f"{loss_name} not implemented yet\n")
        sys.exit()
    return loss