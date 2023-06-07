import torch
import torch.nn.functional as F
import numpy as np
#--------------------------------code from Enua----------------------------------------

## Positive loss
def measure_positive_loss(C):
    assert(len(C.shape) == 2)
    positive_loss = -C[torch.eye(C.shape[0], C.shape[1])==1].view(C.shape[0], -1).mean()
    return float(positive_loss)

## Negative loss
def measure_negative_loss(C):
    assert(len(C.shape) == 2)
    negative_loss = torch.logsumexp(C[torch.eye(C.shape[0], C.shape[1])==0].view(C.shape[0], -1), dim=1).mean() ## denominator without positive pair
    # negative_loss = torch.logsumexp(C.view(C.shape[0], -1), dim=1).mean() ## denominator including positive pair
    return float(negative_loss)

## Align loss
def measure_align_loss(x, y, alpha=2):
    norm_x = F.normalize(x)
    norm_y = F.normalize(y)
    return float((norm_x - norm_y).norm(p=2, dim=1).pow(alpha).mean())

## Uniform loss
def measure_uniform_loss(x, t=2):
    norm_x = F.normalize(x)
    return float(torch.pdist(norm_x, p=2).pow(2).mul(-t).exp().mean().log())

## Isotropy
def get_partition_function(reps):
    with torch.no_grad():
        embed = F.normalize(reps)
        # rep_size = reps.shape[1]
        # values_list = []
        # for ii in range(reps.shape[0]//reps.shape[1]):
        #     each_embed = embed[rep_size*ii:rep_size*(ii+1)]
        #     eig_vec = torch.symeig(torch.matmul(each_embed.T,each_embed),eigenvectors=True)[1]
        #     values_list.append(torch.exp(torch.matmul(each_embed, eig_vec.T)).sum(0))
        # print(values_list)
        # values = torch.cat(values_list)
        eig_vec = torch.symeig(torch.matmul(embed.T,embed),eigenvectors=True)[1]
        values = torch.exp(torch.matmul(embed, eig_vec.T)).sum(0)
        return values

def measure_isotropy(reps):
    partitions = get_partition_function(reps)
    return float((partitions.min() / partitions.max()).item())

## Rank
def measure_approx_rank(X, power=0.99):
    W = X.clone().detach()
    assert(len(W.shape) == 2)
    Z = torch.nn.functional.normalize(W)
    rho = torch.matmul(Z.T,Z) / Z.shape[0]
    eig_val = torch.symeig(rho,eigenvectors=True)[0][-Z.shape[0]:]

    return float((eig_val.sort(descending=True)[0].cumsum(0) < power).sum().item())

## Average cosine similarity
def measure_avg_cos(X):
    W = X.clone().detach()
    assert(len(W.shape) == 2)
    ## W: embedding matrix, shape=[n_data, dim]
    n_data = W.shape[0]
    W_norm = torch.norm(W, dim=1)
    W2 = W / W_norm.unsqueeze(1)
    WW = torch.matmul(W2, W2.T) ## WW.shape[n_data, n_data]
    triu_numel = n_data * (n_data-1) / 2
    cos = torch.sum(torch.triu(WW, diagonal=1)) / triu_numel
    return float(cos)

## Norm (default: Frobenius norm)
def measure_avg_norm(X):
    W = X.clone().detach()
    assert(len(W.shape) == 2)
    ## W: embedding matrix, shape=[n_data, dim]
    W_norm = torch.norm(W, dim=1)
    assert(W_norm.shape[0] == W.shape[0])
    avg_norm = torch.sum(W_norm) / W.shape[0]
    return float(avg_norm)

def measure_disentanglement(X):
    W = X.clone().detach()
    assert(len(W.shape) == 2)
    ## W: embedding matrix, shape=[n_data, dim]
    W_T = W.T
    dim = W_T.shape[0]
    W_T_norm = torch.norm(W_T, dim=1)
    W_T /= W_T_norm.unsqueeze(1)
    WW = torch.matmul(W_T, W) ## WW.shape[dim, dim]
    # print("WW.shape", WW.shape)
    triu_numel = dim * (dim-1) / 2
    cos = torch.sum(torch.triu(WW, diagonal=1)) / triu_numel
    return float(cos)

## Compute VNE
def get_vne(H):
    assert(len(H.shape) == 2)
    Z = torch.nn.functional.normalize(H, dim=1)
    sing_val = torch.svd(Z / np.sqrt(Z.shape[0]))[1]
    eig_val = sing_val ** 2
    return - (eig_val * torch.log(eig_val)).nansum().item()


#--------------------------------implementation from JW------------------------------
'''
def align_loss(x, y, alpha=2):
    return (torch.nn.functional.normalize(x, dim=1) - torch.nn.functional.normalize(y, dim=1)).norm(p=2, dim=1).pow(alpha).mean().item()

def uniform_loss(x, t=2):
    return torch.pdist(torch.nn.functional.normalize(x, dim=1), p=2).pow(2).mul(-t).exp().mean().log().item()

def vne_loss(H):
    # https://github.com/xlpczv/SentRep/blob/main/SimCSE/simcse/models.py#L29
    #
    assert (len(H.shape) == 2)
    Z = torch.nn.functional.normalize(H, dim=1)
    sing_val = torch.svd(Z / np.sqrt(Z.shape[0]))[1]
    eig_val = sing_val ** 2
    return - (eig_val * torch.log(eig_val)).nansum().item()

## Rank
def measure_approx_rank(X, power=0.99):
    W = X.clone().detach()
    assert(len(W.shape) == 2)
    Z = torch.nn.functional.normalize(W)
    rho = torch.matmul(Z.T,Z) / Z.shape[0]
    eig_val = torch.symeig(rho,eigenvectors=True)[0][-Z.shape[0]:]

    return (eig_val.sort(descending=True)[0].cumsum(0) < power).sum().item()

def get_avg_cos_sim(X):
    z = X.clone().detach()
    z = torch.nn.functional.normalize(z)
    mat = torch.matmul(z, z.transpose(0,1))
    el_num=mat.shape[0]*(mat.shape[0]-1)/2
    return (mat.triu(1).sum()/el_num).item()

def get_avg_cos_sim_for_gen(x1, x2):
    z1 = x1.clone().detach()
    z2 = x2.clone().detach()
    z1 = torch.nn.functional.normalize(z1)
    z2 = torch.nn.functional.normalize(z2)
    mat = torch.matmul(z1, z2.transpose(0,1))
    el_num=mat.shape[0]*(mat.shape[0]+1)/2
    return (mat.triu().sum()/el_num).item()
'''
