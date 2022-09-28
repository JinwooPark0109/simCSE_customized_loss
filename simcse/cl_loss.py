import torch
import json

def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()

def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

#todo: method to class
def align_uniform_loss(cls, tup, hparams):
    '''
    tup is tuple of (x,y,z)
    x (b,d) : one side of positive pair
    y (b,d) : other side of positive pair
    z (b,d) : currently not used. for hard negative
    '''
    # print("custom loss used")
    z1, z2,_= tup
    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))

    get_statistics(cos_sim)
    return torch.nn.CrossEntropyLoss(), cos_sim, hparams['lambda_align'] * align_loss(z1,z2) + hparams['lambda_uniform'] * uniform_loss(z1)

def original_contrastive_loss(cls, tup):
    '''
    contrastive loss from original code
    '''
    z1, z2, z3 = tup

    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    # Hard negative
    if z3:
        z1_z3_cos = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0))
        cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    loss_fct = torch.nn.CrossEntropyLoss()

    # Calculate loss with hard negatives
    if z3:
        # Note that weights are actually logits of weights
        z3_weight = cls.model_args.hard_negative_weight
        weights = torch.tensor(
            [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (
                        z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
        ).to(cls.device)
        cos_sim = cos_sim + weights

    loss=loss_fct(cos_sim, labels)

    print("original loss")
    #print("input:", z1.shape, z2.shape, tup)
    #print("cos sim:", cos_sim.shape, cos_sim)
    print("loss:", loss.item())

    return cos_sim, loss_fct(cos_sim, labels)

def cpc(cls, tup, hparams):
    z1, z2,_= tup
    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))

    batch_size = z1.shape[0]

    mask_positive = torch.eye(batch_size)==1
    mask_negative = torch.ones([batch_size,batch_size])==1
    
    similarity_matrix = torch.nn.CosineSimilarity(dim=-1)(z1.unsqueeze(1), z2.unsqueeze(0)) / hparams['tau']

    loss = (- hparams['lambda_pos'] * similarity_matrix[mask_positive].view(similarity_matrix.shape[0], -1)\
            + hparams['lambda_neg'] * torch.logsumexp(similarity_matrix[mask_negative].view(similarity_matrix.shape[0], -1), dim=1).unsqueeze(1)).mean()

    return torch.nn.CrossEntropyLoss(), cos_sim, loss


def cpc_3views(cls, tup, hparams):
    z1, z2, z3= tup
    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))

    batch_size = z1.shape[0]

    mask_x, mask_y = torch.broadcast_tensors(torch.cat([torch.arange(batch_size)]).view(1,-1), torch.cat([torch.arange(batch_size)]).view(-1,1))
    mask_positive = torch.eye(batch_size)==1
    mask_negative = torch.ones([batch_size,batch_size])==1

    loss = 0

    similarity_matrix = torch.nn.CosineSimilarity(dim=-1)(z1.unsqueeze(1), z2.unsqueeze(0)) / hparams['tau']
    loss += (- hparams['lambda_pos'] * similarity_matrix[mask_positive].view(similarity_matrix.shape[0], -1)\
             + hparams['lambda_neg'] * torch.logsumexp(similarity_matrix[mask_negative].view(similarity_matrix.shape[0], -1), dim=1).unsqueeze(1)).mean()

    similarity_matrix = torch.nn.CosineSimilarity(dim=-1)(z1.unsqueeze(1), z3.unsqueeze(0)) / hparams['tau']
    loss += (- hparams['lambda_pos'] * similarity_matrix[mask_positive].view(similarity_matrix.shape[0], -1)\
             + hparams['lambda_neg'] * torch.logsumexp(similarity_matrix[mask_negative].view(similarity_matrix.shape[0], -1), dim=1).unsqueeze(1)).mean()

    similarity_matrix = torch.nn.CosineSimilarity(dim=-1)(z2.unsqueeze(1), z3.unsqueeze(0)) / hparams['tau']
    loss += (- hparams['lambda_pos'] * similarity_matrix[mask_positive].view(similarity_matrix.shape[0], -1)\
             + hparams['lambda_neg'] * torch.logsumexp(similarity_matrix[mask_negative].view(similarity_matrix.shape[0], -1), dim=1).unsqueeze(1)).mean()

    return torch.nn.CrossEntropyLoss(), cos_sim, loss

class custom_loss_function(object): 
    def __init__(self, args=None):
        self.args = args
        # import IPython; IPython.embed()
        self.hparams =  json.loads(args.hparams)
        print('#################################')
        print('#################################')
        print('Initializing custom loss function')
        print(self.hparams)
        print('#################################')
        print('#################################')

    def __call__(self, cls, tup):
        if self.args.mycl_name == 'original_contrastive':
            return original_contrastive_loss(cls, tup)
        elif self.args.mycl_name == 'uniform':
            return align_uniform_loss(cls, tup, self.hparams)
        elif self.args.mycl_name == 'cpc':
            return cpc(cls, tup, self.hparams)
        elif self.args.mycl_name == 'cpc_3views':
            return cpc_3views(cls, tup, self.hparams)

def get_cl_loss(loss_name=None, args=None):
    if not loss_name:
        return None
    if loss_name=="align_uniform_loss":
        return align_uniform_loss
    elif loss_name=="original_contrastive_loss":
        return original_contrastive_loss
    elif loss_name=="custom_loss":
        return custom_loss_function(args)
    else:
        raise ValueError("not implemented yet")