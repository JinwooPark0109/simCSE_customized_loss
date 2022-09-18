import torch

def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()

def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

#todo: method to class
def align_uniform_loss(cls, tup):
    '''
    tup is tuple of (x,y,z)
    x (b,d) : one side of positive pair
    y (b,d) : other side of positive pair
    z (b,d) : currently not used. for hard negative
    '''
    print("custom loss used")
    z1, z2,_= tup
    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))

    return torch.nn.CrossEntropyLoss(), cos_sim, align_loss(z1,z2)+uniform_loss(z1)

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

    return torch.nn.CrossEntropyLoss(), cos_sim, loss_fct(cos_sim, labels)

def get_cl_loss(loss_name=None):
    if not loss_name:
        return None
    if loss_name=="align_uniform_loss":
        return align_uniform_loss
    elif loss_name=="original_contrastive_loss":
        return original_contrastive_loss
    else:
        raise ValueError("not implemented yet")