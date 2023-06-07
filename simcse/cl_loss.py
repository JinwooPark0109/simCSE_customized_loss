import torch
import sys
from sklearn.mixture import GaussianMixture
import numpy as np

def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


def vne_loss(H):
    # https://github.com/xlpczv/SentRep/blob/main/SimCSE/simcse/models.py#L29
    assert (len(H.shape) == 2)
    Z = torch.nn.functional.normalize(H, dim=1)
    sing_val = torch.svd(Z / np.sqrt(Z.shape[0]))[1]
    eig_val = sing_val ** 2
    return - (eig_val * torch.log(eig_val)).nansum()


# todo: method to class
def align_uniform_loss(cls, tup):
    '''
    tup is tuple of (x,y,z)
    x (b,d) : one side of positive pair
    y (b,d) : other side of positive pair
    z (b,d) : currently not used. for hard negative
    '''

    z1, z2, _ = tup
    cos_sim = cls.sim(z1, z2)
    a = align_loss(z1, z2)
    u = uniform_loss(z1)
    loss = a + u

    print("custom loss used")
    # print("input:", z1.shape, z2.shape, tup)
    # print("cos sim:", cos_sim.shape, cos_sim)
    print("align loss:", a.item())
    print("uniform loss:", u.item())
    print("loss:", loss.item())

    return cos_sim, align_loss(z1, z2) + uniform_loss(z1)


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

    # from IPython import embed; embed()
    # sys.exit()

    loss = loss_fct(cos_sim, labels)

    # print("original loss")
    # print("input:", z1.shape, z2.shape, tup)
    # print("cos sim:", cos_sim.shape, cos_sim)
    # print("loss:", loss.item())

    return cos_sim, loss_fct(cos_sim, labels)

def simcse_vne_loss(cls, tup):

    z1, z2, _ = tup

    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    loss_fct = torch.nn.CrossEntropyLoss()

    loss=loss_fct(cos_sim, labels)-cls.coef*vne_loss(z1)
    #print(loss_fct(cos_sim, labels).item(), vne_loss(z1).item())

    return cos_sim, loss

def simcse_inverse_unifomity_loss(cls, tup):
    '''
    contrastive loss from original code
    '''
    z1, z2, _ = tup

    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    loss_fct = torch.nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels) - cls.coef * uniform_loss(z1)
    # loss = loss_fct(cos_sim, labels) + cls.coef * uniform_loss(z1)
    # loss = loss_fct(cos_sim, labels)
    print(loss_fct(cos_sim, labels).item(), uniform_loss(z1).item(),
          torch.pdist(z1, p=2).pow(2).mul(-2).exp().mean().item())

    return cos_sim, loss



def simcse_fixed_neg_loss(cls, tup):
    '''
    contrastive loss from original code
    '''
    z1, z2, _ = tup

    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    loss_fct = torch.nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)

    return cos_sim, loss


def simcse_fixed_vne_loss(cls, tup):
    '''
    contrastive loss from original code
    '''
    z1, z2, _ = tup

    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    loss_fct = torch.nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels) + cls.coef * vne_loss(z1 + cls.lower_bound)

    return cos_sim, loss


def original_detached_loss(cls, tup):
    '''
    contrastive loss from original code
    '''
    z1, z2, z3 = tup

    pos_cos_sim = cls.sim(z1, z2).unsqueeze(1)
    neg_cos_sim = cls.sim(z1.unsqueeze(1), z2.detach().clone().unsqueeze(0))
    mask = torch.eye(pos_cos_sim.size(0)).to(cls.device)
    cos_sim = pos_cos_sim * mask + neg_cos_sim * (1 - mask)

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    loss_fct = torch.nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)

    return cos_sim, loss


def original_pos_detached_loss(cls, tup):
    '''
    contrastive loss from original code
    '''
    z1, z2, z3 = tup

    pos_cos_sim = cls.sim(z1, z2).unsqueeze(1).detach().clone()
    neg_cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    mask = torch.eye(pos_cos_sim.size(0)).to(cls.device)
    cos_sim = pos_cos_sim * mask + neg_cos_sim * (1 - mask)

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    loss_fct = torch.nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)

    return cos_sim, loss


def original_neg_detached_loss(cls, tup):
    '''
    contrastive loss from original code
    '''
    z1, z2, z3 = tup

    pos_cos_sim = cls.sim(z1, z2).unsqueeze(1)
    neg_cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0)).detach().clone()
    mask = torch.eye(pos_cos_sim.size(0)).to(cls.device)
    cos_sim = pos_cos_sim * mask + neg_cos_sim * (1 - mask)

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    loss_fct = torch.nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)

    return cos_sim, loss


def original_rewrite_loss(cls, tup):
    '''
    contrastive loss from original code
    '''
    z1, z2, z3 = tup

    sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))

    pos_cos_sim = cls.sim(z1, z2).unsqueeze(1)
    neg_cos_sim = torch.triu(sim, diagonal=1)[:, 1:] + torch.tril(sim, diagonal=-1)[:, :-1]  # bx(b-1)

    cos_sim = torch.concat([pos_cos_sim, neg_cos_sim], dim=1)  # (b, b)
    labels = torch.zeros(cos_sim.size(0)).long().to(cls.device)  # 0th is pos at all batch items.

    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(cos_sim, labels)

    return cos_sim, loss


def original_eye_loss(cls, tup):
    '''
    contrastive loss from original code
    '''
    z1, z2, z3 = tup

    sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    mask = torch.eye(sim.size(0)).to(cls.device)
    cos_sim = sim * mask + sim * (1 - mask)
    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)

    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(cos_sim, labels)

    return cos_sim, loss


def tensor_contrastive_loss(cls, outputs, attention_mask):
    batch_size = attention_mask.size(0)
    num_sent = attention_mask.size(1)
    hidden = outputs.last_hidden_state
    hidden = hidden.view((*(attention_mask.shape), -1))  # b, num_sent, seq len, dim

    z1, z2 = hidden[:, 0], hidden[:, 1]  # b, seq len, dim

    # bxb
    bxb_cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    bxb_labels = torch.arange(bxb_cos_sim.size(0)).long().to(cls.device)
    # sxs
    sxs_cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    sxs_labels = torch.arange(sxs_cos_sim.size(0)).long().to(cls.device)
    # dxd
    dxd_cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    dxd_labels = torch.arange(dxd_cos_sim.size(0)).long().to(cls.device)

    loss_fct = torch.nn.CrossEntropyLoss()

    bxb_loss = loss_fct(bxb_cos_sim, bxb_labels)
    sxs_loss = loss_fct(sxs_cos_sim, sxs_labels)
    dxd_loss = loss_fct(dxd_cos_sim, dxd_labels)

    loss = bxb_loss + sxs_loss + dxd_loss

    return loss


def sbb_loss(cls, outputs, attention_mask):
    batch_size = attention_mask.size(0)
    num_sent = attention_mask.size(1)  # b,2,s
    hidden = outputs.last_hidden_state
    hidden = hidden.view((*(attention_mask.shape), -1))  # (b, num_sent, seq len, dim)

    z1, z2 = hidden[:, 0], hidden[:, 1]  # b,s,d

    sbb_cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))  # (b,1,s,d)x(1,b,s,d) -> (b,b,s)
    mask = attention_mask[:, 0, :]  # b,2,s -> b,s
    mask = (mask.unsqueeze(2) * mask.unsqueeze(1)).bool()  # (b,s,1) + (b,1, s) -> (b,s,s)

    cos_sim = sbb_cos_sim[mask]  # b,b, non pad only
    cos_sim = cos_sim.permute(2, 1, 0).reshape(-1, batch_size)
    sbb_labels = torch.arange(batch_size).long().to(cls.device).expand(cos_sim.size(0), 1)

    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(cos_sim, sbb_labels)

    return cos_sim, loss


def bss_loss(cls, outputs, attention_mask):
    # compare token in each sentence, so negative pair is came from in-sequence instead of in-batch

    batch_size, num_sent, seq_len = attention_mask.shape
    hidden = outputs.last_hidden_state
    hidden = hidden.view((*(attention_mask.shape), -1))  # (b, num_sent, seq len, dim)

    z1, z2 = hidden[:, 0], hidden[:, 1]  # b,s,d

    bss_cos_sim = cls.sim(z1.unsqueeze(2), z2.unsqueeze(1))  # (b,s,1,d)x(b,1,s,d)->(b,s,s)
    no_pad_idx = attention_mask[:, 0, :]  # b,2,s -> b,s, all attention mask are same in each pair
    no_pad_idx = (no_pad_idx.unsqueeze(2) * no_pad_idx.unsqueeze(1)).bool()  # (b,s,1) + (b,1, s) -> (b,s,s)

    # from IPython import embed;embed()
    pos_idx = no_pad_idx * (torch.eye(seq_len) == 1).bool().to(cls.device)

    '''
    bss_pos=bss_cos_sim[pos_idx]
    bss_numerater=torch.sum(bss_pos, 0)
    bss_denominator=bss_cos_sim[no_pad_idx].view(batch_size, -1)
    loss=(-bss_numerater + torch.logsumexp(bss_denominator)).mean()
    '''

    loss = 0
    for i in range(batch_size):
        bss_numerater = (bss_cos_sim[i][pos_idx[i]]).sum()
        bss_denominator = torch.logsumexp(bss_cos_sim[i][no_pad_idx[i]], 0)
        loss += (-bss_numerater + bss_denominator)
    # print("hi, I'm dummy for loop")

    return loss


def bb_dd_loss(cls, tup):
    z1, z2, _ = tup  # b,d
    loss_fct = torch.nn.CrossEntropyLoss()

    bb_cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))  # (b,1,d)x(1,b,d)=(b,b)
    bb_labels = torch.arange(bb_cos_sim.size(0)).long().to(cls.device)
    bb_loss = loss_fct(bb_cos_sim, bb_labels)

    dd_cos_sim = cls.sim(z1.permute(1, 0).unsqueeze(1), z2.permute(1, 0).unsqueeze(0))  # (d,1,b)x(1,d,b)=(d,d)
    dd_labels = torch.arange(dd_cos_sim.size(0)).long().to(cls.device)
    dd_loss = loss_fct(dd_cos_sim, dd_labels)

    loss = bb_loss + dd_loss

    return bb_cos_sim, loss


def negative_only_loss(cls, tup):
    z1, z2, _ = tup  # b,d

    batch_size, hidden_dim = z1.size()

    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))  # (b,b)
    neg_idx = torch.eye(batch_size) == 0

    loss = torch.logsumexp(cos_sim[neg_idx].view(batch_size, -1), dim=1).mean()  # (b, b-1)
    return cos_sim, loss


def no_augmented_loss(cls, tup):
    # print("no augmented")
    z1, _, _ = tup  # b,d

    cos_sim = cls.sim(z1.unsqueeze(1), z1.unsqueeze(0))

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    loss_fct = torch.nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)
    return cos_sim, loss


def no_augmented_label_smoothing_loss(cls, tup):
    # print("no augmented label smoothing")
    z1, _, _ = tup  # b,d

    cos_sim = cls.sim(z1.unsqueeze(1), z1.unsqueeze(0))

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    loss_fct = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    loss = loss_fct(cos_sim, labels)
    return cos_sim, loss


def no_augmented_label_smoothing_double_loss(cls, tup):
    # print("no augmented label smoothing")
    z1, _, _ = tup  # b,d

    cos_sim = cls.sim(z1.unsqueeze(1), z1.unsqueeze(0))

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    loss_fct = torch.nn.CrossEntropyLoss(label_smoothing=0.2)

    loss = loss_fct(cos_sim, labels)
    return cos_sim, loss


def no_augmented_band_loss(cls, tup):
    # print("no augmented label smoothing")
    z1, _, _ = tup  # b,d

    cos_sim = cls.sim(z1.unsqueeze(1), z1.unsqueeze(0) + z1.std(0))

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    loss_fct = torch.nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)
    return cos_sim, loss


def original_label_smoothing_loss(cls, tup):
    z1, z2, _ = tup

    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    loss_fct = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    loss = loss_fct(cos_sim, labels)
    return cos_sim, loss


def no_augmented_detached_loss(cls, tup):
    # print("no augmented")
    z1, _, _ = tup  # b,d

    z2 = z1.detach().clone()

    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    loss_fct = torch.nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)
    return cos_sim, loss


def really_dummy_positive_loss(cls, tup):
    # print("really dummy positive")
    z1, _, _ = tup  # b,d

    cos_sim = cls.sim(z1.unsqueeze(1), z1.unsqueeze(0))  # now there is only in-batch negative
    cos_sim.fill_diagonal_(1.0 / cls.sim.temp)  # set always 1

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    loss_fct = torch.nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)
    return cos_sim, loss


def noise_add_positive_loss(cls, tup):
    # print("noisy add positive")
    z1, _, _ = tup  # b,d

    z2 = z1 + torch.normal(mean=0, std=1, size=z1.shape).to(cls.device)  # d(x+noise)/dx=1
    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    loss_fct = torch.nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)  #
    return cos_sim, loss


def noise_add_simlar_loss(cls, tup):
    # print("noisy add similar")
    z1, _, _ = tup  # b,d

    noise_list = []
    zero = torch.zeros(z1.size(1)).to(cls.device)  # d
    z_std = z1.std(0)  # d
    for _ in z1:
        noise_list.append(torch.normal(mean=zero, std=z_std))
    noise = torch.stack(noise_list).to(cls.device)

    z2 = z1 + noise  # d(x+noise)/dx=1
    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    loss_fct = torch.nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)  #
    return cos_sim, loss


def noise_add_label_smoothing_loss(cls, tup):
    # print("noise_add_label_smoothing_loss")
    z1, _, _ = tup  # b,d

    z2 = z1 + torch.normal(mean=0, std=1, size=z1.shape).to(cls.device)  # d(x+noise)/dx=1
    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    loss_fct = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    loss = loss_fct(cos_sim, labels)  #
    return cos_sim, loss


def noise_add_similar_label_smoothing_loss(cls, tup):
    # print("noise_add_label_smoothing_loss")
    z1, _, _ = tup  # b,d

    noise_list = []
    zero = torch.zeros(z1.size(1)).to(cls.device)  # d
    z_std = z1.std(0)  # d
    for _ in z1:
        noise_list.append(torch.normal(mean=zero, std=z_std))
    noise = torch.stack(noise_list).to(cls.device)

    z2 = z1 + noise  # d(x+noise)/dx=1
    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    loss_fct = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    loss = loss_fct(cos_sim, labels)  #
    return cos_sim, loss


def noise_add_simlar_detached_loss(cls, tup):
    # print("noisy add similar")
    z1, _, _ = tup  # b,d

    noise_list = []
    zero = torch.zeros(z1.size(1)).to(cls.device)  # d
    z_std = z1.std(0)  # d
    for _ in z1:
        noise_list.append(torch.normal(mean=zero, std=z_std))
    noise = torch.stack(noise_list)

    z2 = (z1.clone().detach() + noise).to(cls.device)
    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    loss_fct = torch.nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)  #
    return cos_sim, loss


def noise_mul_positive_loss(cls, tup):
    # print("noisy mul positive")
    z1, _, _ = tup  # b,d

    z2 = z1 * torch.normal(mean=0, std=1, size=z1.shape).to(cls.device)  # d(x*noise)/dx=noise
    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    loss_fct = torch.nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)  #
    return cos_sim, loss


def noise_mul_one_loss(cls, tup):
    # print("noisy mul positive")
    z1, _, _ = tup  # b,d

    z2 = z1 * torch.normal(mean=1, std=1, size=z1.shape).to(cls.device)  # d(x*noise)/dx=noise
    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    loss_fct = torch.nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)  #
    return cos_sim, loss

def noise_mul_bernoulli_mask_loss(cls, tup):
    # print("noisy mul bernoulli")
    z1, _, _ = tup  # b,d

    z2 = z1 * torch.bernoulli(torch.full(z1.shape, 0.9)).to(cls.device)  # d(x*noise)/dx=noise
    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    loss_fct = torch.nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)  #
    return cos_sim, loss

def noise_mul_bernoulli_mask_08_loss(cls, tup):
    # print("noisy mul bernoulli")
    z1, _, _ = tup  # b,d

    z2 = z1 * torch.bernoulli(torch.full(z1.shape, 0.8)).to(cls.device)  # d(x*noise)/dx=noise
    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    loss_fct = torch.nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)  #
    return cos_sim, loss


def noise_mul_bernoulli_mask_05_loss(cls, tup):
    # print("noisy mul bernoulli")
    z1, _, _ = tup  # b,d

    z2 = z1 * torch.bernoulli(torch.full(z1.shape, 0.5)).to(cls.device)  # d(x*noise)/dx=noise
    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    loss_fct = torch.nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)  #
    return cos_sim, loss


def noise_mul_bernoulli_mask_detached_loss(cls, tup):
    # print("noisy mul bernoulli")
    z1, _, _ = tup  # b,d

    z2 = z1.detach() * torch.bernoulli(torch.full(z1.shape, 0.9)).to(cls.device)  # d(x*noise)/dx=noise
    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    loss_fct = torch.nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)  #
    return cos_sim, loss


def noise_add_3view_loss(cls, tup):
    # print("noisy add 3view")
    z1, _, _ = tup  # b,d

    z2 = z1 + torch.normal(mean=0, std=1, size=z1.shape).to(cls.device)
    z3 = z1 + torch.normal(mean=0, std=1, size=z1.shape).to(cls.device)

    cos_sim_12 = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    cos_sim_23 = cls.sim(z2.unsqueeze(1), z3.unsqueeze(0))
    cos_sim_31 = cls.sim(z3.unsqueeze(1), z1.unsqueeze(0))

    labels = torch.arange(cos_sim_12.size(0)).long().to(cls.device)
    loss_fct = torch.nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim_12, labels) + loss_fct(cos_sim_23, labels) + loss_fct(cos_sim_31, labels)
    return cos_sim_12, loss


def very_easy_negative_loss(cls, tup):
    # print("very easy negative")
    z1, z2, _ = tup  # b,d

    # noise = torch.normal(mean=0, std=1, size=z1.shape).to(cls.device)
    # noise = torch.normal(mean=z1.mean(), std=z1.std(), size=z1.shape).to(cls.device)

    noise_list = []
    z_mean = z1.mean(0)
    z_std = z1.std(0)
    for _ in range(z1.size(0)):
        noise_list.append(torch.normal(mean=z_mean, std=z_std))
    noise = torch.stack(noise_list)

    pos_cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    neg_cos_sim = cls.sim(z1.unsqueeze(1), noise.unsqueeze(0))

    mask = torch.eye(pos_cos_sim.size(0)).to(cls.device)
    cos_sim = pos_cos_sim * mask + neg_cos_sim * (1 - mask)  # diagonal from positive, others from random noise

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    loss_fct = torch.nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)
    return cos_sim, loss


def very_hard_negative_loss(cls, tup):
    # print("very hard negative")
    z1, z2, _ = tup  # b,d

    hard = z1 + torch.normal(mean=0, std=1, size=z1.shape).to(cls.device)  # d(x+noise)/dx=1
    pos_cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    neg_cos_sim = cls.sim(z1.unsqueeze(1), hard.unsqueeze(0))

    mask = torch.eye(pos_cos_sim.size(0)).to(cls.device)
    cos_sim = pos_cos_sim * mask + neg_cos_sim * (1 - mask)

    labels = torch.arange(pos_cos_sim.size(0)).long().to(cls.device)
    loss_fct = torch.nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)
    return cos_sim, loss


def very_hard_similar_loss(cls, tup):
    # print("very hard negative")
    z1, z2, _ = tup  # b,d

    hard_list = []
    zero = torch.zeros(z1.size(1)).to(cls.device)  # d
    z_std = z1.std(0)  # d
    for sample in z1:
        hard_list.append(sample + torch.normal(mean=zero, std=z_std))
    hard = torch.stack(hard_list).to(cls.device)

    pos_cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    neg_cos_sim = cls.sim(z1.unsqueeze(1), hard.unsqueeze(0))

    mask = torch.eye(pos_cos_sim.size(0)).to(cls.device)
    cos_sim = pos_cos_sim * mask + neg_cos_sim * (1 - mask)

    labels = torch.arange(pos_cos_sim.size(0)).long().to(cls.device)
    loss_fct = torch.nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)
    return cos_sim, loss


def very_hard_similar_detached_loss(cls, tup):
    # print("very hard negative")
    z1, z2, _ = tup  # b,d

    hard_list = []
    zero = torch.zeros(z1.size(1)).to(cls.device)  # d
    z_std = z1.std(0)  # d
    for sample in z1:
        hard_list.append(sample.detach().clone() + torch.normal(mean=zero, std=z_std))
    hard = torch.stack(hard_list).to(cls.device)

    pos_cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    neg_cos_sim = cls.sim(z1.unsqueeze(1), hard.unsqueeze(0))

    mask = torch.eye(pos_cos_sim.size(0)).to(cls.device)
    cos_sim = pos_cos_sim * mask + neg_cos_sim * (1 - mask)

    labels = torch.arange(pos_cos_sim.size(0)).long().to(cls.device)
    loss_fct = torch.nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)
    return cos_sim, loss


def very_hard_similar_detached_rewrite_loss(cls, tup):
    # print("very hard negative")
    z1, z2, _ = tup  # b,d

    hard_list = []
    zero = torch.zeros(z1.size(1)).to(cls.device)  # d
    z_std = z1.std(0)  # d
    for sample in z1:
        hard_list.append(sample.detach().clone() + torch.normal(mean=zero, std=z_std))
    hard = torch.stack(hard_list).to(cls.device)

    pos_cos_sim = cls.sim(z1, z2).unsqueeze(1)
    neg_cos_sim = cls.sim(z1.unsqueeze(1), hard.unsqueeze(0))  # remove diagnoal
    neg_cos_sim = torch.triu(neg_cos_sim, diagonal=1)[:, 1:] + torch.tril(neg_cos_sim, diagonal=-1)[:, :-1]  # bx(b-1)

    cos_sim = torch.concat([pos_cos_sim, neg_cos_sim], dim=1)  # (b, b)
    labels = torch.zeros(cos_sim.size(0)).long().to(cls.device)  # 0th is pos at all batch items.

    loss_fct = torch.nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)
    return cos_sim, loss


def very_hard_similar_detached_rewrite_0_01_loss(cls, tup):
    # print("very hard negative")
    z1, z2, _ = tup  # b,d

    hard_list = []
    zero = torch.zeros(z1.size(1)).to(cls.device)  # d
    z_std = z1.std(0)  # d
    for sample in z1:
        hard_list.append(sample.detach().clone() + torch.normal(mean=zero, std=0.01))
    hard = torch.stack(hard_list).to(cls.device)

    pos_cos_sim = cls.sim(z1, z2).unsqueeze(1)
    neg_cos_sim = cls.sim(z1.unsqueeze(1), hard.unsqueeze(0))  # remove diagnoal
    neg_cos_sim = torch.triu(neg_cos_sim, diagonal=1)[:, 1:] + torch.tril(neg_cos_sim, diagonal=-1)[:, :-1]  # bx(b-1)

    cos_sim = torch.concat([pos_cos_sim, neg_cos_sim], dim=1)  # (b, b)
    labels = torch.zeros(cos_sim.size(0)).long().to(cls.device)  # 0th is pos at all batch items.

    loss_fct = torch.nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)
    return cos_sim, loss


def very_hard_similar_detached_rewrite_0_1_loss(cls, tup):
    # print("very hard negative")
    z1, z2, _ = tup  # b,d

    hard_list = []
    zero = torch.zeros(z1.size(1)).to(cls.device)  # d
    z_std = z1.std(0)  # d
    for sample in z1:
        hard_list.append(sample.detach().clone() + torch.normal(mean=zero, std=0.1))
    hard = torch.stack(hard_list).to(cls.device)

    pos_cos_sim = cls.sim(z1, z2).unsqueeze(1)
    neg_cos_sim = cls.sim(z1.unsqueeze(1), hard.unsqueeze(0))  # remove diagnoal
    neg_cos_sim = torch.triu(neg_cos_sim, diagonal=1)[:, 1:] + torch.tril(neg_cos_sim, diagonal=-1)[:, :-1]  # bx(b-1)

    cos_sim = torch.concat([pos_cos_sim, neg_cos_sim], dim=1)  # (b, b)
    labels = torch.zeros(cos_sim.size(0)).long().to(cls.device)  # 0th is pos at all batch items.

    loss_fct = torch.nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)
    return cos_sim, loss


def very_hard_similar_detached_rewrite_1_loss(cls, tup):
    # print("very hard negative")
    z1, z2, _ = tup  # b,d

    hard_list = []
    zero = torch.zeros(z1.size(1)).to(cls.device)  # d
    for sample in z1:
        hard_list.append(sample.detach().clone() + torch.normal(mean=zero, std=1))
    hard = torch.stack(hard_list).to(cls.device)

    pos_cos_sim = cls.sim(z1, z2).unsqueeze(1)
    neg_cos_sim = cls.sim(z1.unsqueeze(1), hard.unsqueeze(0))  # remove diagnoal
    neg_cos_sim = torch.triu(neg_cos_sim, diagonal=1)[:, 1:] + torch.tril(neg_cos_sim, diagonal=-1)[:, :-1]  # bx(b-1)

    cos_sim = torch.concat([pos_cos_sim, neg_cos_sim], dim=1)  # (b, b)
    labels = torch.zeros(cos_sim.size(0)).long().to(cls.device)  # 0th is pos at all batch items.

    loss_fct = torch.nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)
    return cos_sim, loss


def many_very_hard_similar_loss(cls, tup):
    # print("very hard negative")
    z1, z2, _ = tup  # b,d

    hard_list = []
    zero = torch.zeros(z1.size(1)).to(cls.device)  # d
    z_std = z1.std(0)  # d
    for sample in z1:
        hard_list.append(sample + torch.normal(mean=zero, std=z_std))
    hard = torch.stack(hard_list).to(cls.device)

    hard_extra_list = []
    for _ in range(9):  # just for remove in-batch negative
        for sample in z1:
            hard_extra_list.append(sample + torch.normal(mean=zero, std=z_std))
    hard_extra = torch.stack(hard_extra_list).to(cls.device)  # 9b, d

    pos_cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))  # b,b
    neg_cos_sim = cls.sim(z1.unsqueeze(1), hard.unsqueeze(0))
    neg_extra_cos_sim = cls.sim(z1.unsqueeze(1), hard_extra.unsqueeze(0))  # (b,1,d)x(1,9b,d)=(b,9b)

    mask = torch.eye(pos_cos_sim.size(0)).to(cls.device)
    cos_sim = torch.cat([pos_cos_sim * mask + neg_cos_sim * (1 - mask), neg_extra_cos_sim], dim=1)  # b,10b

    labels = torch.arange(pos_cos_sim.size(0)).long().to(cls.device)
    loss_fct = torch.nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)
    return cos_sim, loss


def easy_origin_negative_loss(cls, tup):
    # print("easy origin negative")
    z1, z2, _ = tup  # b,d

    noise_list = []
    z_mean = z1.mean(0)
    z_std = z1.std(0)
    for _ in range(z1.size(0)):
        noise_list.append(torch.normal(mean=z_mean, std=z_std))
    noise = torch.stack(noise_list)

    origin_cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    easy_cos_sim = cls.sim(z1.unsqueeze(1), noise.unsqueeze(0))

    cls.log_sim(origin_cos_sim, easy_cos_sim)

    cos_sim = torch.concat([origin_cos_sim, easy_cos_sim], dim=1)  # (b, 2b)

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)  # b

    loss_fct = torch.nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)  #
    return cos_sim, loss


def very_easy_origin_negative_loss(cls, tup):
    # print("easy origin negative")
    z1, z2, _ = tup  # b,d

    noise = torch.normal(mean=0, std=1, size=z1.shape).to(cls.device)

    origin_cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    easy_cos_sim = cls.sim(z1.unsqueeze(1), noise.unsqueeze(0))

    cos_sim = torch.concat([origin_cos_sim, easy_cos_sim], dim=1)  # (b, 2b)

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)  # b

    loss_fct = torch.nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)  #
    return cos_sim, loss


def easy_easy_origin_negative_loss(cls, tup):
    # print("easy easy origin negative")
    z1, z2, _ = tup  # b,d

    noise_list = []
    z_mean = z1.mean(0)
    z_std = z1.std(0)
    for _ in range(z1.size(0) * 2):
        noise_list.append(torch.normal(mean=z_mean, std=z_std))
    noise = torch.stack(noise_list)

    origin_cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    easy_cos_sim = cls.sim(z1.unsqueeze(1), noise.unsqueeze(0))  # (b,1,d)x(1,2b,d)=(bx2b)

    cos_sim = torch.concat([origin_cos_sim, easy_cos_sim], dim=1)  # (b, 3b)

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)  # b

    loss_fct = torch.nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)  #
    return cos_sim, loss


def many_easy_origin_negative_loss(cls, tup):
    # print("easy easy origin negative")
    z1, z2, _ = tup  # b,d

    noise_list = []
    z_mean = z1.mean(0)
    z_std = z1.std(0)
    for _ in range(z1.size(0) * 10):
        noise_list.append(torch.normal(mean=z_mean, std=z_std))
    noise = torch.stack(noise_list)

    origin_cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    easy_cos_sim = cls.sim(z1.unsqueeze(1), noise.unsqueeze(0))  # (b,1,d)x(1,2b,d)=(bx2b)

    cos_sim = torch.concat([origin_cos_sim, easy_cos_sim], dim=1)  # (b, 3b)

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)  # b

    loss_fct = torch.nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)  #
    return cos_sim, loss


def k_easy_origin_negative_loss(cls, tup):
    # print("easy easy origin negative")
    z1, z2, _ = tup  # b,d

    noise_list = []
    z_mean = z1.mean(0)
    z_std = z1.std(0)
    for _ in range(z1.size(0) * cls.k):
        noise_list.append(torch.normal(mean=z_mean, std=z_std))
    noise = torch.stack(noise_list)

    origin_cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    easy_cos_sim = cls.sim(z1.unsqueeze(1), noise.unsqueeze(0))  # (b,1,d)x(1,2b,d)=(bx2b)

    cos_sim = torch.concat([origin_cos_sim, easy_cos_sim], dim=1)  # (b, 3b)

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)  # b

    loss_fct = torch.nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)  #
    return cos_sim, loss


def easy_origin_negative_wo_inbatch_loss(cls, tup):
    # print("easy origin negative")
    z1, z2, _ = tup  # b,d

    noise_list = []
    z_mean = z1.mean(0).detach().clone()
    z_std = z1.std(0).detach().clone()
    for _ in range(z1.size(0)):
        noise_list.append(torch.normal(mean=z_mean, std=z_std))
    noise = torch.stack(noise_list).to(cls.device)

    origin_cos_sim = cls.sim(z1, z2).unsqueeze(1)  # (b,1)
    # easy_cos_sim = cls.sim(z1.unsqueeze(1), noise.unsqueeze(0)) # (b,b)
    easy_cos_sim = cls.sim(z1.unsqueeze(1), noise.unsqueeze(0))  # remove diagnoal
    easy_cos_sim = torch.triu(easy_cos_sim, diagonal=1)[:, 1:] + torch.tril(easy_cos_sim, diagonal=-1)[:,
                                                                 :-1]  # bx(b-1)

    cos_sim = torch.concat([origin_cos_sim, easy_cos_sim], dim=1)  # (b, b+1)

    labels = torch.zeros(cos_sim.size(0)).long().to(cls.device)  # 0th is pos at all batch items.

    loss_fct = torch.nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)  #
    return cos_sim, loss


def many_very_easy_origin_negative_loss(cls, tup):
    # print("easy easy origin negative")
    z1, z2, _ = tup  # b,d

    noise = torch.normal(mean=0, std=1, size=(10 * z1.size(0), z1.size(1))).to(cls.device)  # very easy noise

    origin_cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    easy_cos_sim = cls.sim(z1.unsqueeze(1), noise.unsqueeze(0))

    cos_sim = torch.concat([origin_cos_sim, easy_cos_sim], dim=1)  # (b,3b)
    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)  # b
    loss_fct = torch.nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)  #
    return cos_sim, loss


def easy_cov_origin_loss(cls, tup):
    # print("easy origin negative")
    z1, z2, _ = tup  # b,d

    noise_list = []
    z_mean = z1.mean(0)
    z_std = z1.std(0)
    cov_mat = torch.cov(z1.T)
    for _ in range(z1.size(0)):
        noise_list.append(torch.matmul(torch.normal(mean=z_mean, std=z_std), cov_mat))
    noise = torch.stack(noise_list)

    origin_cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    easy_cos_sim = cls.sim(z1.unsqueeze(1), noise.unsqueeze(0))

    cls.log_sim(origin_cos_sim, easy_cos_sim)

    cos_sim = torch.concat([origin_cos_sim, easy_cos_sim], dim=1)  # (b, 2b)

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)  # b

    loss_fct = torch.nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)  #
    return cos_sim, loss


def very_easy_cov_origin_loss(cls, tup):
    # print("easy origin negative")
    z1, z2, _ = tup  # b,d

    noise_list = []
    cov_mat = torch.cov(z1.T)
    for _ in range(z1.size(0)):
        noise_list.append(torch.matmul(torch.normal(mean=0, std=1, size=z1[0].shape).to(cls.device), cov_mat))
    noise = torch.stack(noise_list)

    origin_cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    easy_cos_sim = cls.sim(z1.unsqueeze(1), noise.unsqueeze(0))

    cls.log_sim(origin_cos_sim, easy_cos_sim)

    cos_sim = torch.concat([origin_cos_sim, easy_cos_sim], dim=1)  # (b, 2b)

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)  # b

    loss_fct = torch.nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)  #
    return cos_sim, loss


def hard_origin_negative_loss(cls, tup):
    # print("hard origin negative")
    z1, z2, _ = tup  # b,d

    hard = z1 + torch.normal(mean=0, std=1, size=z1.shape).to(cls.device)  # d(x+noise)/dx=1

    origin_cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    hard_cos_sim = cls.sim(z1.unsqueeze(1), hard.unsqueeze(0))

    cos_sim = torch.concat([origin_cos_sim, hard_cos_sim], dim=1)  # (b, 2b)

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)  # b

    loss_fct = torch.nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)  #
    return cos_sim, loss


def hard_similar_origin_negative_loss(cls, tup):
    # print("hard origin negative")
    z1, z2, _ = tup  # b,d

    hard_list = []
    zero = torch.zeros(z1.size(1)).to(cls.device)  # d
    z_std = z1.std(0)  # d
    for sample in z1:
        hard_list.append(sample + torch.normal(mean=zero, std=z_std))
    hard = torch.stack(hard_list).to(cls.device)

    origin_cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    hard_cos_sim = cls.sim(z1.unsqueeze(1), hard.unsqueeze(0))

    cos_sim = torch.concat([origin_cos_sim, hard_cos_sim], dim=1)  # (b, 2b)

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)  # b

    loss_fct = torch.nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)  #
    return cos_sim, loss


def hard_similar_origin_negative_detach_loss(cls, tup):
    # print("hard origin negative")
    z1, z2, _ = tup  # b,d

    hard_list = []
    zero = torch.zeros(z1.size(1)).to(cls.device)  # d
    z_std = z1.std(0)  # d
    for sample in z1:
        hard_list.append(sample.detach().clone() + torch.normal(mean=zero, std=z_std))
    hard = torch.stack(hard_list).to(cls.device)

    origin_cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    # hard_cos_sim = cls.sim(z1.unsqueeze(1), hard.unsqueeze(0))
    hard_cos_sim = cls.sim(z1.unsqueeze(1), hard.unsqueeze(0))  # remove diagnoal
    hard_cos_sim = torch.triu(hard_cos_sim, diagonal=1)[:, 1:] + torch.tril(hard_cos_sim, diagonal=-1)[:,
                                                                 :-1]  # bx(b-1)

    cos_sim = torch.concat([origin_cos_sim, hard_cos_sim], dim=1)  # (b, 2b)

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)  # b

    loss_fct = torch.nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)  #
    return cos_sim, loss


def many_hard_similar_origin_negative_loss(cls, tup):
    # print("hard origin negative")
    z1, z2, _ = tup  # b,d

    hard_list = []
    zero = torch.zeros(z1.size(1)).to(cls.device)  # d
    z_std = z1.std(0)  # d
    for _ in range(10):
        for sample in z1:
            hard_list.append(sample + torch.normal(mean=zero, std=z_std))
    hard = torch.stack(hard_list).to(cls.device)  # 10b,d

    origin_cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    hard_cos_sim = cls.sim(z1.unsqueeze(1), hard.unsqueeze(0))

    cos_sim = torch.concat([origin_cos_sim, hard_cos_sim], dim=1)  # (b, 11b)

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)  # b

    loss_fct = torch.nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)  #
    return cos_sim, loss


def hard_hard_origin_negative_loss(cls, tup):
    # print("hard hard origin negative")
    z1, z2, _ = tup  # b,d

    hard = z1 + torch.normal(mean=0, std=1, size=z1.shape).to(cls.device)  # d(x+noise)/dx=1
    hard2 = z1 + torch.normal(mean=0, std=1, size=z1.shape).to(cls.device)

    origin_cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    hard_cos_sim = cls.sim(z1.unsqueeze(1), hard.unsqueeze(0))
    hard2_cos_sim = cls.sim(z1.unsqueeze(1), hard2.unsqueeze(0))

    cos_sim = torch.concat([origin_cos_sim, hard_cos_sim, hard2_cos_sim], dim=1)  # (b, 3b)

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)  # b

    loss_fct = torch.nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)  #
    return cos_sim, loss


def easy_hard_origin_negative_loss(cls, tup):
    # print("easy hard origin negative")
    z1, z2, _ = tup  # b,d

    noise_list = []
    z_mean = z1.mean(0)
    z_std = z1.std(0)
    for _ in range(z1.size(0)):
        noise_list.append(torch.normal(mean=z_mean, std=z_std))
    noise = torch.stack(noise_list)
    hard = z1 + torch.normal(mean=0, std=1, size=z1.shape).to(cls.device)  # d(x+noise)/dx=1

    origin_cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    easy_cos_sim = cls.sim(z1.unsqueeze(1), noise.unsqueeze(0))
    hard_cos_sim = cls.sim(z1.unsqueeze(1), hard.unsqueeze(0))

    cos_sim = torch.concat([origin_cos_sim, easy_cos_sim, hard_cos_sim], dim=1)  # (b, 3b)

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)  # b

    loss_fct = torch.nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)  #
    return cos_sim, loss


def easy_similar_hard_origin_negative_loss(cls, tup):
    # print("easy hard origin negative")
    z1, z2, _ = tup  # b,d

    noise_list = []
    z_mean = z1.mean(0)
    z_std = z1.std(0)
    for _ in range(z1.size(0)):
        noise_list.append(torch.normal(mean=z_mean, std=z_std))
    noise = torch.stack(noise_list)

    hard_list = []
    zero = torch.zeros(z1.size(1)).to(cls.device)  # d
    z_std = z1.std(0)  # d
    for sample in z1:
        hard_list.append(sample + torch.normal(mean=zero, std=z_std))
    hard = torch.stack(hard_list).to(cls.device)

    origin_cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    easy_cos_sim = cls.sim(z1.unsqueeze(1), noise.unsqueeze(0))
    hard_cos_sim = cls.sim(z1.unsqueeze(1), hard.unsqueeze(0))

    cos_sim = torch.concat([origin_cos_sim, easy_cos_sim, hard_cos_sim], dim=1)  # (b, 3b)

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)  # b

    loss_fct = torch.nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)  #
    return cos_sim, loss


def beam_hard_1_loss(cls, tup):
    z1, _, _ = tup  # b,d

    z2_list = []
    for sample in z1:
        z2_list.append(sample.detach().clone() + torch.normal(mean=0, std=1, size=sample.shape).to(cls.device))
    z2 = torch.stack(z2_list).to(cls.device)

    hard_list = []
    zero = torch.zeros(z1.size(1)).to(cls.device)  # d
    z_std = z1.std(0)  # d
    for sample in z1:
        hard_list.append(sample.detach().clone() + torch.normal(mean=zero, std=z_std))
    hard = torch.stack(hard_list).to(cls.device)

    pos_cos_sim = cls.sim(z1, z2).unsqueeze(1)
    neg_cos_sim = cls.sim(z1.unsqueeze(1), hard.unsqueeze(0))  # remove diagnoal
    neg_cos_sim = torch.triu(neg_cos_sim, diagonal=1)[:, 1:] + torch.tril(neg_cos_sim, diagonal=-1)[:, :-1]  # bx(b-1)

    cos_sim = torch.concat([pos_cos_sim, neg_cos_sim], dim=1)  # (b, b)
    labels = torch.zeros(cos_sim.size(0)).long().to(cls.device)  # 0th is pos at all batch items.

    loss_fct = torch.nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)
    return cos_sim, loss


def beam_hard_0_1_loss(cls, tup):
    z1, _, _ = tup  # b,d

    z2_list = []
    for sample in z1:
        z2_list.append(sample.detach().clone() + torch.normal(mean=0, std=0.1, size=sample.shape).to(cls.device))
    z2 = torch.stack(z2_list).to(cls.device)

    hard_list = []
    zero = torch.zeros(z1.size(1)).to(cls.device)  # d
    z_std = z1.std(0)  # d
    for sample in z1:
        hard_list.append(sample.detach().clone() + torch.normal(mean=zero, std=z_std))
    hard = torch.stack(hard_list).to(cls.device)

    pos_cos_sim = cls.sim(z1, z2).unsqueeze(1)
    neg_cos_sim = cls.sim(z1.unsqueeze(1), hard.unsqueeze(0))  # remove diagnoal
    neg_cos_sim = torch.triu(neg_cos_sim, diagonal=1)[:, 1:] + torch.tril(neg_cos_sim, diagonal=-1)[:, :-1]  # bx(b-1)

    cos_sim = torch.concat([pos_cos_sim, neg_cos_sim], dim=1)  # (b, b)
    labels = torch.zeros(cos_sim.size(0)).long().to(cls.device)  # 0th is pos at all batch items.

    loss_fct = torch.nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)
    return cos_sim, loss


def beam_hard_0_01_loss(cls, tup):
    z1, _, _ = tup  # b,d

    z2_list = []
    for sample in z1:
        z2_list.append(sample.detach().clone() + torch.normal(mean=0, std=0.01, size=sample.shape).to(cls.device))
    z2 = torch.stack(z2_list).to(cls.device)

    hard_list = []
    zero = torch.zeros(z1.size(1)).to(cls.device)  # d
    z_std = z1.std(0)  # d
    for sample in z1:
        hard_list.append(sample.detach().clone() + torch.normal(mean=zero, std=z_std))
    hard = torch.stack(hard_list).to(cls.device)

    pos_cos_sim = cls.sim(z1, z2).unsqueeze(1)
    neg_cos_sim = cls.sim(z1.unsqueeze(1), hard.unsqueeze(0))  # remove diagnoal
    neg_cos_sim = torch.triu(neg_cos_sim, diagonal=1)[:, 1:] + torch.tril(neg_cos_sim, diagonal=-1)[:, :-1]  # bx(b-1)

    cos_sim = torch.concat([pos_cos_sim, neg_cos_sim], dim=1)  # (b, b)
    labels = torch.zeros(cos_sim.size(0)).long().to(cls.device)  # 0th is pos at all batch items.

    loss_fct = torch.nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)
    return cos_sim, loss


def add_and_sample_loss(cls, tup):
    z1, _, _ = tup  # b,d

    z_mean = z1.mean(0)
    z_std = z1.std(0)  # d

    z2_list = []
    zero = torch.zeros(z1.size(1)).to(cls.device)  # d
    for sample in z1:
        z2_list.append(sample.detach().clone() + torch.normal(mean=zero, std=z_std))
    z2 = torch.stack(z2_list).to(cls.device)

    hard_list = []
    for _ in range(z1.size(0)):
        hard_list.append(torch.normal(mean=z_mean, std=z_std))
    hard = torch.stack(hard_list)

    pos_cos_sim = cls.sim(z1, z2).unsqueeze(1)
    neg_cos_sim = cls.sim(z1.unsqueeze(1), hard.unsqueeze(0))  # remove diagnoal
    neg_cos_sim = torch.triu(neg_cos_sim, diagonal=1)[:, 1:] + torch.tril(neg_cos_sim, diagonal=-1)[:, :-1]  # bx(b-1)

    cos_sim = torch.concat([pos_cos_sim, neg_cos_sim], dim=1)  # (b, b)
    labels = torch.zeros(cos_sim.size(0)).long().to(cls.device)  # 0th is pos at all batch items.

    loss_fct = torch.nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)
    return cos_sim, loss


def k_gmm_negative_loss(cls, tup):
    z1, z2, _ = tup  # b,d

    gmm = GaussianMixture(n_components=cls.gmm_comp).fit(z1.cpu().detach().numpy())
    noise = torch.tensor(gmm.sample(z1.size(0) * cls.k)[0]).to(cls.device)

    origin_cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    easy_cos_sim = cls.sim(z1.unsqueeze(1), noise.unsqueeze(0))  # (b,1,d)x(1,kb,d)=(b,kb)

    cos_sim = torch.cat([origin_cos_sim, easy_cos_sim], dim=1)  # (b, kb+1)
    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)  # b

    loss_fct = torch.nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)
    return cos_sim, loss


def k_gmm_buffer_loss(cls, tup):
    z1, z2, _ = tup  # b,d

    cls.replace_gmm(z1)
    noise = cls.gmm_sampling(z1.size(0) * cls.k)

    origin_cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    easy_cos_sim = cls.sim(z1.unsqueeze(1), noise.unsqueeze(0))  # (b,1,d)x(1,kb,d)=(b,kb)

    cos_sim = torch.concat([origin_cos_sim, easy_cos_sim], dim=1)  # (b, kb+1)
    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)  # b

    loss_fct = torch.nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)
    return cos_sim, loss


def k_gmm_maintained_loss(cls, tup):
    z1, z2, _ = tup  # b,d

    cls.update_gmm(z1)
    noise = cls.gmm_sampling(z1.size(0) * cls.k)

    origin_cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    easy_cos_sim = cls.sim(z1.unsqueeze(1), noise.unsqueeze(0))  # (b,1,d)x(1,kb,d)=(b,kb)

    cos_sim = torch.concat([origin_cos_sim, easy_cos_sim], dim=1)  # (b, kb+1)
    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)  # b

    loss_fct = torch.nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)
    return cos_sim, loss


def k_gmm_buffer_periodic_update_loss(cls, tup):
    z1, z2, _ = tup  # b,d

    noise = cls.periodic_update_gmm_sampler(z1, z1.size(0) * cls.k)

    origin_cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    easy_cos_sim = cls.sim(z1.unsqueeze(1), noise.unsqueeze(0))  # (b,1,d)x(1,kb,d)=(b,kb)

    cos_sim = torch.concat([origin_cos_sim, easy_cos_sim], dim=1)  # (b, kb+1)
    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)  # b

    loss_fct = torch.nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)
    return cos_sim, loss


def k_N_buffer_loss(cls, tup):
    # print("easy origin negative")
    z1, z2, _ = tup  # b,d

    noise_list = []
    z_mean, z_std = cls.get_buffer_mean_and_std(z1)
    for _ in range(z1.size(0)):
        noise_list.append(torch.normal(mean=z_mean, std=z_std))
    noise = torch.stack(noise_list).to(cls.device)

    origin_cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    easy_cos_sim = cls.sim(z1.unsqueeze(1), noise.unsqueeze(0))

    cos_sim = torch.concat([origin_cos_sim, easy_cos_sim], dim=1)  # (b, 2b)

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)  # b

    loss_fct = torch.nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)  #
    return cos_sim, loss


def k_vae_loss(cls, tup):
    z1, z2, _ = tup  # b,d

    cls.update_vae(z1.clone().detach())
    noise = cls.vae_sampling(z1.size(0) * cls.k)

    origin_cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    neg_cos_sim = cls.sim(z1.unsqueeze(1), noise.unsqueeze(0))  # (b,1,d)x(1,kb,d)=(b,kb)

    cos_sim = torch.concat([origin_cos_sim, neg_cos_sim], dim=1)  # (b, kb+1)
    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)  # b

    loss_fct = torch.nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)
    return cos_sim, loss


def no_inbatch_neg_loss(cls, tup):
    z1, z2, _ = tup

    hard_list = []
    zero = torch.zeros(z1.size(1)).to(cls.device)  # d
    for sample in z1:
        appender = []
        for _ in range(len(z1) - 1):  # b-1
            appender.append(sample.detach().clone() + torch.normal(mean=zero, std=0.1))
        hard_list.append(torch.stack(appender))
    hard = torch.stack(hard_list).to(cls.device)  # (b,b-1,d)

    pos_cos_sim = cls.sim(z1, z2).unsqueeze(1)  # (b,1)
    neg_cos_sim = cls.sim(z1.unsqueeze(1), hard)  # (b, 1, d)x(b, b-1, d) = (b, b-1). (b,1) broadcasted to (b,b)

    cos_sim = torch.concat([pos_cos_sim, neg_cos_sim], dim=1)  # (b, b)
    labels = torch.zeros(cos_sim.size(0)).long().to(cls.device)  # 0th is pos at all batch items.

    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(cos_sim, labels)

    return cos_sim, loss


def no_inbatch_neg_origin_loss(cls, tup):
    z1, z2, _ = tup

    hard_list = []
    zero = torch.zeros(z1.size(1)).to(cls.device)  # d
    for sample in z1:
        appender = []
        for _ in range(len(z1)):  # b
            appender.append(sample.detach().clone() + torch.normal(mean=zero, std=0.1))
        hard_list.append(torch.stack(appender))
    hard = torch.stack(hard_list).to(cls.device)  # (b,b,d)

    pos_cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    neg_cos_sim = cls.sim(z1.unsqueeze(1), hard)  # (b, 1, d)x(b, b, d) = (b, b). (b,1) broadcasted to (b,b)

    cos_sim = torch.concat([pos_cos_sim, neg_cos_sim], dim=1)
    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)

    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(cos_sim, labels)

    return cos_sim, loss


def no_inbatch_neg_one_origin_loss(cls, tup):
    z1, z2, _ = tup

    hard_list = []
    zero = torch.zeros(z1.size(1)).to(cls.device)  # d
    for sample in z1:
        hard_list.append(sample.detach().clone() + torch.normal(mean=zero, std=0.1))
    hard = torch.stack(hard_list).to(cls.device)  # (b,1,d)

    pos_cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    neg_cos_sim = cls.sim(z1.unsqueeze(1),
                          hard.unsqueeze(1))  # (b, 1, d)x(b, 1, d) = (b, b). (b,1) broadcasted to (b,b)

    cos_sim = torch.concat([pos_cos_sim, neg_cos_sim], dim=1)
    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)

    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(cos_sim, labels)

    return cos_sim, loss


def no_inbatch_similar_neg_loss(cls, tup):
    z1, z2, _ = tup

    hard_list = []
    zero = torch.zeros(z1.size(1)).to(cls.device)  # d
    z_std = z1.std(0)
    for sample in z1:
        appender = []
        for _ in range(len(z1) - 1):  # b-1
            appender.append(sample.detach().clone() + torch.normal(mean=zero, std=z_std))
        hard_list.append(torch.stack(appender))
    hard = torch.stack(hard_list).to(cls.device)  # (b,b-1,d)

    pos_cos_sim = cls.sim(z1, z2).unsqueeze(1)
    neg_cos_sim = cls.sim(z1.unsqueeze(1), hard)

    cos_sim = torch.concat([pos_cos_sim, neg_cos_sim], dim=1)  # (b, b)
    labels = torch.zeros(cos_sim.size(0)).long().to(cls.device)  # 0th is pos at all batch items.

    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(cos_sim, labels)

    return cos_sim, loss


def no_inbatch_neg_attach_loss(cls, tup):
    z1, z2, _ = tup

    hard_list = []
    zero = torch.zeros(z1.size(1)).to(cls.device)  # d
    for sample in z1:
        appender = []
        for _ in range(len(z1) - 1):  # b-1
            appender.append(sample + torch.normal(mean=zero, std=0.1))
        hard_list.append(torch.stack(appender))
    hard = torch.stack(hard_list).to(cls.device)  # (b,b-1,d)

    pos_cos_sim = cls.sim(z1, z2).unsqueeze(1)  # (b,1)
    neg_cos_sim = cls.sim(z1.unsqueeze(1), hard)  # (b, 1, d)x(b, b-1, d) = (b, b-1). (b,1) broadcasted to (b,b)

    cos_sim = torch.concat([pos_cos_sim, neg_cos_sim], dim=1)  # (b, b)
    labels = torch.zeros(cos_sim.size(0)).long().to(cls.device)  # 0th is pos at all batch items.

    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(cos_sim, labels)

    return cos_sim, loss


def no_inbatch_pos_origin_loss(cls, tup):
    '''
    multi-view
    '''

    z1, z2, _ = tup

    hard_list = []
    zero = torch.zeros(z1.size(1)).to(cls.device)  # d
    for sample in z1:
        appender = []
        for _ in range(len(z1)):  # b
            appender.append(sample.detach().clone() + torch.normal(mean=zero, std=0.1))
        hard_list.append(torch.stack(appender))
    hard = torch.stack(hard_list).to(cls.device)  # (b,b,d)

    pos_cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    multi_pos_cos_sim = cls.sim(z1.unsqueeze(1), hard)  # (b, 1, d)x(b, b, d) = (b, b). (b,1) broadcasted to (b,b)

    cos_sim = torch.concat([pos_cos_sim, multi_pos_cos_sim], dim=1)  # (b,2b)

    positive_pos = torch.concat([torch.eye(cos_sim.size(0)), torch.ones(multi_pos_cos_sim.shape)], dim=1).to(cls.device)
    loss = (-(positive_pos * cos_sim).sum(dim=-1) + torch.logsumexp(cos_sim, dim=-1)).mean()

    return cos_sim, loss


def supercom_intensive_test_loss(cls, tup):
    z1, z2, _ = tup  # b,d
    origin_cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    # print(cls.k)
    if cls.k != 0:
        if cls.difficulty == "easy":
            # print("easy")
            noise = torch.normal(mean=0, std=1, size=(cls.k * z1.size(0), z1.size(1))).to(cls.device)
        elif cls.difficulty == "normal":
            # print("normal")
            noise_list = []
            z_mean = z1.mean(0)
            z_std = z1.std(0)
            for _ in range(z1.size(0) * cls.k):
                noise_list.append(torch.normal(mean=z_mean, std=z_std))
            noise = torch.stack(noise_list)
        elif cls.difficulty == "hard":
            # print("hard")
            noise_list = []
            zero = torch.zeros(z1.size(1)).to(cls.device)
            z_std = z1.std(0)  # d
            for _ in range(cls.k):
                for sample in z1:
                    noise_list.append(sample + torch.normal(mean=zero, std=z_std))
            noise = torch.stack(noise_list).to(cls.device)

        neg_cos_sim = cls.sim(z1.unsqueeze(1), noise.unsqueeze(0))  # (b,1,d)x(1,kb,d)=(b,kb)
        cos_sim = torch.cat([origin_cos_sim, neg_cos_sim], dim=1)  # (b,kb+1)
    else:
        cos_sim = origin_cos_sim

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)  # b
    loss_fct = torch.nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)  #
    return cos_sim, loss


'''
def mixup_loss(cls,tup):
    print("noisy positive")
    z1, z2, _ = tup  # b,d

    z2 = z1 + torch.normal(mean=0, std=1, size=z1.shape)
    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    loss_fct = torch.nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)  #
    return cos_sim, loss
'''


def get_cl_loss(loss_name=None):
    if not loss_name:
        return None
    try:
        modules = sys.modules[__name__]
        loss = getattr(modules, loss_name)
    except AttributeError as e:
        # print(e)
        print("------------current implemented loss list------------------")
        loss_list = [m for m in dir(modules) if callable(getattr(modules, m))]
        print("------------------------------------")
        print(loss_list)
        print(f"{loss_name} not implemented yet\n")
        sys.exit()
    return loss


'''
def get_cl_loss(loss_name=None):
    if not loss_name:
        return None
    if loss_name=="align_uniform_loss":
        return align_uniform_loss
    elif loss_name=="original_contrastive_loss":
        return original_contrastive_loss
    elif loss_name=="negative_only_loss":
        return negative_only_loss
    elif loss_name=="really_dummy_positive_loss":
        return really_dummy_positive_loss
    else:
        raise ValueError("not implemented yet")

'''
