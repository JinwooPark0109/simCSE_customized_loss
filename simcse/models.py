import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import transformers
from transformers import RobertaTokenizer
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead,BertEncoder, BertPooler
from transformers.activations import gelu
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions

#from simcse.cl_loss import get_cl_loss, bss_loss
#from simcse.cl_loss import bss_loss
#from simcse.log_cl_loss import get_cl_loss
from simcse.loss_graduate import get_cl_loss

from sklearn.mixture import GaussianMixture
import numpy as np

from typing import Optional


class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError

class VAE(torch.nn.Module):
    def __init__(self, dim_0, dim_1, dim_2, device):
        super().__init__()
        self.enc1=torch.nn.Linear(dim_0, dim_1)
        self.enc2=torch.nn.Linear(dim_1, dim_2)
        self.to_z_mean=torch.nn.Linear(dim_2, dim_2)
        self.to_z_var=torch.nn.Linear(dim_2, dim_2)
        self.dec1=torch.nn.Linear(dim_2, dim_1)
        self.dec2=torch.nn.Linear(dim_1, dim_0)
        self.act=torch.nn.functional.relu
        self.dropout=torch.nn.functional.dropout
        self.dim_2=dim_2
        self.device=device
    def encode(self,x):
        #return self.act(self.enc2(self.act(self.enc1(x))))
        return self.dropout(self.act(self.enc2(self.dropout(self.act(self.enc1(x))))))
    def decode(self,z):
        #return self.act(self.dec2(self.act(self.dec1(z))))
        return self.dec2(self.dropout(self.act(self.dec1(z))))
    def reparameterize(self, x):
        z_mean = self.to_z_mean(x)
        z_var = self.to_z_var(x)
        noise = torch.randn(x.shape).to(x.device)
        return noise * z_var + z_mean, z_mean, z_var
    def forward(self, x):
        enc_out=self.encode(x)
        z, z_mean, z_var=self.reparameterize(enc_out)
        dec_out=self.decode(z)
        return dec_out, z_mean, z_var
    def generate(self, n):
        noise=torch.randn((n,self.dim_2)).to(self.device)
        return self.decode(noise)

class VAE_runner:
    '''
    https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py#L124
    '''
    def __init__(self,device):
        self.model=VAE(768, 400, 100, device).to(device)
        self.opt=torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.epoch=30

        self.model.load_state_dict(torch.load("/home/jwpark/SimCSE/vae/vae_state_dict.pt"))
        print("vae loaded")
    def loss(self, x, x_hat, z_mean, z_var):
        recon_loss=torch.nn.functional.mse_loss(x_hat, x)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + z_var - z_mean ** 2 - z_var.exp(), dim = 1), dim = 0)
        return recon_loss+kld_loss
    def train(self,x):
        for _ in range(self.epoch):
            x_hat, z_mean, z_var=self.model(x)
            loss=self.loss(x, x_hat, z_mean, z_var)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
    def generate(self, n):
        return self.model.generate(n)

def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)
    if cls.model_args.pooler_type == "cls":
        cls.mlp = MLPLayer(config)
    cls.sim = Similarity(temp=cls.model_args.temp)
    cls.init_weights()

def cl_forward(cls,
    encoder,
    cl_loss,
    use_bss_loss,
    get_vae,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    mlm_input_ids=None,
    mlm_labels=None
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    ori_input_ids = input_ids
    batch_size = input_ids.size(0)
    # Number of sentences in one instance
    # 2: pair instance; 3: pair instance with a hard negative
    num_sent = input_ids.size(1)

    mlm_outputs = None
    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
    attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)

    kwargs={"input_ids":input_ids,
            "attention_mask":attention_mask,
            "token_type_ids":token_type_ids,
            "position_ids":position_ids,
            "head_mask":head_mask,
            "inputs_embeds":inputs_embeds,
            "output_attentions":output_attentions,
            "output_hidden_states": True if cls.model_args.pooler_type in['avg_top2', 'avg_first_last'] else False,
            "return_dict":True,
            "batch_size":batch_size,
            "num_sent":num_sent,
            "pooler_type":cls.pooler_type
            }

    #---------
    if cls.awp is not None:
        cls.awp.perturb(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
            batch_size=batch_size,
            num_sent=num_sent,
            pooler_type=cls.pooler_type
        )
    #---------
    if cls.sam is not None:
        cls.sam.sam_forward_backward(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
            batch_size=batch_size,
            num_sent=num_sent,
            pooler_type=cls.pooler_type
        )
        cls.sam.sam_step() #no grad
    #---------

    if cls.add_noise or cls.mul_noise or cls.mul_bern or cls.mul_feat_bern:
        noise_size=(input_ids.size(0), input_ids.size(1), cls.embedding_dim)
        noise=None
        if cls.add_noise:
            noise = torch.normal(size=noise_size, mean=0, std=cls.noise_std).to(cls.device)
        elif cls.mul_noise:
            noise = torch.normal(size=noise_size, mean=1, std=cls.noise_std).to(cls.device)
        elif cls.mul_bern:
            noise = torch.bernoulli(torch.full(noise_size, cls.bern_ratio)).to(cls.device)
        elif cls.mul_feat_bern:
            # (b,d) -broad cast to-> (b,l,d)
            noise = torch.bernoulli(torch.full((noise_size[0], noise_size[2]), cls.bern_ratio)).unsqueeze(1).expand((-1, noise_size[1], -1)).to(cls.device)
        cls.bert.set_noise(noise)

    #---------
    if cls.fgsm is not None:
        cls.fgsm.perturb(**kwargs)
    #----------

    # Get raw embeddings
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )
    #if cls.add_noise or cls.mul_noise or cls.mul_bern or cls.mul_feat_bern:
    if cls.bert.has_noise():# for prevent noise injection at sent_forward.
        cls.bert.reset_noise()

    # MLM auxiliary objective
    if mlm_input_ids is not None:
        mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1)))
        mlm_outputs = encoder(
            mlm_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
        )


    if use_bss_loss:
        my_loss=bss_loss(cls, outputs, attention_mask.view(batch_size, num_sent, -1))

    # Pooling
    pooler_output = cls.pooler(attention_mask, outputs)
    pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1))) # (bs, num_sent, hidden)

    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    if cls.pooler_type == "cls":
        pooler_output = cls.mlp(pooler_output)

    # Separate representation
    z1, z2 = pooler_output[:,0], pooler_output[:,1]

    # Hard negative
    z3=None # inited as None for custom loss. used instead of num_sent
    if num_sent == 3:
        z3 = pooler_output[:, 2]

    # Gather all embeddings if using distributed training
    if dist.is_initialized() and cls.training:
        # Gather hard negative
        if num_sent >= 3:
            z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
            z3_list[dist.get_rank()] = z3
            z3 = torch.cat(z3_list, 0)

        # Dummy vectors for allgather
        z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
        z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
        # Allgather
        dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
        dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

        # Since allgather results do not have gradients, we replace the
        # current process's corresponding embeddings with original tensors
        z1_list[dist.get_rank()] = z1
        z2_list[dist.get_rank()] = z2
        # Get full batch embeddings: (bs x N, hidden)
        z1 = torch.cat(z1_list, 0)
        z2 = torch.cat(z2_list, 0)

    #-------------------------------------------------------------

    cos_sim, loss = cl_loss(cls, (z1, z2, z3))
    #-------------------------------------------------------------

    # Calculate loss for MLM
    if mlm_outputs is not None and mlm_labels is not None:
        #print("do mlm")
        MLM_loss_fct = nn.CrossEntropyLoss()
        mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
        prediction_scores = cls.lm_head(mlm_outputs.last_hidden_state)
        masked_lm_loss = MLM_loss_fct(prediction_scores.view(-1, cls.config.vocab_size), mlm_labels.view(-1))
        loss = loss + cls.model_args.mlm_weight * masked_lm_loss


    if use_bss_loss:
        loss= loss + 0.001*my_loss
    if get_vae:
        loss= loss*0

    if not return_dict:
        output = (cos_sim,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
    return SequenceClassifierOutput(
        loss=loss,
        logits=cos_sim,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def sentemb_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    pooler_output = cls.pooler(attention_mask, outputs)
    if cls.pooler_type == "cls" and not cls.model_args.mlp_only_train:
        pooler_output = cls.mlp(pooler_output)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )


class BertForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]

        #self.bert = BertModel(config, add_pooling_layer=False)
        self.bert = BertModel_with_noise(config, add_pooling_layer=False)

        self.cl_loss=get_cl_loss(self.model_args.cl_loss)
        self.use_bss_loss=self.model_args.use_bss_loss
        self.k = self.model_args.k
        self.difficulty = self.model_args.difficulty
        self.coef=self.model_args.coef
        self.lower_bound=self.model_args.lower_bound

        self.gmm_comp=self.model_args.gmm_comp
        self.encoding_buffer_size = self.model_args.encoding_buffer_size
        self.encoding_buffer=[]
        self.prev_buffer=[]
        self.prev_buffer_size = self.model_args.encoding_buffer_size

        self.gmm = None
        self.vae_runner= None

        self.get_vae=self.model_args.get_vae

        self.update_step=self.model_args.update_step
        self.step_cnt=0

        self.pos_sim=[]
        self.origin_neg_sim=[]
        self.generated_neg_sim=[]
        self.orig_gen_neg_sim=[]

        if self.model_args.do_mlm:
            self.lm_head = BertLMPredictionHead(config)

        self.awp=None
        self.sam=None
        self.fgsm=None

        self.embedding_dim=config.hidden_size
        self.add_noise=self.model_args.add_noise
        if self.add_noise: self.bert.set_apply_noise("add")
        self.mul_noise = self.model_args.mul_noise
        self.mul_bern = self.model_args.mul_bern
        self.mul_feat_bern = self.model_args.mul_feat_bern
        if self.mul_noise or self.mul_bern or self.mul_feat_bern: self.bert.set_apply_noise("mul")
        self.noise_std = self.model_args.noise_std
        self.bern_ratio = self.model_args.bern_ratio

        cl_init(self, config)

    def set_awp(self, awp):
        self.awp=awp

    def set_sam(self, sam):
        self.sam = sam

    def set_fgsm(self, fgsm):
        self.fgsm = fgsm

    def update_encoding_buffer(self, t):
        self.encoding_buffer.append(t.cpu().detach().numpy())
        if len(self.encoding_buffer)>self.encoding_buffer_size:
            self.encoding_buffer.pop(0)

    def replace_gmm(self, t):
        self.update_encoding_buffer(t)
        self.gmm=GaussianMixture(n_components=self.gmm_comp).fit(np.concatenate(self.encoding_buffer, axis=0))

    def update_gmm(self, t):
        self.update_encoding_buffer(t)
        if self.gmm:
            self.gmm.fit(np.concatenate(self.encoding_buffer, axis=0))
        else:
            self.gmm=GaussianMixture(n_components=self.gmm_comp).fit(np.concatenate(self.encoding_buffer, axis=0))
    def gmm_sampling(self, size):
        return torch.tensor(self.gmm.sample(size)[0]).to(self.device)
    def periodic_update_gmm_sampler(self, t, size):
        self.encoding_buffer.append(t.cpu().detach().numpy())
        if self.step_cnt%self.update_step==0: # for 0, 100, 200, ...
            print("update gmm!!")
            self.gmm = GaussianMixture(n_components=self.gmm_comp).fit(np.concatenate(self.encoding_buffer, axis=0))
            self.encoding_buffer = []
        self.step_cnt += 1
        return torch.tensor(self.gmm.sample(size)[0]).to(self.device)
    def reset_step_cnt(self):
        self.step_cnt=0

    def get_buffer_mean_and_std(self, t):
        self.update_encoding_buffer(t)
        n=np.concatenate(self.encoding_buffer, axis=0)
        return torch.tensor(n.mean(0)), torch.tensor(n.std(0))

    def get_prev_buffer(self, t):
        ret = self.prev_buffer[:self.prev_buffer_size]
        self.prev_buffer.append(t.cpu().detach().numpy())
        if len(self.prev_buffer) > self.prev_buffer_size:
            self.prev_buffer.pop(0)
        return ret

    def update_vae(self, t):
        if not self.vae_runner:
            self.vae_runner=VAE_runner(t.device)
        self.vae_runner.train(t)

    def vae_sampling(self, size):
        return self.vae_runner.generate(size)

    def get_vae_param(self):
        return self.vae_runner.model.state_dict()
    '''
    def log_sim(self, simcse_sim, generated_neg_sim):
        self.pos_sim.append(simcse_sim.diagonal().mean().cpu().item())
        b=simcse_sim.size(0)
        mask = 1 - torch.eye(b)
        self.origin_neg_sim.append(((simcse_sim.cpu() * mask).sum() / (b * (b - 1))).item())
        self.generated_neg_sim.append(generated_neg_sim.mean().cpu().item())
    '''
    def log_sim(self, pos_sim=None, origin_neg_sim=None, generated_neg_sim=None):
        if pos_sim is not None:
            self.pos_sim.append(pos_sim.nanmean().cpu().item())
        if origin_neg_sim is not None:
            self.origin_neg_sim.append(origin_neg_sim.nanmean().cpu().item())
        if generated_neg_sim is not None:
            self.generated_neg_sim.append(generated_neg_sim.nanmean().cpu().item())

        #print("---")
        #print(self.pos_sim)
        #print(self.origin_neg_sim)
        #print(self.generated_neg_sim)

    def get_log_sim(self):
        return self.pos_sim, self.origin_neg_sim, self.generated_neg_sim

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.bert, self.cl_loss, self.use_bss_loss, self.get_vae,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )



class RobertaForCL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.roberta = RobertaModel(config, add_pooling_layer=False)

        self.cl_loss=get_cl_loss(self.model_args.cl_loss)

        if self.model_args.do_mlm:
            self.lm_head = RobertaLMHead(config)

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.roberta, self.cl_loss,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )


class BertModel_with_noise(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings_with_Noise(config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.init_weights()

    def set_apply_noise(self, apply_noise):
        self.embeddings.set_apply_noise(apply_noise)

    def set_noise(self, noise):
        self.embeddings.set_noise(noise)

    def reset_noise(self):
        self.embeddings.reset_noise()

    def has_noise(self):
        return self.embeddings.noise is not None

    def get_current_embed(self):
        return self.embeddings.get_current_embed()

    def reset_current_embed(self):
        self.embeddings.reset_current_embed()

class BertEmbeddings_with_Noise(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )
        self.noise=None
        self.apply_noise=None
        self.current_inputs_embeds=None
    #--------------------------
    def set_apply_noise(self, apply_noise):
        self.apply_noise = apply_noise

    def set_noise(self, noise):
        self.noise=noise

    def reset_noise(self):
        self.noise=None

    def get_current_embed(self):
        return self.current_inputs_embeds

    def reset_current_embed(self):
        self.current_inputs_embeds = None
    #--------------------------

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        #-------------------------------
        self.current_inputs_embeds=inputs_embeds
        if self.current_inputs_embeds.requires_grad: # in case of training
            self.current_inputs_embeds.retain_grad()

        if self.noise is not None:
            if self.apply_noise=="add":
                inputs_embeds += self.noise
            elif self.apply_noise=="mul":
                inputs_embeds *= self.noise
        #------------------------------

        embeddings = inputs_embeds + token_type_embeddings

        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings