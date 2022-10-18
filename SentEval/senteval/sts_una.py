"""
Implement Alignment and Uniformity following https://github.com/princeton-nlp/SimCSE/issues/41
"""

from __future__ import absolute_import, division, unicode_literals

import os
import io
import numpy as np
import logging
import torch
import torch.nn.functional as F

from scipy.stats import spearmanr, pearsonr

from senteval.utils import cosine

class STSUniformityAndAlignment(object):
    def loadFile(self, fpath):
        self.data = {}
        self.samples = []

        for dataset in self.datasets:
            sent1, sent2 = zip(*[l.split("\t") for l in
                               io.open(fpath + '/STS.input.%s.txt' % dataset,
                                       encoding='utf8').read().splitlines()])
            raw_scores = np.array([x for x in
                                   io.open(fpath + '/STS.gs.%s.txt' % dataset,
                                           encoding='utf8')
                                   .read().splitlines()])
            not_empty_idx = raw_scores != ''

            gs_scores = [float(x) for x in raw_scores[not_empty_idx]]
            sent1 = np.array([s.split() for s in sent1])[not_empty_idx] ## list of str tokens
            sent2 = np.array([s.split() for s in sent2])[not_empty_idx]
            # sort data by length to minimize padding in batcher
            sorted_data = sorted(zip(sent1, sent2, gs_scores),
                                 key=lambda z: (len(z[0]), len(z[1]), z[2]))
            sent1, sent2, gs_scores = map(list, zip(*sorted_data))

            self.data[dataset] = (sent1, sent2, gs_scores)
            self.samples += sent1 + sent2

    def do_prepare(self, params, prepare):
        if 'similarity' in params:
            self.similarity = params.similarity
        else:  # Default similarity is cosine
            self.similarity = lambda s1, s2: np.nan_to_num(cosine(np.nan_to_num(s1), np.nan_to_num(s2)))
        return prepare(params, self.samples)

    def run(self, params, batcher, task_name):
        results = {}
        all_sys_scores = []
        all_gs_scores = []
        ################# newly added
        all_enc1 = []
        all_enc2 = []
        all_enc1_pos = []
        all_enc2_pos = []
        #################
        for dataset in self.datasets:
            sys_scores = []
            input1, input2, gs_scores = self.data[dataset] ## len(input1)=750
            for ii in range(0, len(gs_scores), params.batch_size):
                batch1 = input1[ii:ii + params.batch_size] ## len(batch1) = token length(?)
                batch2 = input2[ii:ii + params.batch_size]
                batch_gs_scores = gs_scores[ii:ii + params.batch_size]  # newly added

                # we assume get_batch already throws out the faulty ones
                if len(batch1) == len(batch2) and len(batch1) > 0:
                    enc1 = batcher(params, batch1) ## enc1.shape = [token length(?), dim]
                    enc2 = batcher(params, batch2)
                    ################# newly added
                    all_enc1.append(enc1)
                    all_enc2.append(enc2)

                    ## Pos indices
                    pos_indices = [i for i in range(len(batch_gs_scores)) if batch_gs_scores[i] >= 4.0] ## select positive (scores=4 or 5) samples in the batch
                    if len(pos_indices) == 0: ## There might be no pos indices in some batches
                        pass
                    else:
                        enc1_pos = enc1[pos_indices] ## shape=[# pos samples, dim]
                        enc2_pos = enc2[pos_indices]

                        all_enc1_pos.append(enc1_pos)
                        all_enc2_pos.append(enc2_pos)
                    #################

                    for kk in range(enc2.shape[0]):
                        sys_score = self.similarity(enc1[kk], enc2[kk])
                        sys_scores.append(sys_score)
            
            all_sys_scores.extend(sys_scores)
            all_gs_scores.extend(gs_scores)
            
            ## Uniformity and Alignment
            all_enc1_ts = torch.cat(all_enc1, dim=0)
            all_enc2_ts = torch.cat(all_enc2, dim=0)
            norm_all_enc1_ts = F.normalize(all_enc1_ts)
            norm_all_enc2_ts = F.normalize(all_enc2_ts)

            all_enc1_pos_ts = torch.cat(all_enc1_pos, dim=0)
            all_enc2_pos_ts = torch.cat(all_enc2_pos, dim=0)
            norm_all_enc1_pos_ts = F.normalize(all_enc1_pos_ts)
            norm_all_enc2_pos_ts = F.normalize(all_enc2_pos_ts)

            all_loss_align = align_loss(norm_all_enc1_pos_ts, norm_all_enc2_pos_ts)
            all_loss_uniform = uniform_loss(torch.cat((norm_all_enc1_ts, norm_all_enc2_ts), dim=0))
            W = torch.cat((all_enc1_ts, all_enc2_ts), dim=0)
            all_IW_scores = measure_I_W(W)
            all_avgcos_scores = measure_avg_cos(W)
            all_disentanglement_scores = measure_disentanglement(W)
            if params.output_reps:
                print("=== saving output reps ===")
                torch.save(W.data, params.save_path + "/" + task_name + "_" + dataset + "_reps.pt")

            results[dataset] = {'pearson': pearsonr(sys_scores, gs_scores), ## len(sys_scores) = 750
                                'spearman': spearmanr(sys_scores, gs_scores),
                                'nsamples': len(sys_scores),
                                'align_loss': all_loss_align,  # newly added
                                'uniform_loss': all_loss_uniform,  # newly added
                                'IW': all_IW_scores,  # newly added
                                'avgcos': all_avgcos_scores,  # newly added
                                'disentanglement': all_disentanglement_scores,  # newly added
                                }
            logging.debug('%s : pearson = %.4f, spearman = %.4f, align_loss = %.4f, uniform_loss = %.4f, IW = %.4f, avgcos = %.4f, disentanglement = %.4f' %
                        (dataset, results[dataset]['pearson'][0], results[dataset]['spearman'][0],
                        results[dataset]['align_loss'], results[dataset]['uniform_loss'],
                        results[dataset]['IW'], results[dataset]['avgcos'], results[dataset]['disentanglement'])
                        )
        
        weights = [results[dset]['nsamples'] for dset in results.keys()]
        list_prs = np.array([results[dset]['pearson'][0] for dset in results.keys()])
        list_spr = np.array([results[dset]['spearman'][0] for dset in results.keys()])
        list_align = np.array([results[dset]['align_loss'] for dset in results.keys()])
        list_uniform = np.array([results[dset]['uniform_loss'] for dset in results.keys()])
        list_IW = np.array([results[dset]['IW'] for dset in results.keys()])
        list_avgcos = np.array([results[dset]['avgcos'] for dset in results.keys()])
        list_disentanglement = np.array([results[dset]['disentanglement'] for dset in results.keys()])

        avg_pearson = np.average(list_prs)
        avg_spearman = np.average(list_spr)
        avg_align = np.average(list_align)
        avg_uniform = np.average(list_uniform)
        avg_IW = np.average(list_IW)
        avg_avgcos = np.average(list_avgcos)
        avg_disentanglement = np.average(list_disentanglement)

        wavg_pearson = np.average(list_prs, weights=weights)
        wavg_spearman = np.average(list_spr, weights=weights)
        wavg_align = np.average(list_align, weights=weights)
        wavg_uniform = np.average(list_uniform, weights=weights)
        wavg_IW = np.average(list_IW, weights=weights)
        wavg_avgcos = np.average(list_avgcos, weights=weights)
        wavg_disentanglement = np.average(list_disentanglement, weights=weights)

        all_pearson = pearsonr(all_sys_scores, all_gs_scores)
        all_spearman = spearmanr(all_sys_scores, all_gs_scores)
        results['all'] = {'pearson': {'all': all_pearson[0],
                                      'mean': avg_pearson,
                                      'wmean': wavg_pearson},
                          'spearman': {'all': all_spearman[0],
                                       'mean': avg_spearman,
                                       'wmean': wavg_spearman},
                          'align_loss': {'mean': avg_align,
                                    'wmean': wavg_align},
                          'uniform_loss': {'mean': avg_uniform,
                                    'wmean': wavg_uniform},
                          'IW': {'mean': avg_IW,
                                'wmean': wavg_IW},
                          'avgcos': {'mean': avg_avgcos,
                                    'wmean': wavg_avgcos},
                          'disentanglement': {'mean': avg_disentanglement,
                                    'wmean': wavg_disentanglement},
                                       }
        logging.debug('ALL : Pearson = %.4f, \
            Spearman = %.4f' % (all_pearson[0], all_spearman[0]))
        logging.debug('ALL (weighted average) : Pearson = %.4f, \
            Spearman = %.4f' % (wavg_pearson, wavg_spearman))
        logging.debug('ALL (average) : Pearson = %.4f, \
            Spearman = %.4f\n' % (avg_pearson, avg_spearman))

        return results
        
def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()

def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

## I(W)
def measure_I_W(W):
    assert(len(W.shape) == 2)
    ## W: embedding matrix, shape=[n_data, dim]
    u, s, v = torch.svd(W) ## v.shape=[dim, n_data]
#     print(v.shape)
    z = torch.exp(torch.matmul(W, v)).sum(0) ## (W*v).shape=[n_data, n_data]
#     print(z.min(), z.max())
    i_w = z.min() / z.max()
    return float(i_w)

def measure_avg_cos(W):
    assert(len(W.shape) == 2)
    ## W: embedding matrix, shape=[n_data, dim]
    n_data = W.shape[0]
    W_norm = torch.norm(W, dim=1)
    W /= W_norm.unsqueeze(1)
    WW = torch.matmul(W, W.T) ## WW.shape[n_data, n_data]
    triu_numel = n_data * (n_data-1) / 2
    cos = torch.sum(torch.triu(WW, diagonal=1)) / triu_numel
    return float(cos)

def measure_disentanglement(W):
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

class UNASTS12Eval(STSUniformityAndAlignment):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS12 *****\n\n')
        self.seed = seed
        self.datasets = ['MSRpar', 'MSRvid', 'SMTeuroparl',
                         'surprise.OnWN', 'surprise.SMTnews']
        self.loadFile(taskpath)


class UNASTS13Eval(STSUniformityAndAlignment):
    # STS13 here does not contain the "SMT" subtask due to LICENSE issue
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS13 (-SMT) *****\n\n')
        self.seed = seed
        self.datasets = ['FNWN', 'headlines', 'OnWN']
        self.loadFile(taskpath)


class UNASTS14Eval(STSUniformityAndAlignment):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS14 *****\n\n')
        self.seed = seed
        self.datasets = ['deft-forum', 'deft-news', 'headlines',
                         'images', 'OnWN', 'tweet-news']
        self.loadFile(taskpath)


class UNASTS15Eval(STSUniformityAndAlignment):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS15 *****\n\n')
        self.seed = seed
        self.datasets = ['answers-forums', 'answers-students',
                         'belief', 'headlines', 'images']
        self.loadFile(taskpath)


class UNASTS16Eval(STSUniformityAndAlignment):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS16 *****\n\n')
        self.seed = seed
        self.datasets = ['answer-answer', 'headlines', 'plagiarism',
                         'postediting', 'question-question']
        self.loadFile(taskpath)


class UNASTSBenchmarkEval(STSUniformityAndAlignment):
    def __init__(self, task_path, seed=1111):
        logging.debug('\n\n***** Transfer task : STSBenchmark*****\n\n')
        self.seed = seed
        self.samples = []
        train = self.loadFile(os.path.join(task_path, 'sts-train.csv'))
        dev = self.loadFile(os.path.join(task_path, 'sts-dev.csv'))
        test = self.loadFile(os.path.join(task_path, 'sts-test.csv'))
        self.datasets = ['train', 'dev', 'test']
        self.data = {'train': train, 'dev': dev, 'test': test}

    def loadFile(self, fpath):
        sick_data = {'X_A': [], 'X_B': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip().split('\t')
                sick_data['X_A'].append(text[5].split())
                sick_data['X_B'].append(text[6].split())
                sick_data['y'].append(text[4])

        sick_data['y'] = [float(s) for s in sick_data['y']]
        self.samples += sick_data['X_A'] + sick_data["X_B"]
        return (sick_data['X_A'], sick_data["X_B"], sick_data['y'])
        
class UNASICKRelatednessEval(STSUniformityAndAlignment):
    def __init__(self, task_path, seed=1111):
        logging.debug('\n\n***** Transfer task : SICKRelatedness*****\n\n')
        self.seed = seed
        self.samples = []
        train = self.loadFile(os.path.join(task_path, 'SICK_train.txt'))
        dev = self.loadFile(os.path.join(task_path, 'SICK_trial.txt'))
        test = self.loadFile(os.path.join(task_path, 'SICK_test_annotated.txt'))
        self.datasets = ['train', 'dev', 'test']
        self.data = {'train': train, 'dev': dev, 'test': test}
    
    def loadFile(self, fpath):
        skipFirstLine = True
        sick_data = {'X_A': [], 'X_B': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                if skipFirstLine:
                    skipFirstLine = False
                else:
                    text = line.strip().split('\t')
                    sick_data['X_A'].append(text[1].split())
                    sick_data['X_B'].append(text[2].split())
                    sick_data['y'].append(text[3])

        sick_data['y'] = [float(s) for s in sick_data['y']]
        self.samples += sick_data['X_A'] + sick_data["X_B"]
        return (sick_data['X_A'], sick_data["X_B"], sick_data['y'])
