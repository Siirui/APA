import math
from collections import defaultdict

import sys
sys.path.append("/hanpengfei/PEER_Benchmark/script/")
from augmentations import random_augment

from captum.attr import IntegratedGradients, LayerIntegratedGradients
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from torchviz import make_dot


import torch
from torch import nn
from torch.nn import functional as F

from torchdrug import core, layers, tasks, metrics, utils, data
from torchdrug.core import Registry as R
from torchdrug.layers import functional

import random
import seaborn as sns
import copy
residue_symbol2id = {"G": 0, "A": 1, "S": 2, "P": 3, "V": 4, "T": 5, "C": 6, "I": 7, "L": 8, "N": 9,
                         "D": 10, "Q": 11, "K": 12, "E": 13, "M": 14, "H": 15, "F": 16, "R": 17, "Y": 18, "W": 19}
id2residue_symbol = {v: k for k, v in residue_symbol2id.items()}
all_possible_ids = list(residue_symbol2id.values())

def augmentation_batch(batch, n_aug=1, sub_policy=None, aug_list=None, aug_num=None):
    if "graph" in batch:
        graph = batch["graph"]
        device = graph.device
        sequences = graph.to_sequence()
        sequences = [seq.replace(".", "") for seq in sequences]
        sequence_list = [list(seq) for seq in sequences]
        for _ in range(n_aug):
            for seq in sequence_list:
                tmp_seq = copy.deepcopy(seq)
                tmp_seq = random_augment(tmp_seq, sub_policy=sub_policy, aug_list=aug_list, aug_num=aug_num)
                sequences.append("".join(tmp_seq))
        graph = data.PackedProtein.from_sequences(sequences, atom_feature=None, bond_feature=None)
        graph = graph.to(device)
        aug_batch = copy.deepcopy(batch)
        aug_batch["graph"] = graph
        return aug_batch
    
    elif "graph1" in batch and "graph2" in batch:
        graph1 = batch["graph1"]
        graph2 = batch["graph2"]
        device = graph1.device
        sequences1 = graph1.to_sequence()
        sequences2 = graph2.to_sequence()
        sequences1 = [seq.replace(".", "") for seq in sequences1]
        sequences2 = [seq.replace(".", "") for seq in sequences2]
        sequence_list1 = [list(seq) for seq in sequences1]
        sequence_list2 = [list(seq) for seq in sequences2]
        for _ in range(n_aug):
            for seq in sequence_list1:
                tmp_seq = copy.deepcopy(seq)
                tmp_seq = random_augment(tmp_seq, sub_policy=sub_policy, aug_list=aug_list, aug_num=aug_num)
                sequences1.append("".join(tmp_seq))
            for seq in sequence_list2:
                tmp_seq = copy.deepcopy(seq)
                tmp_seq = random_augment(tmp_seq, sub_policy=sub_policy, aug_list=aug_list, aug_num=aug_num)
                sequences2.append("".join(tmp_seq))
        graph1 = data.PackedProtein.from_sequences(sequences1, atom_feature=None, bond_feature=None)
        graph2 = data.PackedProtein.from_sequences(sequences2, atom_feature=None, bond_feature=None)
        graph1 = graph1.to(device)
        graph2 = graph2.to(device)
        aug_batch = copy.deepcopy(batch)
        aug_batch["graph1"] = graph1
        aug_batch["graph2"] = graph2
        return aug_batch

def augmentation_batch_ig(batch, n_aug=1, sub_policy=None, attributions=None, attributions2=None, aug_list=None, aug_num=None):
    amino_acid_contributions = attributions.sum(dim=-1)
    amino_acid_contributions = amino_acid_contributions[:,1:-1]
    thresholds = 0.7 * torch.max(amino_acid_contributions, dim=1).values
    # 生成一个list，每个蛋白质的最大贡献度的80%值的index
    above_threshold_indices = amino_acid_contributions > thresholds[:, None]
    above_threshold_indices = ~above_threshold_indices 
    
    if attributions2 is not None:
        amino_acid_contributions2 = attributions2.sum(dim=-1)
        amino_acid_contributions2 = amino_acid_contributions2[:,1:-1]
        thresholds2 = 0.7 * torch.max(amino_acid_contributions2, dim=1).values
        above_threshold_indices2 = amino_acid_contributions2 > thresholds2[:, None]
        above_threshold_indices2 = ~above_threshold_indices2

    graph = batch["graph"]
    device = graph.device
    sequences = graph.to_sequence()
    sequences = [seq.replace(".", "") for seq in sequences]
    sequence_list = [list(seq) for seq in sequences]
    for _ in range(n_aug):
        for seq_index, seq in enumerate(sequence_list):
    
            tmp_seq = copy.deepcopy(seq)
            above_threshold_indices[seq_index][len(tmp_seq):] = False
            replace_indices = torch.where(above_threshold_indices[seq_index])[0].tolist()
            tmp_seq = random_augment(tmp_seq, sub_policy=sub_policy, aug_list=aug_list, indices=replace_indices, aug_num=aug_num)
            sequences.append("".join(tmp_seq))
    
    graph = data.PackedProtein.from_sequences(sequences, atom_feature=None, bond_feature=None)
    graph = graph.to(device)
    aug_batch = copy.deepcopy(batch)
    aug_batch["graph"] = graph
    return aug_batch

@R.register("tasks.PropertyPrediction")
class PropertyPrediction(tasks.Task, core.Configurable):
    """
    Graph / molecule / protein property prediction task.

    This class is also compatible with semi-supervised learning.

    Parameters:
        model (nn.Module): graph representation model
        task (str, list or dict, optional): training task(s).
            For dict, the keys are tasks and the values are the corresponding weights.
        criterion (str, list or dict, optional): training criterion(s). For dict, the keys are criterions and the values
            are the corresponding weights. Available criterions are ``mse``, ``bce`` and ``ce``.
        metric (str or list of str, optional): metric(s).
            Available metrics are ``mae``, ``rmse``, ``auprc`` and ``auroc``.
        num_mlp_layer (int, optional): number of layers in mlp prediction head
        normalization (bool, optional): whether to normalize the target
        num_class (int, optional): number of classes
        mlp_batch_norm (bool, optional): apply batch normalization in mlp or not
        mlp_dropout (float, optional): dropout in mlp
        graph_construction_model (nn.Module, optional): graph construction model
        verbose (int, optional): output verbose level
        batchnorm (bool): whether to apply batch normalization before MLP
        aug (str): which augmentation to use
        n_aug (int): number of augmented samples
        aug_list (list, optional): list of augmentation functions
        ig (bool): whether to use integrated gradients
        n_step (int): number of steps for integrated gradients
        aug_num (int): number of augmentation implemented
    """

    eps = 1e-10
    _option_members = {"task", "criterion", "metric"}

    def __init__(self, model, task=(), criterion="mse", metric=("mae", "rmse"), num_mlp_layer=1,
                 normalization=True, num_class=None, mlp_batch_norm=False, mlp_dropout=0,
                 graph_construction_model=None, verbose=0, batchnorm=False, aug=None, n_aug=1, aug_list=None, ig=None, n_step=5, aug_num=None):
        super(PropertyPrediction, self).__init__()
        self.model = model
        self.task = task
        self.criterion = criterion
        self.metric = metric
        self.num_mlp_layer = num_mlp_layer
        # For classification tasks, we disable normalization tricks.
        self.normalization = normalization and ("ce" not in criterion) and ("bce" not in criterion)
        self.num_class = (num_class,) if isinstance(num_class, int) else num_class
        self.mlp_batch_norm = mlp_batch_norm
        self.mlp_dropout = mlp_dropout
        self.graph_construction_model = graph_construction_model
        self.verbose = verbose
        self.batchnorm = batchnorm
        self.aug = aug
        self.n_aug = n_aug
        self.aug_list = aug_list
        if self.batchnorm == True:
            self.bn = nn.BatchNorm1d(self.model.output_dim)
        self.ig = ig
        self.n_step = n_step
        self.aug_num = aug_num

    def preprocess(self, train_set, valid_set, test_set):
        """
        Compute the mean and derivation for each task on the training set.
        """
        values = defaultdict(list)
        for sample in train_set:
            if not sample.get("labeled", True):
                continue
            for task in self.task:
                if not math.isnan(sample[task]):
                    values[task].append(sample[task])
        mean = []
        std = []
        weight = []
        num_class = []
        for task, w in self.task.items():
            value = torch.tensor(values[task])
            mean.append(value.float().mean())
            std.append(value.float().std())
            weight.append(w)
            if value.ndim > 1:
                num_class.append(value.shape[1])
            elif value.dtype == torch.long:
                task_class = value.max().item()
                if task_class == 1 and "bce" in self.criterion:
                    num_class.append(1)
                else:
                    num_class.append(task_class + 1)
            else:
                num_class.append(1)

        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float))
        self.register_buffer("weight", torch.as_tensor(weight, dtype=torch.float))
        self.num_class = self.num_class or num_class

        hidden_dims = [self.model.output_dim] * (self.num_mlp_layer - 1)
        self.mlp = layers.MLP(self.model.output_dim, hidden_dims + [sum(self.num_class)],
                            batch_norm=self.mlp_batch_norm, dropout=self.mlp_dropout)

    def visualize_attributions(self, attributions, epoch):
        # 将属性转换为numpy数组
        attributions_np = attributions.cpu().detach().numpy()
        amino_acid_contributions = attributions.sum(dim=-1).cpu().detach().numpy()
        
        # 计算每个蛋白质的最大贡献度的50%值
        threshold = 0.5 * amino_acid_contributions.max(axis=1, keepdims=True)
        
        # 仅保留大于或等于该阈值的值，其余值设为np.nan
        masked_contributions = np.where(amino_acid_contributions >= threshold, amino_acid_contributions, np.nan)
    
    
        plt.figure(figsize=(10, 8))  # 设置图的大小

        sns.heatmap(masked_contributions, cmap="rocket", cbar_kws={'label': 'Contribution Value'})
        plt.title("Top Attributions Heatmap")
        plt.ylabel("Batch Index")
        plt.xlabel("Sequence Position")
        # 保存图片
        plt.savefig("/path/{}.png".format(epoch))

    def forward(self, batch, ig=False, sub_policy=None, aug_list=None):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}
        if self.ig is False:
            ig = False
        if self.aug == "APA":
            if ig is False:
                aug_batch = augmentation_batch(batch, self.n_aug, sub_policy, self.aug_list, self.aug_num)
                pred = self.predict(aug_batch, all_loss, metric)
            
                if all([t not in batch for t in self.task]):
                    # unlabeled data
                    return all_loss, metric

                target = self.target(batch)
                target = target.repeat(self.n_aug + 1, 1)
            else:
                if "graph" in batch:
                    if type(self.model).__name__ == "ProteinResNet":
                        processed_input, mask = self.model.preprocess_graph(batch["graph"])
                    else:
                        processed_input = self.model.preprocess_graph(batch["graph"])

                    processed_input = processed_input.float()
                    processed_input.requires_grad = True

                    if type(self.model).__name__ == "ProteinResNet":
                        pred = self.predict_wrapper_with_mask(processed_input, mask, batch["graph"])
                    else:
                        pred = self.predict_wrapper(processed_input)

                    if type(self.model).__name__ == "ProteinLSTM":
                        integrated_gradients = IntegratedGradients(self.predict_wrapper)
                    elif type(self.model).__name__ == "ProteinResNet":
                        integrated_gradients = IntegratedGradients(self.predict_wrapper_with_mask)
                    elif type(self.model).__name__ == "EvolutionaryScaleModeling":
                        lig = LayerIntegratedGradients(self.predict_wrapper, self.model.model)
                    
                    baseline = torch.zeros_like(processed_input).to(processed_input.device)
                    baseline[:,-1] = 1
                    _, predicted_class = torch.max(pred, 1)

                    if type(self.model).__name__ == "ProteinLSTM":
                        attributions = integrated_gradients.attribute(processed_input, target=predicted_class, n_steps=self.n_step)
                    elif type(self.model).__name__ == "ProteinResNet":
                        attributions = integrated_gradients.attribute(processed_input, target=predicted_class, n_steps=self.n_step, additional_forward_args=(mask, batch["graph"]))
                    elif type(self.model).__name__ == "EvolutionaryScaleModeling":
                        attributions = lig.attribute(processed_input, baseline, target=predicted_class, n_steps=self.n_step)

                    target = self.target(batch)
                    aug_batch = augmentation_batch_ig(batch, n_aug=self.n_aug, attributions=attributions, sub_policy=sub_policy, aug_num=self.aug_num)
                # add interact property
                pred = self.predict(aug_batch, all_loss, metric)
                target = self.target(batch)
                target = target.repeat(self.n_aug + 1, 1)
                
        elif self.aug == "ig":
            if type(self.model).__name__ == "ProteinResNet":
                processed_input, mask = self.model.preprocess_graph(batch["graph"])
            else:
                processed_input = self.model.preprocess_graph(batch["graph"])

            processed_input = processed_input.float()
            processed_input.requires_grad = True
            
            if type(self.model).__name__ == "ProteinResNet":
                pred = self.predict_wrapper_with_mask(processed_input, mask, batch["graph"])
            else:
                pred = self.predict_wrapper(processed_input)
            
            if type(self.model).__name__ == "ProteinLSTM":
                integrated_gradients = IntegratedGradients(self.predict_wrapper)
            elif type(self.model).__name__ == "ProteinResNet":
                integrated_gradients = IntegratedGradients(self.predict_wrapper_with_mask)
            elif type(self.model).__name__ == "EvolutionaryScaleModeling":
                lig = LayerIntegratedGradients(self.predict_wrapper, self.model.model)

            baseline = torch.zeros_like(processed_input).to(processed_input.device)
            baseline[:,-1] = 1
            _, predicted_class = torch.max(pred, 1)

            if type(self.model).__name__ == "ProteinLSTM":
                attributions = integrated_gradients.attribute(processed_input, target=predicted_class, n_steps=self.n_step)
            elif type(self.model).__name__ == "ProteinResNet":
                attributions = integrated_gradients.attribute(processed_input, target=predicted_class, n_steps=self.n_step, additional_forward_args=(mask, batch["graph"]))
            elif type(self.model).__name__ == "EvolutionaryScaleModeling":
                attributions = lig.attribute(processed_input, baseline, target=predicted_class, n_steps=self.n_step)

            target = self.target(batch)
            
            aug_batch = augmentation_batch_ig(batch, n_aug=self.n_aug, attributions=attributions, aug_list=["ig_random_substitute"])
            pred = self.predict(aug_batch, all_loss, metric)
            target = self.target(batch)
            target = target.repeat(self.n_aug + 1, 1)
        elif self.aug == "aug":
            aug_batch = augmentation_batch(batch, self.n_aug, sub_policy, self.aug_list)
            pred = self.predict(aug_batch, all_loss, metric)
            
            if all([t not in batch for t in self.task]):
                # unlabeled data
                return all_loss, metric

            target = self.target(batch)
            target = target.repeat(self.n_aug + 1, 1)
        else:
            pred = self.predict(batch, all_loss, metric)
            if all([t not in batch for t in self.task]):
                # unlabeled data
                return all_loss, metric
            target = self.target(batch)

        labeled = ~torch.isnan(target)
        target[~labeled] = 0


        for criterion, weight in self.criterion.items():
            if criterion == "mse":
                if self.normalization:
                    loss = F.mse_loss((pred - self.mean) / self.std, (target - self.mean) / self.std, reduction="none")
                else:
                    loss = F.mse_loss(pred, target, reduction="none")
            elif criterion == "bce":
                loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
            elif criterion == "ce":
                loss = F.cross_entropy(pred, target.long().squeeze(-1), reduction="none").unsqueeze(-1)
            else:
                raise ValueError("Unknown criterion `%s`" % criterion)
            loss = functional.masked_mean(loss, labeled, dim=0)

            name = tasks._get_criterion_name(criterion)
            if self.verbose > 0:
                for t, l in zip(self.task, loss):
                    metric["%s [%s]" % (name, t)] = l
            loss = (loss * self.weight).sum() / self.weight.sum()
            metric[name] = loss
            all_loss += loss * weight

        return all_loss, metric

    def predict(self, batch, all_loss=None, metric=None):
        graph = batch["graph"]
        if self.graph_construction_model:
            graph = self.graph_construction_model(graph)
        output = self.model(graph, graph.node_feature, all_loss=all_loss, metric=metric)
        if self.batchnorm == True:
            pred = self.mlp(self.bn(output["graph_feature"]))
        else:
            pred = self.mlp(output["graph_feature"])
        if self.normalization:
            pred = pred * self.std + self.mean
        return pred
    
    def predict_wrapper_with_mask(self, processed_input, mask=None, graph=None):
        output = self.model(graph=graph, input=processed_input, mask=mask, all_loss=None, metric=None, ig=True)
        if self.batchnorm == True:
            pred = self.mlp(self.bn(output["graph_feature"]))
        else:
            pred = self.mlp(output["graph_feature"])
        if self.normalization:
            pred = pred * self.std + self.mean
        return pred

    def predict_wrapper(self, processed_input):
        output = self.model(None, processed_input, None, None, True)
        if self.batchnorm == True:
            pred = self.mlp(self.bn(output["graph_feature"]))
        else:
            pred = self.mlp(output["graph_feature"])
        if self.normalization:
            pred = pred * self.std + self.mean
        return pred
    
    def target(self, batch):
        target = torch.stack([batch[t].float() for t in self.task], dim=-1)
        labeled = batch.get("labeled", torch.ones(len(target), dtype=torch.bool, device=target.device))
        target[~labeled] = math.nan
        return target

    def evaluate(self, pred, target):
        labeled = ~torch.isnan(target)

        metric = {}
        for _metric in self.metric:
            if _metric == "mae":
                score = F.l1_loss(pred, target, reduction="none")
                score = functional.masked_mean(score, labeled, dim=0)
            elif _metric == "rmse":
                score = F.mse_loss(pred, target, reduction="none")
                score = functional.masked_mean(score, labeled, dim=0).sqrt()
            elif _metric == "acc":
                score = []
                num_class = 0
                for i, cur_num_class in enumerate(self.num_class):
                    _pred = pred[:, num_class:num_class + cur_num_class]
                    _target = target[:, i]
                    _labeled = labeled[:, i]
                    _score = metrics.accuracy(_pred[_labeled], _target[_labeled].long())
                    score.append(_score)
                    num_class += cur_num_class
                score = torch.stack(score)
            elif _metric == "mcc":
                score = []
                num_class = 0
                for i, cur_num_class in enumerate(self.num_class):
                    _pred = pred[:, num_class:num_class + cur_num_class]
                    _target = target[:, i]
                    _labeled = labeled[:, i]
                    _score = metrics.matthews_corrcoef(_pred[_labeled], _target[_labeled].long())
                    score.append(_score)
                    num_class += cur_num_class
                score = torch.stack(score)
            elif _metric == "auroc":
                score = []
                for _pred, _target, _labeled in zip(pred.t(), target.long().t(), labeled.t()):
                    _score = metrics.area_under_roc(_pred[_labeled], _target[_labeled])
                    score.append(_score)
                score = torch.stack(score)
            elif _metric == "auprc":
                score = []
                for _pred, _target, _labeled in zip(pred.t(), target.long().t(), labeled.t()):
                    _score = metrics.area_under_prc(_pred[_labeled], _target[_labeled])
                    score.append(_score)
                score = torch.stack(score)
            elif _metric == "r2":
                score = []
                for _pred, _target, _labeled in zip(pred.t(), target.t(), labeled.t()):
                    _score = metrics.r2(_pred[_labeled], _target[_labeled])
                    score.append(_score)
                score = torch.stack(score)
            elif _metric == "spearmanr":
                score = []
                for _pred, _target, _labeled in zip(pred.t(), target.t(), labeled.t()):
                    _score = metrics.spearmanr(_pred[_labeled], _target[_labeled])
                    score.append(_score)
                score = torch.stack(score)
            elif _metric == "pearsonr":
                score = []
                for _pred, _target, _labeled in zip(pred.t(), target.t(), labeled.t()):
                    _score = metrics.pearsonr(_pred[_labeled], _target[_labeled])
                    score.append(_score)
                score = torch.stack(score)
            else:
                raise ValueError("Unknown metric `%s`" % _metric)

            name = tasks._get_metric_name(_metric)
            for t, s in zip(self.task, score):
                metric["%s [%s]" % (name, t)] = s

        return metric


@R.register("tasks.MultipleBinaryClassification")
class MultipleBinaryClassification(tasks.Task, core.Configurable):
    """
    Multiple binary classification task for graphs / molecules / proteins.

    Parameters:
        model (nn.Module): graph representation model
        task (list of int, optional): training task id(s).
        criterion (list or dict, optional): training criterion(s). For dict, the keys are criterions and the values
            are the corresponding weights. Available criterions are ``bce``.
        metric (str or list of str, optional): metric(s).
            Available metrics are ``auroc@macro``, ``auprc@macro``, ``auroc@micro``, ``auprc@micro`` and ``f1_max``.
        num_mlp_layer (int, optional): number of layers in the MLP prediction head
        normalization (bool, optional): whether to normalize the target
        reweight (bool, optional): whether to re-weight tasks according to the number of positive samples
        graph_construction_model (nn.Module, optional): graph construction model
        verbose (int, optional): output verbose level
        aug (str): which augmentation to use
        n_aug (int): number of augmented samples
        aug_list (list, optional): list of augmentation functions
        ig (bool): whether to use integrated gradients
        n_step (int): number of steps for integrated gradients
    """

    eps = 1e-10
    _option_members = {"criterion", "metric"}

    def __init__(self, model, task=(), criterion="bce", metric=("auprc@micro", "f1_max"), num_mlp_layer=1,
                 normalization=True, reweight=False, graph_construction_model=None, verbose=0, batchnorm=False, aug=None, n_aug=1, aug_list=None, ig=None):
        super(MultipleBinaryClassification, self).__init__()
        self.model = model
        self.task = task
        self.register_buffer("task_indices", torch.LongTensor(task))
        self.criterion = criterion
        self.metric = metric
        self.num_mlp_layer = num_mlp_layer
        self.normalization = normalization
        self.reweight = reweight
        self.graph_construction_model = graph_construction_model
        self.verbose = verbose
        self.batchnorm = batchnorm
        self.aug = aug
        self.n_aug = n_aug
        self.aug_list = aug_list
        if self.batchnorm == True:
            self.bn = nn.BatchNorm1d(self.model.output_dim)
        self.ig = ig

        hidden_dims = [self.model.output_dim] * (self.num_mlp_layer - 1)
        self.mlp = layers.MLP(self.model.output_dim, hidden_dims + [len(task)])

    def preprocess(self, train_set, valid_set, test_set):
        """
        Compute the weight for each task on the training set.
        """
        values = []
        for data in train_set:
            values.append(data["targets"][self.task_indices])
        values = torch.stack(values, dim=0)    
        
        if self.reweight:
            num_positive = values.sum(dim=0)
            weight = (num_positive.mean() / num_positive).clamp(1, 10)
        else:
            weight = torch.ones(len(self.task), dtype=torch.float)

        self.register_buffer("weight", torch.as_tensor(weight, dtype=torch.float))

    def forward(self, batch, ig=False, sub_policy=None, aug_list=None):
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        if self.ig is False:
            ig = False
        if self.aug == "APA":
            if ig is False:
                aug_batch = augmentation_batch(batch, self.n_aug, sub_policy, self.aug_list)
                pred = self.predict(aug_batch, all_loss, metric)

                target = self.target(batch)
                target = target.repeat(self.n_aug + 1, 1)
            else:
                if type(self.model).__name__ == "ProteinResNet":
                    processed_input, mask = self.model.preprocess_graph(batch["graph"])
                else:
                    processed_input = self.model.preprocess_graph(batch["graph"])
                processed_input = processed_input.float()
                processed_input.requires_grad = True
                if type(self.model).__name__ == "ProteinResNet":
                    pred = self.predict_wrapper_with_mask(processed_input, mask, batch["graph"])
                else:
                    pred = self.predict_wrapper(processed_input)
                
                if type(self.model).__name__ == "ProteinLSTM":
                    integrated_gradients = IntegratedGradients(self.predict_wrapper)
                elif type(self.model).__name__ == "ProteinResNet":
                    integrated_gradients = IntegratedGradients(self.predict_wrapper_with_mask)
                elif type(self.model).__name__ == "EvolutionaryScaleModeling":
                    lig = LayerIntegratedGradients(self.predict_wrapper, self.model.model)
                baseline = torch.zeros_like(processed_input).to(processed_input.device)
                baseline[:,-1] = 1
                _, predicted_class = torch.max(pred, 1)
                if type(self.model).__name__ == "ProteinLSTM":
                    attributions = integrated_gradients.attribute(processed_input, target=predicted_class, n_steps=self.n_step)
                elif type(self.model).__name__ == "ProteinResNet":
                    attributions = integrated_gradients.attribute(processed_input, target=predicted_class, n_steps=self.n_step, additional_forward_args=(mask, batch["graph"]))
                elif type(self.model).__name__ == "EvolutionaryScaleModeling":
                    attributions = lig.attribute(processed_input, baseline, target=predicted_class, n_steps=self.n_step)
                
                target = self.target(batch)
                aug_batch = augmentation_batch_ig(batch, n_aug=self.n_aug, attributions=attributions, sub_policy=sub_policy)
                pred = self.predict(aug_batch, all_loss, metric)
                target = self.target(batch)
                target = target.repeat(self.n_aug + 1, 1)
                
        elif self.aug == "ig":
            if type(self.model).__name__ == "ProteinResNet":
                processed_input, mask = self.model.preprocess_graph(batch["graph"])
            else:
                processed_input = self.model.preprocess_graph(batch["graph"])
            processed_input = processed_input.float()
            processed_input.requires_grad = True
            if type(self.model).__name__ == "ProteinResNet":
                pred = self.predict_wrapper_with_mask(processed_input, mask, batch["graph"])
            else:
                pred = self.predict_wrapper(processed_input)
            
            if type(self.model).__name__ == "ProteinLSTM":
                integrated_gradients = IntegratedGradients(self.predict_wrapper)
            elif type(self.model).__name__ == "ProteinResNet":
                integrated_gradients = IntegratedGradients(self.predict_wrapper_with_mask)
            elif type(self.model).__name__ == "EvolutionaryScaleModeling":
                lig = LayerIntegratedGradients(self.predict_wrapper, self.model.model)
            baseline = torch.zeros_like(processed_input).to(processed_input.device)
            baseline[:,-1] = 1
            _, predicted_class = torch.max(pred, 1)
            if type(self.model).__name__ == "ProteinLSTM":
                attributions = integrated_gradients.attribute(processed_input, target=predicted_class, n_steps=self.n_step)
            elif type(self.model).__name__ == "ProteinResNet":
                attributions = integrated_gradients.attribute(processed_input, target=predicted_class, n_steps=self.n_step, additional_forward_args=(mask, batch["graph"]))
            elif type(self.model).__name__ == "EvolutionaryScaleModeling":
                attributions = lig.attribute(processed_input, baseline, target=predicted_class, n_steps=self.n_step)
            
            target = self.target(batch)
            
            aug_batch = augmentation_batch_ig(batch, n_aug=self.n_aug, attributions=attributions, aug_list=["ig_random_substitute"])
            pred = self.predict(aug_batch, all_loss, metric)
            target = self.target(batch)
            target = target.repeat(self.n_aug + 1, 1)
        elif self.aug == "aug":
            aug_batch = augmentation_batch(batch, self.n_aug, sub_policy, self.aug_list)
            pred = self.predict(aug_batch, all_loss, metric)

            target = self.target(batch)
            target = target.repeat(self.n_aug + 1, 1)
        else:
            pred = self.predict(batch, all_loss, metric)
            target = self.target(batch)

        for criterion, weight in self.criterion.items():
            if criterion == "bce":
                loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
            else:
                raise ValueError("Unknown criterion `%s`" % criterion)
            loss = loss.mean(dim=0)
            loss = (loss * self.weight).sum() / self.weight.sum()
            
            name = tasks._get_criterion_name(criterion)
            metric[name] = loss
            metric["pred"] = pred.mean(dim=-1).mean(dim=-1)
            all_loss += loss * weight

        return all_loss, metric

    def predict(self, batch, all_loss=None, metric=None):
        graph = batch["graph"]
        if self.graph_construction_model:
            graph = self.graph_construction_model(graph)
        output = self.model(graph, graph.node_feature.float(), all_loss=all_loss, metric=metric)
        if self.batchnorm == True:
            pred = self.mlp(self.bn(output["graph_feature"]))
        else:
            pred = self.mlp(output["graph_feature"])
        return pred
    
    def predict_wrapper_with_mask(self, processed_input, mask, graph):
        output = self.model(graph, processed_input, mask, None, None, True)
        if self.batchnorm == True:
            pred = self.mlp(self.bn(output["graph_feature"]))
        else:
            pred = self.mlp(output["graph_feature"])
        return pred
    
    def predict_wrapper(self, processed_input):
        output = self.model(None, processed_input, None, None, True)
        if self.batchnorm == True:
            pred = self.mlp(self.bn(output["graph_feature"]))
        else:
            pred = self.mlp(output["graph_feature"])
        return pred

    def target(self, batch):
        target = batch["targets"][:, self.task_indices]
        return target

    def evaluate(self, pred, target):
        metric = {}
        for _metric in self.metric:
            if _metric == "auroc@micro":
                score = metrics.area_under_roc(pred.flatten(), target.long().flatten())
            elif _metric == "auroc@macro":
                score = metrics.variadic_area_under_roc(pred, target.long(), dim=0).mean()
            elif _metric == "auprc@micro":
                score = metrics.area_under_prc(pred.flatten(), target.long().flatten())
            elif _metric == "auprc@macro":
                score = metrics.variadic_area_under_prc(pred, target.long(), dim=0).mean()
            elif _metric == "f1_max":
                score = metrics.f1_max(pred, target)
            else:
                raise ValueError("Unknown criterion `%s`" % _metric)

            name = tasks._get_metric_name(_metric)
            metric[name] = score

        return metric


@R.register("tasks.NodePropertyPrediction")
class NodePropertyPrediction(tasks.Task, core.Configurable):
    """
    Node / atom / residue property prediction task.

    Parameters:
        model (nn.Module): graph representation model
        criterion (str, list or dict, optional): training criterion(s). For dict, the keys are criterions and the values
            are the corresponding weights. Available criterions are ``mse``, ``bce`` and ``ce``.
        metric (str or list of str, optional): metric(s).
            Available metrics are ``mae``, ``rmse``, ``auprc`` and ``auroc``.
        num_mlp_layer (int, optional): number of layers in mlp prediction head
        normalization (bool, optional): whether to normalize the target
            Available entities are ``node``, ``atom`` and ``residue``.
        num_class (int, optional): number of classes
        verbose (int, optional): output verbose level
    """

    _option_members = {"criterion", "metric"}

    def __init__(self, model, criterion="bce", metric=("macro_auprc", "macro_auroc"), num_mlp_layer=1,
                 normalization=True, num_class=None, verbose=0, batchnorm=False, augmentation=False):
        super(NodePropertyPrediction, self).__init__()
        self.model = model
        self.criterion = criterion
        self.metric = metric
        # For classification tasks, we disable normalization tricks.
        self.normalization = normalization and ("ce" not in criterion) and ("bce" not in criterion)
        self.num_mlp_layer = num_mlp_layer
        self.num_class = num_class
        self.verbose = verbose
        self.batchnorm = batchnorm
        self.augmentation = augmentation

    def preprocess(self, train_set, valid_set, test_set):
        """
        Compute the mean and derivation on the training set.
        """
        self.view = getattr(train_set[0]["graph"], "view", "atom")
        values = torch.cat([data["graph"].target for data in train_set])
        mean = values.float().mean()
        std = values.float().std()
        if values.dtype == torch.long:
            num_class = values.max().item()
            if num_class > 1 or "bce" not in self.criterion:
                num_class += 1
        else:
            num_class = 1

        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float))
        self.num_class = self.num_class or num_class

        if hasattr(self.model, "node_output_dim"):
            model_output_dim = self.model.node_output_dim
        else:
            model_output_dim = self.model.output_dim
        hidden_dims = [model_output_dim] * (self.num_mlp_layer - 1)
        self.mlp = layers.MLP(model_output_dim, hidden_dims + [self.num_class])
        if self.batchnorm == True:
            self.bn = nn.BatchNorm1d(model_output_dim)

    def predict(self, batch, all_loss=None, metric=None):
        graph = batch["graph"]
        output = self.model(graph, graph.node_feature.float(), all_loss=all_loss, metric=metric)
        if self.view in ["node", "atom"]:
            output_feature = output["node_feature"]
        else:
            output_feature = output.get("residue_feature", output.get("node_feature"))
        # pred = self.mlp(output_feature)
        if self.batchnorm == True:
            pred = self.mlp(self.bn(output_feature))
        else:
            pred = self.mlp(output_feature)
        if self.normalization:
            pred = pred * self.std + self.mean
        return pred

    def target(self, batch):
        size = batch["graph"].num_nodes if self.view in ["node", "atom"] else batch["graph"].num_residues
        return {
            "label": batch["graph"].target,
            "mask": batch["graph"].mask,
            "size": size
        }

    def forward(self, batch):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}
        # breakpoint()
        # pred, target = self.predict_and_target(batch, all_loss, metric)
        if self.augmentation is True:
            target = self.target(batch)
            target["label"] = torch.concat((target["label"], target["label"]), dim=0)
            target["mask"] = torch.concat((target["mask"], target["mask"]), dim=0)
            target["size"] = target["size"] * 2
            
            aug_batch = replacement_dictionary_aug(batch, p=0.01)
            pred = self.predict(aug_batch, all_loss, metric)
            
        else:
            pred = self.predict(batch, all_loss, metric)
            target = self.target(batch)

        labeled = ~torch.isnan(target["label"]) & target["mask"]

        for criterion, weight in self.criterion.items():
            if criterion == "mse":
                if self.normalization:
                    loss = F.mse_loss((pred - self.mean) / self.std, (target - self.mean) / self.std, reduction="none")
                else:
                    loss = F.mse_loss(pred, target, reduction="none")
            elif criterion == "bce":
                loss = F.binary_cross_entropy_with_logits(pred, target["label"].float(), reduction="none")
            elif criterion == "ce":
                loss = F.cross_entropy(pred, target["label"], reduction="none")
            else:
                raise ValueError("Unknown criterion `%s`" % criterion)
            loss = functional.masked_mean(loss, labeled, dim=0)

            name = tasks._get_criterion_name(criterion)
            metric[name] = loss
            all_loss += loss * weight

        all_loss += loss

        return all_loss, metric

    def evaluate(self, pred, target):
        metric = {}
        _target = target["label"]
        _labeled = ~torch.isnan(_target) & target["mask"]
        _size = functional.variadic_sum(_labeled.long(), target["size"]) 
        for _metric in self.metric:
            if _metric == "micro_acc":
                score = metrics.accuracy(pred[_labeled], _target[_labeled].long())
            elif metric == "micro_auroc":
                score = metrics.area_under_roc(pred[_labeled], _target[_labeled])
            elif metric == "micro_auprc":
                score = metrics.area_under_prc(pred[_labeled], _target[_labeled])
            elif _metric == "macro_auroc":
                score = metrics.variadic_area_under_roc(pred[_labeled], _target[_labeled], _size).mean()
            elif _metric == "macro_auprc":
                score = metrics.variadic_area_under_prc(pred[_labeled], _target[_labeled], _size).mean()
            elif _metric == "macro_acc":
                score = pred[_labeled].argmax(-1) == _target[_labeled]
                score = functional.variadic_mean(score.float(), _size).mean()
            else:
                raise ValueError("Unknown criterion `%s`" % _metric)

            name = tasks._get_metric_name(_metric)
            metric[name] = score

        return metric


@R.register("tasks.InteractionPrediction")
@utils.copy_args(PropertyPrediction, ignore=("graph_construction_model",))
class InteractionPrediction(PropertyPrediction):
    """
    Predict the interaction property of graph pairs.

    Parameters:
        model (nn.Module): graph representation model
        model2 (nn.Module, optional): graph representation model for the second item. If ``None``, use tied-weight
            model for the second item.
        **kwargs
    """

    def __init__(self, model, model2=None, **kwargs):
        super(InteractionPrediction, self).__init__(model, **kwargs)
        self.model2 = model2 or model

    def preprocess(self, train_set, valid_set, test_set):
        """
        Compute the mean and derivation for each task on the training set.
        """
        values = defaultdict(list)
        for sample in train_set:
            if not sample.get("labeled", True):
                continue
            for task in self.task:
                if not math.isnan(sample[task]):
                    values[task].append(sample[task])
        mean = []
        std = []
        weight = []
        num_class = []
        for task, w in self.task.items():
            value = torch.tensor(values[task])
            mean.append(value.float().mean())
            std.append(value.float().std())
            weight.append(w)
            if value.ndim > 1:
                num_class.append(value.shape[1])
            elif value.dtype == torch.long:
                task_class = value.max().item()
                if task_class == 1 and "bce" in self.criterion:
                    num_class.append(1)
                else:
                    num_class.append(task_class + 1)
            else:
                num_class.append(1)

        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float))
        self.register_buffer("weight", torch.as_tensor(weight, dtype=torch.float))
        self.num_class = self.num_class or num_class

        hidden_dims = [self.model.output_dim] * (self.num_mlp_layer - 1)
        self.mlp = layers.MLP(self.model.output_dim + self.model2.output_dim, hidden_dims + [sum(self.num_class)])
        if self.batchnorm == True:
            self.bn = nn.BatchNorm1d(self.model.output_dim + self.model2.output_dim)

    def predict(self, batch, all_loss=None, metric=None):
        graph1 = batch["graph1"]
        output1 = self.model(graph1, graph1.node_feature.float(), all_loss=all_loss, metric=metric)
        graph2 = batch["graph2"]
        output2 = self.model2(graph2, graph2.node_feature.float(), all_loss=all_loss, metric=metric)
        # pred = self.mlp(torch.cat([output1["graph_feature"], output2["graph_feature"]], dim=-1))
        if self.batchnorm == True:
            pred = self.mlp(self.bn(torch.cat([output1["graph_feature"], output2["graph_feature"]], dim=-1)))
        else:
            pred = self.mlp(torch.cat([output1["graph_feature"], output2["graph_feature"]], dim=-1))
        if self.normalization:
            pred = pred * self.std + self.mean
        return pred
    
    def predict_wrapper(self, *inputs):
        # breakpoint()
        processed_input1 = inputs[0]
        processed_input2 = inputs[1]
        output1 = self.model(None, processed_input1, None, None, True)
        output2 = self.model2(None, processed_input2, None, None, True)
        if self.batchnorm == True:
            pred = self.mlp(self.bn(torch.cat([output1["graph_feature"], output2["graph_feature"]], dim=-1)))
        else:
            pred = self.mlp(torch.cat([output1["graph_feature"], output2["graph_feature"]], dim=-1))
        if self.normalization:
            pred = pred * self.std + self.mean
        return pred

        

@R.register("tasks.Unsupervised")
class Unsupervised(nn.Module, core.Configurable):
    """
    Wrapper task for unsupervised learning.

    The unsupervised loss should be computed by the model.

    Parameters:
        model (nn.Module): any model
    """

    def __init__(self, model, graph_construction_model=None):
        super(Unsupervised, self).__init__()
        self.model = model
        self.graph_construction_model = graph_construction_model

    def forward(self, batch):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred = self.predict(batch, all_loss, metric)

        return all_loss, metric

    def predict(self, batch, all_loss=None, metric=None):
        graph = batch["graph"]
        if self.graph_construction_model:
            graph = self.graph_construction_model(graph)
        pred = self.model(graph, graph.node_feature.float(), all_loss=all_loss, metric=metric)
        return pred
