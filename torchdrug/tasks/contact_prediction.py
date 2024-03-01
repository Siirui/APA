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

def augmentation_batch(batch, n_aug=1, sub_policy=None, aug_list=None):
    if "graph" in batch:
        graph = batch["graph"]
        device = graph.device
        sequences = graph.to_sequence()
        sequences = [seq.replace(".", "") for seq in sequences]
        sequence_list = [list(seq) for seq in sequences]
        for _ in range(n_aug):
            for seq in sequence_list:
                tmp_seq = copy.deepcopy(seq)
                tmp_seq = random_augment(tmp_seq, sub_policy=sub_policy, aug_list=aug_list)
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
                tmp_seq = random_augment(tmp_seq, sub_policy=sub_policy, aug_list=aug_list)
                sequences1.append("".join(tmp_seq))
            for seq in sequence_list2:
                tmp_seq = copy.deepcopy(seq)
                tmp_seq = random_augment(tmp_seq, sub_policy=sub_policy, aug_list=aug_list)
                sequences2.append("".join(tmp_seq))
        graph1 = data.PackedProtein.from_sequences(sequences1, atom_feature=None, bond_feature=None)
        graph2 = data.PackedProtein.from_sequences(sequences2, atom_feature=None, bond_feature=None)
        graph1 = graph1.to(device)
        graph2 = graph2.to(device)
        aug_batch = copy.deepcopy(batch)
        aug_batch["graph1"] = graph1
        aug_batch["graph2"] = graph2
        # breakpoint()
        return aug_batch

def augmentation_batch_ig(batch, n_aug=1, sub_policy=None, attributions=None, attributions2=None):
    amino_acid_contributions = attributions.sum(dim=-1)
    amino_acid_contributions = amino_acid_contributions[:,1:-1]
    # breakpoint()
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

    if "graph" in batch:
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
                # breakpoint()
                # print(policy)
                tmp_seq = random_augment(tmp_seq, sub_policy=sub_policy, aug_list=None, indices=replace_indices)
                # replace_pos = random.sample(replace_indices, int(0.1 * len(replace_indices)))
                # # breakpoint()
                # for index in replace_pos:
                #     tmp_seq[index] = random.choice(list(residue_symbol2id.keys()))

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
            for seq_index, seq in enumerate(sequence_list1):
                tmp_seq = copy.deepcopy(seq)
                above_threshold_indices[seq_index][len(tmp_seq):] = False
                replace_indices = torch.where(above_threshold_indices[seq_index])[0].tolist()
                tmp_seq = random_augment(tmp_seq, sub_policy=sub_policy, aug_list=None, indices=replace_indices)
                sequences1.append("".join(tmp_seq))
            for seq_index, seq in enumerate(sequence_list2):
                tmp_seq = copy.deepcopy(seq)
                above_threshold_indices2[seq_index][len(tmp_seq):] = False
                replace_indices = torch.where(above_threshold_indices2[seq_index])[0].tolist()
                tmp_seq = random_augment(tmp_seq, sub_policy=sub_policy, aug_list=None, indices=replace_indices)
                sequences2.append("".join(tmp_seq))
        graph1 = data.PackedProtein.from_sequences(sequences1, atom_feature=None, bond_feature=None)
        graph2 = data.PackedProtein.from_sequences(sequences2, atom_feature=None, bond_feature=None)
        graph1 = graph1.to(device)
        graph2 = graph2.to(device)
        aug_batch = copy.deepcopy(batch)
        aug_batch["graph1"] = graph1
        aug_batch["graph2"] = graph2
        # breakpoint()
        return aug_batch



@R.register("tasks.ContactPrediction")
class ContactPrediction(tasks.Task, core.Configurable):
    """
    Predict whether each amino acid pair contact or not in the folding structure.

    Parameters:
        model (nn.Module): protein sequence representation model
        max_length (int, optional): maximal length of sequence. Truncate the sequence if it exceeds this limit.
        random_truncate (bool, optional): truncate the sequence at a random position.
            If not, truncate the suffix of the sequence.
        threshold (float, optional): distance threshold for contact
        gap (int, optional): sequential distance cutoff for evaluation
        criterion (str or dict, optional): training criterion. For dict, the key is criterion and the value
            is the corresponding weight. Available criterion is ``bce``.
        metric (str or list of str, optional): metric(s).
            Available metrics are ``accuracy``, ``prec@Lk`` and ``prec@k``.
        num_mlp_layer (int, optional): number of layers in mlp prediction head
        verbose (int, optional): output verbose level
        batchnorm (bool): whether to apply batch normalization before MLP
        aug (str): which augmentation to use
        n_aug (int): number of augmented samples
        aug_list (list, optional): list of augmentation functions
        ig (bool): whether to use integrated gradients
    """

    eps = 1e-10
    _option_members = {"task", "criterion", "metric"}

    def __init__(self, model, max_length=500, random_truncate=True, threshold=8.0, gap=6, criterion="bce",
                 metric=("accuracy", "prec@L5"), num_mlp_layer=1, verbose=0, aug=None, batchnorm=False, n_aug=1, aug_list=None, ig=None):
        super(ContactPrediction, self).__init__()
        self.model = model
        self.max_length = max_length
        self.random_truncate = random_truncate
        self.threshold = threshold
        self.gap = gap
        self.criterion = criterion
        self.metric = metric
        self.num_mlp_layer = num_mlp_layer
        self.verbose = verbose
        self.batchnorm = batchnorm
        self.aug = aug
        self.n_aug = n_aug
        self.aug_list = aug_list
        if self.batchnorm == True:
            self.bn = nn.BatchNorm1d(self.model.output_dim)
        self.ig = ig

        if hasattr(self.model, "node_output_dim"):
            model_output_dim = self.model.node_output_dim
        else:
            model_output_dim = self.model.output_dim

        if self.batchnorm == True:
            self.bn = nn.BatchNorm1d(model_output_dim)

        hidden_dims = [model_output_dim] * (self.num_mlp_layer - 1)

        self.mlp = layers.MLP(2 * model_output_dim, hidden_dims + [1])

    def truncate(self, batch):
        graph = batch["graph"]
        size = graph.num_residues
        if (size > self.max_length).any():
            if self.random_truncate:
                starts = (torch.rand(graph.batch_size, device=graph.device) * \
                          (graph.num_residues - self.max_length).clamp(min=0)).long()
                ends = torch.min(starts + self.max_length, graph.num_residues)
                starts = starts + (graph.num_cum_residues - graph.num_residues)
                ends = ends + (graph.num_cum_residues - graph.num_residues)
                mask = functional.multi_slice_mask(starts, ends, graph.num_residue)
            else:
                starts = size.cumsum(0) - size
                size = size.clamp(max=self.max_length)
                ends = starts + size
                mask = functional.multi_slice_mask(starts, ends, graph.num_residue)
            graph = graph.subresidue(mask)

        return {
            "graph": graph
        }

    def forward(self, batch, ig=False, sub_policy=None, aug_list=None):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        batch = self.truncate(batch)
        # breakpoint()
        if self.ig is False:
            ig = False
        if self.aug == "PAA":
            if ig is False:
                aug_batch = augmentation_batch(batch, self.n_aug, sub_policy, self.aug_list)
                pred = self.predict(aug_batch, all_loss, metric)

                proteins = batch["graph"]
                pro_lis = [pro for pro in proteins]
                pro_lis = pro_lis + pro_lis
                duplicate_proteins = data.Protein.pack(pro_lis)
                duplicate_batch = copy.deepcopy(batch)
                duplicate_batch["graph"] = duplicate_proteins
                target = self.target(duplicate_batch)
            else:
                processed_input = self.model.preprocess_graph(batch["graph"])
                processed_input = processed_input.float()
                processed_input.requires_grad = True
                pred = self.predict_wrapper(processed_input)
                lig = LayerIntegratedGradients(self.predict_wrapper, self.model.model)
                baseline = torch.zeros_like(processed_input).to(processed_input.device)
                baseline[:,-1] = 1
                _, predicted_class = torch.max(pred, 1)
                attributions = lig.attribute(processed_input, baseline, target=predicted_class, n_steps=5)
                aug_batch = augmentation_batch_ig(batch, n_aug=self.n_aug, attributions=attributions, sub_policy=sub_policy)

                proteins = batch["graph"]
                pro_lis = [pro for pro in proteins]
                pro_lis = pro_lis + pro_lis
                duplicate_proteins = data.Protein.pack(pro_lis)
                duplicate_batch = copy.deepcopy(batch)
                duplicate_batch["graph"] = duplicate_proteins
                target = self.target(duplicate_batch)
        elif self.aug == "ig":
            processed_input = self.model.preprocess_graph(batch["graph"])
            processed_input = processed_input.float()
            processed_input.requires_grad = True
            pred = self.predict_wrapper(processed_input)
            lig = LayerIntegratedGradients(self.predict_wrapper, self.model.model)
            baseline = torch.zeros_like(processed_input).to(processed_input.device)
            baseline[:,-1] = 1
            _, predicted_class = torch.max(pred, 1)

            attributions = lig.attribute(processed_input, baseline, target=predicted_class, n_steps=5)
            
            aug_batch = augmentation_batch_ig(batch, n_aug=self.n_aug, attributions=attributions, aug_list=["ig_random_substitute"])
            pred = self.predict(aug_batch, all_loss, metric)
            proteins = batch["graph"]
            pro_lis = [pro for pro in proteins]
            pro_lis = pro_lis + pro_lis
            duplicate_proteins = data.Protein.pack(pro_lis)
            duplicate_batch = copy.deepcopy(batch)
            duplicate_batch["graph"] = duplicate_proteins
            target = self.target(duplicate_batch)
        elif self.aug == "aug":
            breakpoint()
            proteins = batch["graph"]
            pro_lis = [pro for pro in proteins]
            pro_lis = pro_lis + pro_lis
            duplicate_proteins = data.Protein.pack(pro_lis)
            duplicate_batch = copy.deepcopy(batch)
            duplicate_batch["graph"] = duplicate_proteins
            target = self.target(duplicate_batch)

            aug_batch = augmentation_batch(batch, self.n_aug, sub_policy, self.aug_list)

            pred = self.predict(aug_batch, all_loss, metric)
        else:
            pred = self.predict(batch, all_loss, metric)
            target = self.target(batch)

        for criterion, weight in self.criterion.items():
            if criterion == "bce":
                loss = F.binary_cross_entropy_with_logits(pred, target["label"], reduction="none")
                loss = functional.variadic_mean(loss * target["mask"].float(), size=target["size"])
            else:
                raise ValueError("Unknown criterion `%s`" % criterion)
            loss = loss.mean()

            name = tasks._get_criterion_name(criterion)
            metric[name] = loss
            all_loss += loss * weight

        return all_loss, metric

    def predict(self, batch, all_loss=None, metric=None):
        graph = batch["graph"]
        output = self.model(graph, graph.residue_feature.float(), all_loss=all_loss, metric=metric)
        output = output["residue_feature"]
        if self.batchnorm == True:
            output = self.bn(output)
            # print("batch")

        range = torch.arange(graph.num_residue, device=self.device)
        node_in, node_out = functional.variadic_meshgrid(range, graph.num_residues, range, graph.num_residues)
        if all_loss is None and node_in.shape[0] > (self.max_length ** 2) * graph.batch_size:
            # test
            # split large input to reduce memory cost
            size = (self.max_length ** 2) * graph.batch_size
            node_in_splits = node_in.split(size, dim=0)
            node_out_splits = node_out.split(size, dim=0)
            pred = []
            for _node_in, _node_out in zip(node_in_splits, node_out_splits):
                prod = output[_node_in] * output[_node_out]
                diff = (output[_node_in] - output[_node_out]).abs()
                pairwise_features = torch.cat((prod, diff), -1)
                _pred = self.mlp(pairwise_features)
                pred.append(_pred)
            pred = torch.cat(pred, dim=0)
        else:
            prod = output[node_in] * output[node_out]
            diff = (output[node_in] - output[node_out]).abs()
            pairwise_features = torch.cat((prod, diff), -1)
            pred = self.mlp(pairwise_features)

        return pred.squeeze(-1)
    
    def predict_wrapper(self, processed_input):
        output = self.model(None, processed_input, None, None, True)
        output = output["residue_feature"]
        if self.batchnorm == True:
            output = self.bn(output)
            # print("batch")

        range = torch.arange(graph.num_residue, device=self.device)
        node_in, node_out = functional.variadic_meshgrid(range, graph.num_residues, range, graph.num_residues)
        if all_loss is None and node_in.shape[0] > (self.max_length ** 2) * graph.batch_size:
            # test
            # split large input to reduce memory cost
            size = (self.max_length ** 2) * graph.batch_size
            node_in_splits = node_in.split(size, dim=0)
            node_out_splits = node_out.split(size, dim=0)
            pred = []
            for _node_in, _node_out in zip(node_in_splits, node_out_splits):
                prod = output[_node_in] * output[_node_out]
                diff = (output[_node_in] - output[_node_out]).abs()
                pairwise_features = torch.cat((prod, diff), -1)
                _pred = self.mlp(pairwise_features)
                pred.append(_pred)
            pred = torch.cat(pred, dim=0)
        else:
            prod = output[node_in] * output[node_out]
            diff = (output[node_in] - output[node_out]).abs()
            pairwise_features = torch.cat((prod, diff), -1)
            pred = self.mlp(pairwise_features)

        return pred.squeeze(-1)

    def target(self, batch):
        graph = batch["graph"]
        valid_mask = graph.mask
        residue_position = graph.residue_position

        range = torch.arange(graph.num_residue, device=self.device)
        node_in, node_out = functional.variadic_meshgrid(range, graph.num_residues, range, graph.num_residues)
        dist = (residue_position[node_in] - residue_position[node_out]).norm(p=2, dim=-1)
        label = (dist < self.threshold).float()

        mask = valid_mask[node_in] & valid_mask[node_out] & ((node_in - node_out).abs() >= self.gap)

        return {
            "label": label,
            "mask": mask,
            "size": graph.num_residues ** 2
        }

    def evaluate(self, pred, target):
        label = target["label"]
        mask = target["mask"]
        size = functional.variadic_sum(mask.long(), target["size"])
        label = label[mask]
        pred = pred[mask]

        metric = {}
        for _metric in self.metric:
            if _metric == "accuracy":
                score = (pred > 0) == label
                score = functional.variadic_mean(score.float(), size).mean()
            elif _metric.startswith("prec@L"):
                l = target["size"].sqrt().long()
                k = int(_metric[7:]) if len(_metric) > 7 else 1
                l = torch.div(l, k, rounding_mode="floor")
                score = metrics.variadic_top_precision(pred, label, size, l).mean()
            elif _metric.startswith("prec@"):
                k = int(_metric[5:])
                k = torch.full_like(size, k)
                score = metrics.variadic_top_precision(pred, label, size, k).mean()
            else:
                raise ValueError("Unknown criterion `%s`" % _metric)

            name = tasks._get_metric_name(_metric)
            metric[name] = score

        return metric
