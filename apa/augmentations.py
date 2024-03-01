import numpy as np
import torch
import random
import math
import time
import copy
from torchdrug import data

residue_symbol2id = {"G": 0, "A": 1, "S": 2, "P": 3, "V": 4, "T": 5, "C": 6, "I": 7, "L": 8, "N": 9,
                         "D": 10, "Q": 11, "K": 12, "E": 13, "M": 14, "H": 15, "F": 16, "R": 17, "Y": 18, "W": 19}

genetic_code_protein2mRNA = {
    "F": ["TTT", "TTC"],
    "L": ["TTA", "TTG", "CTT", "CTC", "CTA", "CTG"],
    "I": ["ATT", "ATC", "ATA"],
    "M": ["ATG"],
    "V": ["GTT", "GTC", "GTA", "GTG"],
    "S": ["TCT", "TCC", "TCA", "TCG", "AGT", "AGC"],
    "P": ["CCT", "CCC", "CCA", "CCG"],
    "T": ["ACT", "ACC", "ACA", "ACG"],
    "A": ["GCT", "GCC", "GCA", "GCG"],
    "Y": ["TAT", "TAC"],
    "H": ["CAT", "CAC"],
    "Q": ["CAA", "CAG"],
    "N": ["AAT", "AAC"],
    "K": ["AAA", "AAG"],
    "D": ["GAT", "GAC"],
    "E": ["GAA", "GAG"],
    "C": ["TGT", "TGC"],
    "W": ["TGG"],
    "R": ["CGT", "CGC", "CGA", "CGG", "AGA", "AGG"],
    "G": ["GGT", "GGC", "GGA", "GGG"],
    "Stop": ["TAA", "TAG", "TGA"]
}

genetic_code_mRNA2protein = {
    "TTT": "F", "TTC": "F",
    "TTA": "L", "TTG": "L", "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I",
    "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S", "AGT": "S", "AGC": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y",
    "CAT": "H", "CAC": "H",
    "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N",
    "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D",
    "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C",
    "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
    "TAA": "Stop", "TAG": "Stop", "TGA": "Stop"
}

def random_insert(seq, m):
    """
    Randomly insert m gaps into the sequence
    :param seq (list): sequence
    :param m (double): magnitude of operation, here is the proportion of insertions
    """
    seq_len = len(seq)
    insert_positions = random.sample(range(seq_len), int(m * seq_len))
    for pos in insert_positions:
        seq.insert(pos, random.choice(list(residue_symbol2id.keys())))
    return seq

def random_substitute(seq, m):
    """
    Randomly substitute m residues in the sequence
    :param seq (list): sequence
    :param m (double): magnitude of operation, here is the proportion of subsitutions
    """
    seq_len = len(seq)
    substitute_positions = random.sample(range(seq_len), int(m * seq_len))
    for pos in substitute_positions:
        seq[pos] = random.choice(list(residue_symbol2id.keys()))
    return seq

def random_swap(seq, m):
    """
    Randomly swap m residues in the sequence
    :param seq (list): sequence
    :param m (double): magnitude of operation, here is the proportion of swaps
    """
    seq_len = len(seq)
    for _ in range(int(m * seq_len)):
        pos1, pos2 = random.sample(range(seq_len), 2)
        seq[pos1], seq[pos2] = seq[pos2], seq[pos1]
    return seq

def random_delete(seq, m):
    """
    Randomly delete m residues in the sequence
    :param seq (list): sequence
    :param m (double): magnitude of operation, here is the proportion of deletions
    """
    seq_len = len(seq)
    seq = [residue for residue in seq if random.random() > m]
    return seq

def random_crop(seq, m):
    """
    Randomly crop m consecutive residues in the sequence
    :param seq (list): sequence
    :param m (double): magnitude of operation, here is the proportion of crops
    """
    seq_len = len(seq)
    crop_start = random.randint(0, seq_len - int(m * seq_len))
    crop_end = crop_start + int(m * seq_len)
    seq = seq[crop_start:crop_end]
    return seq

def random_shuffle(seq, m):
    """
    Randomly shuffle m consecutive residues in the sequence
    :param seq (list): sequence
    :param m (double): magnitude of operation, here is the proportion of shuffles
    """
    seq_len = len(seq)
    shuffle_start = random.randint(0, seq_len - int(m * seq_len))
    shuffle_end = shuffle_start + int(m * seq_len)
    random.shuffle(seq[shuffle_start:shuffle_end])
    return seq

def global_reverse(seq, m):
    """
    Reverse the sequence
    :param seq (list): sequence
    """
    seq.reverse()
    return seq

def random_cut(seq, m):
    """
    Randomly cut the sequence into m * 10 parts and randomly assemble them
    :param seq (list): sequence
    :param m (double): magnitude of operation, here is the proportion of cuts
    """
    seq_len = len(seq)
    if seq_len < 20:
        return seq
    cut_positions = random.sample(range(seq_len), max(int(m * 10), 1))
    cut_positions.sort()
    cut_positions.append(seq_len)
    seq_list = []
    for i in range(len(cut_positions) - 1):
        seq_list.append(seq[cut_positions[i]:cut_positions[i + 1]])
    random.shuffle(seq_list)
    seq = []
    for i in range(len(seq_list)):
        seq.extend(seq_list[i])
    return seq

def random_subsequence(seq, m):
    """
    Randomly cut the sequence into 10 parts and randomly assemble m * 10 of them
    :param seq (list): sequence
    :param m (double): magnitude of operation, here is the proportion of subsequence
    """
    seq_len = len(seq)
    if seq_len < 20:
        return seq
    cut_positions = random.sample(range(seq_len), max(1, int(m * 10)))
    cut_positions.sort()
    cut_positions.append(seq_len)
    seq_list = []
    for i in range(len(cut_positions) - 1):
        seq_list.append(seq[cut_positions[i]:cut_positions[i + 1]])
    seq_list = random.sample(seq_list, max(int(m * 10), 1))
    seq = []
    for i in range(len(seq_list)):
        seq.extend(seq_list[i])
    return seq

def back_translation_substitute(seq, m):
    """
    Back translate the sequence into mRNA and randomly substitude m nucleotides and translate back to protein
    :param seq (list): sequence
    :param m (double): magnitude of operation, here is the proportion of substitutions
    """
    seq_len = len(seq)
    mRNA = []
    for residue in seq:
        mRNA.extend(random.sample(genetic_code_protein2mRNA[residue], 1)[0])
    # assemble the mRNA into str
    mRNA = "".join(mRNA)
    mRNA = list(mRNA)
    mRNA_len = len(mRNA)
    substitute_positions = random.sample(range(mRNA_len), int(m * mRNA_len))
    for pos in substitute_positions:
        random_nucleotides = random.choice(["T", "C", "A", "G"])
        # check if the substitution produces stop codon
        pre_mRNA = mRNA[pos]
        mRNA[pos] = random_nucleotides
        if ".".join(mRNA[pos - pos % 3:pos - pos % 3 + 3]) in genetic_code_protein2mRNA["Stop"]:
            mRNA[pos] = pre_mRNA
    seq = []
    for i in range(0, len(mRNA), 3):
        codon = "".join(mRNA[i:i + 3])
        seq.append(genetic_code_mRNA2protein[codon])
    return seq

def _find_repeat_sequence(seq):
    """
    Find the repeat subsequence in the sequence
    :param seq (list): sequence
    """
    seq_len = len(seq)
    frequency_map = {}
    
    # Using a sliding window approach to find all repeating subsequences
    for length in range(2, int(math.sqrt(seq_len))):
        for i in range(seq_len - length + 1):
            subseq = "".join(seq[i:i + length])
            frequency_map[subseq] = frequency_map.get(subseq, 0) + 1
            
    # Sort by frequency and find the most frequent subsequence
    sorted_items = sorted(frequency_map.items(), key=lambda x: x[1], reverse=True)
    if sorted_items and sorted_items[0][1] > 1:  # Ensure that the subsequence repeats more than once
        return sorted_items[0][0]
    return None

def repeat_expansion(seq, m):
    """
    Find the most frequent repeat subsequence and expand it following itself in the original sequence by m times.
    :param seq (list): sequence
    :param m (double): magnitude of operation, here is the proportion of expansions
    """
    repeat_subsequence = _find_repeat_sequence(seq)
    if repeat_subsequence is None:
        return seq
    
    repeat_subsequence_len = len(repeat_subsequence)
    seq_len = len(seq)
    repeat_positions = []
    
    # Find positions of the repeat subsequence
    for i in range(seq_len - repeat_subsequence_len + 1):
        if "".join(seq[i:i + repeat_subsequence_len]) == repeat_subsequence:
            repeat_positions.append(i)
    
    repeat_positions = random.sample(repeat_positions, int(m * len(repeat_positions)))
    # Add the repeated subsequences
    for pos in sorted(repeat_positions, reverse=True):  # Reverse, to not affect the following positions
            seq = seq[:pos + repeat_subsequence_len] + list(repeat_subsequence) + seq[pos + repeat_subsequence_len:]
    
    return seq

def repeat_contraction(seq, m):
    """
    Find the most frequent repeat subsequence and contract them in proportion of m.
    :param seq (list): sequence
    :param m (double): magnitude of operation, here is the proportion of contractions
    """
    repeat_subsequence = _find_repeat_sequence(seq)
    if repeat_subsequence is None:
        return seq
    
    repeat_subsequence_len = len(repeat_subsequence)
    seq_len = len(seq)
    repeat_positions = []
    
    # Find positions of the repeat subsequence
    for i in range(seq_len - repeat_subsequence_len + 1):
        if "".join(seq[i:i + repeat_subsequence_len]) == repeat_subsequence:
            repeat_positions.append(i)

    # Randomly select the positions to remove
    repeat_positions = random.sample(repeat_positions, int(m * len(repeat_positions)))    

    # Remove the repeated subsequences
    for pos in sorted(repeat_positions, reverse=True):  # Reverse, to not affect the following positions
        seq = seq[:pos] + seq[pos + repeat_subsequence_len:]
    return seq

def ig_random_substitute(seq, m, indices):
    replace_pos = random.sample(indices, int(m * len(indices)))
    # print(len(seq), replace_pos)
    for index in replace_pos:
        seq[index] = random.choice(list(residue_symbol2id.keys())) 
    return seq

def augment_list():
    l = [
        (random_insert, 0.0, 0.5),
        (random_substitute, 0.0, 0.5),
        (random_swap, 0.0, 0.5),
        (random_delete, 0.0, 0.5),
        (random_crop, 0.4, 1.0),
        (random_shuffle, 0.0, 0.5),
        (global_reverse, 0.0, 0.0),
        (random_cut, 0.2, 1.0),
        (random_subsequence, 0.2, 1.0),
        (back_translation_substitute, 0.0, 0.5),
        (repeat_expansion, 0.0, 1.0),
        (repeat_contraction, 0.0, 1.0),
        (ig_random_substitute, 0.0, 0.5)
    ]
    return l
augment_dict = {fn.__name__: (fn, v1, v2) for fn, v1, v2 in augment_list()}

def get_augment(name):
    return augment_dict[name]

def apply_augment(seq, policy):
    sub_policy = random.choice(policy)
    for name, pr, level in sub_policy:
        if random.random() > pr:
            continue
        augment_fn, low, high = get_augment(name)
        seq = augment_fn(seq, level * (high - low) + low)
    return seq


def random_augment(seq, indices=None, sub_policy=None, aug_list=None, aug_num=None):
    if sub_policy is None:
        if aug_list is None:
            if indices is None:
                augment_fn, low, high = random.choice(augment_list()[:-1])
            else:
                op_list = augment_list()
                if aug_num is not None:
                    op_list = op_list[:aug_num]
                augment_fn, low, high = random.choice(op_list)
        else:
            augment_fn, low, high = random.choice([augment_dict[name] for name in aug_list])
        level = random.random()
        level = level * (high - low) + low
        if augment_fn == ig_random_substitute:
            return augment_fn(seq, level, indices)
        else:
            return augment_fn(seq, level)
    else:
        # if there is ig random substitute in sub_policy, then swap the indices in sub policy, making it be the first augmentation in sub_policy
        for i, (name, pr, level) in enumerate(sub_policy):
            if name == "ig_random_substitute":
                sub_policy[0], sub_policy[i] = sub_policy[i], sub_policy[0]
                break
        for i, (name, pr, level) in enumerate(sub_policy):
            if random.random() > pr:
                continue
            augment_fn, low, high = get_augment(name)
            if augment_fn == ig_random_substitute:
                seq = augment_fn(seq, level * (high - low) + low, indices)
            else:
                seq = augment_fn(seq, level * (high - low) + low)
        return seq

def transform(item, policy):
    graph = item["graph"]
    device = graph.device
    sequences = graph.to_sequence()
    if isinstance(sequences, str):
        sequences = sequences.replace(".", "")
        sequence_list = [list(sequences)]
        sequences = [sequences]
    elif isinstance(sequences, list):
        sequences = [seq.replace(".", "") for seq in sequences]
        sequence_list = [list(seq) for seq in sequences]
    for seq in sequence_list:
        tmp_seq = copy.deepcopy(seq)
        tmp_seq = apply_augment(tmp_seq, policy)
        sequences.append("".join(tmp_seq))
    graph = data.PackedProtein.from_sequences(sequences, atom_feature=None, bond_feature=None)
    graph = graph.to(device)
    aug_item = copy.deepcopy(item)
    aug_item["graph"] = graph
    return aug_item