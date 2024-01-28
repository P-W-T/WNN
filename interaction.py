#!/usr/bin/env python3

import pandas as pd
import itertools
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import copy

def neural_interaction_detection(weight_matrices):
    weight_matrices = copy.deepcopy(weight_matrices)
    assert weight_matrices[-1].shape[0]==1, "The last weight matrix should have a single output."
    
    base = torch.abs(weight_matrices[-1])
    
    for i in range((len(weight_matrices)-2), 0, -1):
        base = torch.matmul(base, torch.abs(weight_matrices[i]))
       
    ending = weight_matrices[0].numpy()
    
    assert base.shape[1] == ending.shape[0], "The final base does not have the right dimensions"
    
    base = base.view(-1)
    result_dict = {}
    for r in range(ending.shape[0]): #loop over the neurons in the first layer
        idx = np.argsort(ending[r,:])[::-1]
        z = base[r].item()
        for j in range(2, ending.shape[1]):
            score = z*np.min(ending[r,idx[:j]])
            idx_str = '-'.join(map(str, np.sort(idx[:j])))
            if idx_str in result_dict.keys():
                result_dict[idx_str] += score
            else:
                result_dict[idx_str] = score
    return result_dict

def neural_interaction_detection_basic(BasicNN):
    NN_layers = [BasicNN.layers[i].weight.detach()
             for i in range(len(BasicNN.layers))]
    return neural_interaction_detection(NN_layers)

def get_main_effects(WNNBase):
    effects = WNNBase.init_layer.weight.detach().numpy()
    return effects.reshape(-1)

def get_track_effects(WNNBase, level):
    assert level < len(WNNBase.layers), "level of track is too high"
    
    weight_matrices = [WNNBase.layers[i].state_layer.weight.detach() for i in range(level+1)]
    weight_matrices.append(WNNBase.layers[level].out_layer.weight.detach())
    return neural_interaction_detection(weight_matrices)

def get_final_effects(WNNBase):
    effects = WNNBase.final.weight.detach().numpy()
    return effects.reshape(-1)

def merge_dict(dict1, dict2):
    return {key: dict1.get(key, 0) + dict2.get(key, 0) for key in set(dict1) | set(dict2)}

def scale_abs_dict(original_dict, constant):
    return {key: abs(value * constant) for key, value in original_dict.items()}

def analyze_WNNBase(WNNBase):
    n_layers = len(WNNBase.layers)
    final_effects = get_final_effects(WNNBase)
    
    overall_dict = {}
    
    for layer in range(n_layers):
        weights = get_track_effects(WNNBase, layer)
        weights = scale_abs_dict(weights, final_effects[layer+1])
        
        overall_dict = merge_dict(overall_dict, weights)
    
    main_effects = get_main_effects(WNNBase)*final_effects[0]
    for i in range(len(main_effects)):
        overall_dict[str(i)] = abs(main_effects[i])
    
    return overall_dict

def create_heatmap(interaction_dict, mask=None):
    max_index = max([int(idx) for key in interaction_dict.keys() for idx in key.split('-') if idx.isdigit()])
    interaction_matrix = np.zeros((max_index + 1, max_index + 1))

    for key, value in interaction_dict.items():
        indices = [int(idx) for idx in key.split('-') if idx.isdigit()]
        if len(indices) > 1:
            pairs = list(itertools.combinations(indices, 2))
            value_per_pair = abs(value) / len(pairs)
            for (i, j) in pairs:
                interaction_matrix[i][j] += value_per_pair
                interaction_matrix[j][i] += value_per_pair
        elif len(indices) == 1:
            interaction_matrix[indices[0]][indices[0]] += value

    scale = np.max(np.abs(interaction_matrix))
    interaction_matrix = interaction_matrix/scale
    if mask is not None:
        if mask=='upper':
            interaction_matrix = np.multiply(interaction_matrix, np.triu(np.ones_like(interaction_matrix, dtype=bool),k=0))
        else:
            interaction_matrix = np.multiply(interaction_matrix, np.tril(np.ones_like(interaction_matrix, dtype=bool),k=0))
    sns.heatmap(interaction_matrix, vmin=0, vmax=1, cmap='viridis')
    plt.show()
