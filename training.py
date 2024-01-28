#!/usr/bin/env python3

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import copy
import itertools
from benchmark_functions import generate_data
from sklearn.model_selection import ParameterSampler
from timeit import default_timer as timer
from torch.utils.data import DataLoader, TensorDataset    
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns


def train(model, train_dataset, val_dataset, epochs, batch_size, early_stopping_patience=50, lr_start=0.1, lr_end=1e-10, betas=(0.9, 0.999), scheduler_patience=10, scheduler_factor=0.5, loss_fn=nn.MSELoss(), device='cpu', verbose=0, train_curve=False):
    
    if train_curve:
        train_list = []
        val_list = []
    
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size)
    
    best_model = copy.deepcopy(model)
    best_loss = np.inf
    current_patience = 0
    
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr_start, betas)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=lr_end, eps=1e-08, verbose=False)
    
    for epoch in range(epochs):
        if verbose > 0:
            start = timer()
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            y_pred = model(X_batch)
            
            optimizer.zero_grad()
            loss = loss_fn(y_pred.view(-1,), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            #print(loss)
        
        val_loss = evaluate(model, val_loader, loss_fn, device)        
        
        scheduler.step(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(model).to('cpu')
            current_patience = 0
        else:
            current_patience += 1
        
        if verbose > 0:
            end = timer()
            current_time = end - start
            minutes = int(current_time/60)
            seconds = current_time - minutes*60
            
            print(f"Epoch {epoch+1}/{epochs} - training loss: {train_loss/len(train_loader):.4f} - val loss: {val_loss/len(val_loader):.4f} - {minutes}m {seconds:2f}s")
        
        if train_curve:
            train_list.append(train_loss/len(train_loader))
            val_list.append(val_loss/len(val_loader))
        
        if current_patience >= early_stopping_patience:
            if train_curve:
                return best_model, best_loss/len(val_loader), epoch, train_list, val_list
            return best_model, best_loss/len(val_loader), epoch
    if train_curve:
        return best_model, best_loss/len(val_loader), epochs, train_list, val_list
    return best_model, best_loss/len(val_loader), epochs

def evaluate(model, test_loader, loss_fn=nn.MSELoss(), device='cpu'):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            y_pred = model(X_batch)

            loss = loss_fn(y_pred.view(-1,), y_batch)
            test_loss += loss.item()
            
    return test_loss

def evaluate_predict(model, test_loader, loss_fn=nn.MSELoss(), device='cpu', return_pred=False):
    model.eval()
    test_loss = 0
    if return_pred:
        full_pred = []
        full_test = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            y_pred = model(X_batch)

            loss = loss_fn(y_pred.view(-1,), y_batch)
            test_loss += loss.item()
            if return_pred:
                full_pred.append(y_pred.numpy())
                full_test.append(y_batch.numpy())
    
    if return_pred:
        return test_loss, np.concatenate(full_pred, axis=0).reshape(-1,), np.concatenate(full_test, axis=0)
    return test_loss

def random_search_val(model, model_params, train_params, other_train_params, num_iter, train_loader, val_loader):
    best_model = copy.deepcopy(model.to('cpu'))
    best_params = None
    best_score = np.inf
    
    # Combine model and training parameters for sampling
    combined_params = {**model_params, **train_params}
    counter = 1
    
    for params_sample in ParameterSampler(combined_params, n_iter=num_iter):
        if verbose > 0:
            start = timer()
        # Separate parameters again after sampling
        model_init_params = {k: params_sample[k] for k in model_params}
        train_hyperparams = {k: params_sample[k] for k in train_params}

        model = model(**model_init_params)
        model, score = train(model, train_loader, val_loader, **train_hyperparams, **other_train_params)
        
        if verbose > 0:
            end = timer()
            current_time = end - start
            minutes = int(current_time/60)
            seconds = current_time - minutes*60
            print(f"Random search {counter}/{num_iter} - Loss: {score:.4f}/{best_score:.4f} - {minutes}m {seconds:2f}s")
            
        if score < best_score:
            best_score = score
            best_params = params_sample
            best_model = copy.deepcopy(model.to('cpu'))
        
        counter += 1

    return best_model, best_params, best_score

def data_fn_generator(data_fn):
    def data_fn(num):
        return generate_data(data_fn, num)
    return data_fn

def param_eval_loop(model_fn, model_param, train_param, data_fn, train_size, val_size, test_size, repeats, random_seeds=None):
    if random_seeds is None:
        random_seeds = [None]*repeats
    
    val_losses = []
    test_losses = []
    epochs = []
    
    for i in range(repeats):
        X_train, y_train = generate_data(train_size, data_fn, random_seeds[i])
        X_val, y_val = generate_data(val_size, data_fn, random_seeds[i])
        X_test, y_test = generate_data(test_size, data_fn, random_seeds[i])
        
        train_dataset = TensorDataset(torch.from_numpy(X_train).to(torch.float32), torch.from_numpy(y_train).to(torch.float32))
        val_dataset = TensorDataset(torch.from_numpy(X_val).to(torch.float32), torch.from_numpy(y_val).to(torch.float32))
        test_dataset = TensorDataset(torch.from_numpy(X_test).to(torch.float32), torch.from_numpy(y_test).to(torch.float32))
        
        test_loader = DataLoader(val_dataset, shuffle=False, batch_size=512)
        
        model = model_fn(**model_param)
        best_model, val_loss, epoch = train(model, train_dataset, val_dataset, **train_param)
        test_loss = evaluate(model, test_loader, loss_fn=train_param['loss_fn'], device=train_param['device'])/len(test_loader)
        
        val_losses.append(val_loss)
        test_losses.append(test_loss)
        epochs.append(epoch)
    
    return val_losses, test_losses, epochs

def grid_search(model_fn, grid_model_params, grid_train_params, data_fn, train_size, val_size, test_size, repeats, random_seeds=None, verbose=0):
    # Store the results
    results = []

    # Create all combinations of model and training parameters
    all_params = [dict(zip(grid_model_params, v)) for v in itertools.product(*grid_model_params.values())]
    all_train_params = [dict(zip(grid_train_params, v)) for v in itertools.product(*grid_train_params.values())]
    
    full_it = len(all_params)*len(all_train_params)
    current=1
    
    # Iterate over all combinations
    for model_param in all_params:
        for train_param in all_train_params:
            if verbose > 0:
                start = timer()
            # Evaluate the current combination of parameters
            val_losses, test_losses, epochs = param_eval_loop(
                model_fn, model_param, train_param, data_fn, train_size, val_size, test_size, repeats, random_seeds
            )

            # Store the results along with the current parameters
            results.append({
                'model_param': model_param,
                'train_param': train_param,
                'val_losses': val_losses,
                'test_losses': test_losses,
                'epochs': epochs,
                'median_val_loss': np.median(val_losses),
                'median_test_loss': np.median(test_losses),
                'median_epochs': np.median(epochs)
            })

            # Print results for each combination (optional)
            if verbose > 0: 
                end = timer()
                current_time = end - start
                minutes = int(current_time/60)
                seconds = current_time - minutes*60
                print(f"Evaluation {current}/{full_it}, Val Losses: {np.median(val_losses)}, Test Losses: {np.median(test_losses)}, Epochs: {np.median(epochs)} - {minutes}m {seconds:2f}s")
                current += 1
            

    results_df = pd.DataFrame(results)
    return results_df

def plot_curves(x_values, y_values, colors, styles, labels):
    """
    Plots multiple curves on a single plot with professional aesthetics.

    :param x_values: List of lists, each containing x-coordinates for a curve.
    :param y_values: List of lists, each containing y-coordinates for a curve.
    :param colors: List of colors, one for each curve.
    :param styles: List of line styles, one for each curve.
    :param labels: List of labels, one for each curve.
    """
    if not (len(x_values) == len(y_values) == len(colors) == len(styles) == len(labels)):
        raise ValueError("All parameter lists must have the same length.")

    sns.set(style="whitegrid")  # Set the seaborn style
    plt.figure(figsize=(10, 6))

    for x, y, color, style, label in zip(x_values, y_values, colors, styles, labels):
        plt.plot(x, y, color=color, linestyle=style, label=label, marker='o')

    plt.xlabel('X-axis', fontsize=14)
    plt.ylabel('Y-axis', fontsize=14)
    plt.title('Multiple Curves Plot', fontsize=16)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()  # Adjusts plot parameters for a nice fit
    plt.show()