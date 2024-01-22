#!/usr/bin/env python3
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

class WNNlayer(nn.Module):
    def __init__(self, input_dimension, state_dimension, out_dimension=1,
                 state_activation=nn.functional.relu, out_activation=None):
        super().__init__()
        self.input_dimension = input_dimension
        self.state_dimension = state_dimension
        self.out_dimension = out_dimension
        self.state_activation = state_activation
        self.out_activation = out_activation
        
        self.state_layer = nn.Linear(in_features=input_dimension, out_features=state_dimension)
        self.out_layer = nn.Linear(in_features=state_dimension, out_features=out_dimension)
        
    def forward(self, inputs):   
        state = self.state_layer(inputs)
        if self.state_activation is not None:
            state = self.state_activation(state)
        
        output = self.out_layer(state)
        if self.out_activation is not None:
            output = self.out_activation(output)
        
        return state, output

class WNNBase(nn.Module):
    def __init__(self, input_dimension, output_dimension, state_neurons, out_neurons, n_layers=None,
                 state_activation=nn.functional.relu, out_activation=None, output_activation=None):
        super().__init__()
        
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.state_neurons = state_neurons
        self.out_neurons = out_neurons
        
        if not hasattr(state_neurons, "__len__") and n_layers is not None:
            self.state_neurons = [state_neurons]*n_layers
        else:
            raise AssertionError('Improper state_neurons dimensions')
        
        if not hasattr(out_neurons, "__len__") and n_layers is not None:
            self.out_neurons = [out_neurons]*(n_layers+1)
        else:
            raise AssertionError('Improper out_neurons dimensions')
        
        if len(self.out_neurons) != len(self.state_neurons)+1:
            raise AssertionError('Improper out_neurons or state_neurons dimensions')
        
        self.state_activation = state_activation
        self.out_activation = out_activation
        self.output_activation = output_activation
        
        self.init_layer = nn.Linear(in_features=self.input_dimension, out_features=self.out_neurons[0])
        self.state_inputs = self.state_neurons.copy()[:-1]
        self.state_inputs.insert(0,self.input_dimension)
        self.layers = nn.ModuleList([WNNlayer(self.state_inputs[i], self.state_neurons[i], self.out_neurons[i+1], self.state_activation, self.out_activation)
                                     for i in range(len(self.state_neurons))])
        self.final = nn.Linear(in_features=sum(self.out_neurons), out_features=self.output_dimension)
        
    def forward(self, inputs):
        outputs = []
        
        out = self.init_layer(inputs)   
        if self.out_activation is not None:
            out = self.out_activation(out)
        outputs.append(out)
        state = inputs
        for i, layer in enumerate(self.layers):
            state, out = layer(state)
            outputs.append(out)
        
        outputs = torch.cat(outputs, dim=-1)
        
        output = self.final(outputs)
        if self.output_activation is not None:
            output = self.output_activation(output)
        
        return output


class WNNVector(nn.Module):
    def __init__(self, input_dimension, output_dimension, state_neurons, n_layers,
                 state_activation=nn.functional.relu, out_activation=None, output_activation=None):
        super().__init__()
        
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        
        if hasattr(state_neurons, "__len__"):
            raise AssertionError('Improper state_neurons dimensions')
        if n_layers is None:
            raise AssertionError('Improper dimensions')
        
        self.state_neurons = [state_neurons]*n_layers
        self.state_activation = state_activation
        self.out_activation = out_activation
        self.output_activation = output_activation
        
        self.state_inputs = self.state_neurons.copy()[:-1]
        self.state_inputs.insert(0,self.input_dimension)
        self.layers = nn.ModuleList([nn.Linear(in_features=self.state_inputs[i], out_features=self.state_neurons[i])
                                     for i in range(len(self.state_neurons))])
        self.final = nn.Linear(in_features=state_neurons, out_features=self.output_dimension)
        
        self.gate_layer = nn.Linear(in_features=state_neurons+input_dimension, out_features=state_neurons)
        
    def forward(self, inputs):
        current_state = torch.zeros((inputs.shape[0], self.state_neurons[0]), dtype=torch.float32)
        state = inputs
        
        for i, layer in enumerate(self.layers):
            gate = nn.functional.sigmoid(self.gate_layer(torch.cat((current_state, inputs), dim=-1)))
            state = layer(state)
            current_state = (1-gate)*current_state + gate*state
        
        output = self.final(current_state)
        return output
    
    
class BasicNN(nn.Module):
    def __init__(self, input_dimension, output_dimension, neurons, n_layers=None,
                 activation=nn.functional.relu, output_activation=None):
        super().__init__()
        
        self.input_dimension = input_dimension
        self.neurons = neurons
        
        if not hasattr(neurons, "__len__") and n_layers is not None:
            self.neurons = [neurons]*n_layers
        
        self.activation = activation
        self.output_activation = output_activation
        self.neurons = [input_dimension] + self.neurons + [output_dimension]
        
        self.layers = nn.ModuleList([nn.Linear(in_features=self.neurons[i], out_features=self.neurons[i+1]) for i in range(len(self.neurons)-1)])
        
    def forward(self, inputs):
        x = inputs
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == len(self.layers)-1 and self.output_activation is not None:
                x = self.output_activation(x)
            else:
                x = self.activation(x)
        
        return x
        
        
        